"""What the models learned (explanations/models/walkforward.md): the linear model's coefficients with standard errors
clustered by day, and, for any model, SHAP values of its predictions (shap) and SAGE values of its squared error
(sage-importance)."""

import numpy as np
import sage
import shap


def linear_inference(x, y, day, coef, sd, scale, penalty):
    """The linear model's coefficients in each feature's own units and their standard errors, clustered by day.

    dsr.models fits the linear model on each feature divided by its training standard deviation `sd` and the target
    divided by `scale`, with an unpenalized intercept and a Ridge penalty `penalty` on the sum of squared errors (0 for
    OLS); `coef` is its coefficients on that scale. With the intercept partialled out (features and target centred)
    the coefficients are A^-1 X'y, A = X'X + penalty * I, so their variance is A^-1 (sum over days of g g') A^-1, where
    g sums x_i * residual_i over one day's rows: errors may be correlated across stocks on the same day, not across
    days. The sum is scaled by G / (G - 1) * (n - 1) / (n - k - 1) (G days, n rows, k features), statsmodels'
    small-sample correction. Returns (beta, se), the target's change per unit of each feature; se is NaN for an L1
    penalty (penalty None), which has no such formula."""
    beta = np.asarray(coef, dtype=np.float64) * scale / sd
    if penalty is None:
        return beta, np.full(len(beta), np.nan)
    x = np.asarray(x, dtype=np.float64) / sd
    x -= x.mean(axis=0)
    y = np.asarray(y, dtype=np.float64) / scale
    e = y - y.mean() - x @ np.asarray(coef, dtype=np.float64)
    _, codes = np.unique(np.asarray(day), return_inverse=True)
    g = np.column_stack([np.bincount(codes, weights=x[:, j] * e) for j in range(x.shape[1])])
    (n, k), days = x.shape, len(g)
    bread = np.linalg.pinv(x.T @ x + penalty * np.eye(k))  # pinv: a constant column gets a standard error of 0
    meat = g.T @ g * days / (days - 1) * (n - 1) / (n - k - 1)
    return beta, np.sqrt(np.diag(bread @ meat @ bread)) * scale / sd


def shap_values(predict, x, background, seed=0):
    """SHAP values of each feature for each row of x: what the feature adds to the row's prediction, against the
    average prediction over `background`. shap's PermutationExplainer with an Independent masker: a feature left out
    takes each background row's value in turn (interventional SHAP), and each row's features are added in one random
    order and its reverse. It only calls `predict`, so it works for any model; shap's TreeExplainer reads LightGBM's
    linear leaves as constants and gets them wrong without an error. Each row's values add up to its prediction
    minus the background's average prediction."""
    explainer = shap.PermutationExplainer(predict, shap.maskers.Independent(background, max_samples=len(background)),
                                          seed=seed)
    return explainer(x, max_evals=2 * x.shape[1] + 2, silent=True).values  # one order and its reverse


def sage_values(predict, x, y, background, permutations, seed=0):
    """SAGE values of each feature, with their standard errors: what the feature adds to the fall in squared error
    against y, from the average prediction over `background` to the model's (sage-importance's PermutationEstimator
    with a MarginalImputer; a feature left out takes each background row's value in turn). `permutations` rows are
    drawn from x at random, each with its own random order of the features. The values add up, in expectation, to
    the model's mean squared error gain over predicting the background's average."""
    estimator = sage.PermutationEstimator(sage.MarginalImputer(predict, background), "mse", random_state=seed)
    explanation = estimator(x, y, batch_size=min(permutations, 512), detect_convergence=False,
                            n_permutations=permutations, bar=False)
    return explanation.values, explanation.std
