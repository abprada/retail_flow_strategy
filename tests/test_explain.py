import numpy as np
import pandas as pd
import statsmodels.api as sm

from dsr.explain import linear_inference, sage_values, shap_values
from dsr.models import fit


def _panel(days=60, stocks=50, seed=0):
    """Rows of stocks by day, with a common shock each day, so errors are correlated within a day."""
    rng = np.random.default_rng(seed)
    day = np.repeat(pd.bdate_range("2020-01-01", periods=days).to_numpy(), stocks)
    x = rng.normal(size=(days * stocks, 3)).astype("float32")
    x[:, 2] = (rng.random(days * stocks) < 0.2)  # a 0/1 flag
    y = (0.3 * x[:, 0] - 0.1 * x[:, 2] + np.repeat(rng.normal(size=days), stocks) * x[:, 1] + rng.normal(size=len(x)))
    return x, y.astype("float32"), day


def test_ols_coefficients_and_day_clustered_errors_match_statsmodels():
    x, y, day = _panel()
    model = fit("linear", {"alpha": 0.0, "l1_ratio": 0.0}, x, y, None, None)
    beta, se = linear_inference(x, y, day, **model.linear)
    ols = sm.OLS(y.astype(float), sm.add_constant(x.astype(float))).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(day)[0]})
    assert np.allclose(beta, ols.params[1:], rtol=1e-4)
    assert np.allclose(se, ols.bse[1:], rtol=1e-4)


def test_ridge_coefficients_are_the_fitted_ones_shrunk_and_with_errors():
    x, y, day = _panel()
    ols = linear_inference(x, y, day, **fit("linear", {"alpha": 0.0, "l1_ratio": 0.0}, x, y, None, None).linear)
    model = fit("linear", {"alpha": 0.1, "l1_ratio": 0.0}, x, y, None, None)
    beta, se = linear_inference(x, y, day, **model.linear)
    # the coefficients reproduce the model's predictions, up to the intercept
    pred = model.predict(x)
    assert np.allclose(pred - pred.mean(), (x - x.mean(axis=0)) @ beta, atol=1e-4)
    sd = model.linear["sd"]  # shrunk on the scale the penalty applies to
    assert np.linalg.norm(beta * sd) < np.linalg.norm(ols[0] * sd) and np.all(se > 0)
    # an L1 penalty has no standard error formula
    lasso = fit("linear", {"alpha": 1e-3, "l1_ratio": 0.5}, x, y, None, None)
    assert np.isnan(linear_inference(x, y, day, **lasso.linear)[1]).all()


def test_shap_values_of_an_additive_model_are_each_feature_against_the_background():
    rng = np.random.default_rng(2)
    x, background = rng.normal(size=(200, 4)), rng.normal(size=(30, 4))
    w = np.array([0.5, -1.0, 0.0, 2.0])
    shap = shap_values(lambda a: a @ w + np.sin(a[:, 0]), x, background)
    expected = (x - background.mean(axis=0)) * w
    expected[:, 0] += np.sin(x[:, 0]) - np.sin(background[:, 0]).mean()
    assert np.allclose(shap, expected, atol=1e-8)


def test_shap_values_add_up_for_lightgbm_with_linear_leaves():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5000, 4)).astype("float32")
    y = (0.5 * x[:, 0] + np.maximum(x[:, 1], 0) * x[:, 2] + 0.1 * rng.normal(size=5000)).astype("float32")
    params = {"learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 50, "feature_fraction": 1.0,
              "bagging_fraction": 1.0, "lambda_l2": 0.0, "linear_tree": True, "max_trees": 50, "seeds": 2}
    model = fit("lgbm", params, x[:4000], y[:4000], x[4000:], y[4000:])
    background = x[:32]
    shap = shap_values(model.predict, x[4000:4200], background)
    assert np.allclose(shap.sum(axis=1), model.predict(x[4000:4200]) - model.predict(background).mean(), atol=1e-6)


def test_sage_values_credit_the_fit_to_the_feature_that_predicts():
    rng = np.random.default_rng(4)
    x, background = rng.normal(size=(5000, 3)), rng.normal(size=(32, 3))
    y = 2 * x[:, 0] + rng.normal(size=5000)
    values, std = sage_values(lambda a: 2 * a[:, 0], x, y, background, permutations=2000)
    # feature 0 takes the error from about 4 + 1 (predicting the background's average, about 0) down to 1
    assert abs(values[0] - 4) < 4 * std[0] + 0.3 and np.all(values[1:] == 0)
