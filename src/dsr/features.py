"""Raw fields -> one signal value per stock per day.

Every feature is transformed within each day (explanations/models/features.md): winsorized at the
day's 1st and 99th percentiles, then standardized to mean 0 and standard deviation 1; missing values become 0, the day's
centre.
"""

import pandas as pd

LOWER, UPPER = 0.01, 0.99  # fixed in advance, not tuned


def winsorize_standardize(x: pd.DataFrame, day: pd.Series) -> pd.DataFrame:
    """Each column of x winsorized at LOWER and UPPER within each day, standardized within each day, missing -> 0.

    day is aligned with x. A column constant on a day (standard deviation 0) is 0 on that day.
    """
    g = x.groupby(day)
    x = x.clip(g.transform("quantile", LOWER), g.transform("quantile", UPPER))
    g = x.groupby(day)
    z = (x - g.transform("mean")) / g.transform("std")
    return z.where(z.abs() != float("inf")).fillna(0.0)
