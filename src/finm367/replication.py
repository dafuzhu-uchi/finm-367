import polars as pl
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
from src.finm367.utils import *
import numpy as np


@dataclass
class OLSReplication:
    X: List[str]
    y: str
    alpha: float
    betas: np.ndarray
    residuals: np.ndarray
    r2: float
    tracking_error_ann: float


def ols_replicate(
        target: pl.Series,
        factors: pl.DataFrame,
        freq: int
    ) -> OLSReplication:
    """
    Perform OLS regression to replicate a target return series using factor exposures.

    Fits a linear model: target = alpha + beta_1*factor_1 + ... + beta_n*factor_n + residual

    Parameters
    ----------
    target : pl.Series
        Target return series to replicate (dependent variable).
    factors : pl.DataFrame
        Factor return series (independent variables). Each column is a factor.
    freq : int
        Number of periods per year (e.g., 12 for monthly, 252 for daily).
        Used to annualize tracking error.

    Returns
    -------
    OLSReplication
        Object containing:
        - alpha: Regression intercept (per-period)
        - betas: Factor loadings/sensitivities
        - residuals: Regression residuals (target - fitted)
        - r2: R-squared (proportion of variance explained)
        - tracking_error_ann: Annualized standard deviation of residuals
    """
    # Transform to numpy
    y_data = target.to_numpy()
    y_mean = np.mean(y_data)
    if factors.shape[1] > 1:
        X_data = factors.to_numpy()
    else:
        X_data = factors.to_numpy().reshape(-1, 1)

    model = LinearRegression().fit(X_data, y_data)
    alpha = model.intercept_
    beta = model.coef_
    y_pred = model.predict(X_data)
    res = y_pred - y_data
    # r squared
    ss_tot = np.sum((y_data - y_mean) ** 2)
    ss_res = np.sum(res ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    # tracking error
    te = np.std(res, ddof=1) * np.sqrt(freq)

    return OLSReplication(
        X=factors.columns, y=target.name, alpha=alpha,
        betas=beta, residuals=res, r2=r_squared, tracking_error_ann=te
    )

def rolling_replication_oos(target, factors, window):
    """
    Parameters
    ----------
    target : pl.Series or pl.DataFrame
        Target return series (dependent variable).
    factors : pl.Series or pl.DataFrame
        Factor return series (independent variables). Each column is a factor.
    window : int

    Returns
    -------
    oos : out-of-sample return list for replication
    """
    target_df = to_frame(target)
    factors_df = to_frame(factors)
    concat_df = pl.concat([target_df, factors_df], how="horizontal")
    t_name = target_df.columns[0]
    cols = factors_df.columns
    oos = []
    for i in range(window, concat_df.shape[0]):
        train = concat_df[i - window: i]
        test = concat_df[i: i + 1]
        y_train = train[t_name]
        X_train = train[cols].to_numpy()
        model = LinearRegression().fit(X_train, y_train)
        X_test = test[cols].to_numpy()
        oos.append(model.predict(X_test)[0])
    return oos



