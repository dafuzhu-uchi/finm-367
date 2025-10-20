import polars as pl
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
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


def ols_replicate(target: pl.Series, factors: pl.DataFrame, freq: int) -> OLSReplication:
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

    return OLSReplication(X=factors.columns, y=target.name, alpha=alpha,
                          betas=beta, residuals=res, r2=r_squared, tracking_error_ann=te)