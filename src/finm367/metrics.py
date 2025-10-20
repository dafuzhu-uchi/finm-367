import polars as pl
import numpy as np
from typing import List, Dict, Literal
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
import pandas as pd
from finm367.utils import only_numeric, load_path


def select_cols(dfs: pl.DataFrame, tickers: List[str]) -> pl.DataFrame:
    return dfs.select(pl.col(tickers))


# Tail risk: skewness, kurtosis
def calc_moment(dfs: pl.DataFrame, m: int) -> pl.DataFrame:
    """
    Calculate moment for return dataframe

    Parameters
    ----------
    dfs: pl.DataFrame
        rows as timeseries returns, columns as ticker names
    m: int
        moment (m >= 3)
    """

    def _skew(col):
        x = dfs[col].drop_nulls()
        return stats.skew(x, bias=False) if len(x) > 2 else np.nan

    def _ex_kurt(col):
        x = dfs[col].drop_nulls()
        return stats.kurtosis(x, fisher=True, bias=False) if len(x) > 3 else np.nan

    if m == 3:
        return pl.DataFrame({col: _skew(col) for col in dfs.columns})
    else:
        return pl.DataFrame({col: _ex_kurt(col) for col in dfs.columns})


# Tail risk: VaR, cVaR
def var_cvar(data, a=0.05):
    """
    Calculate var/cvar based on monthly returns

    Args:
        data: monthly return
        a: percentile
    Return:
        var(.05) / cvar(.05)
    """
    names = data.schema.names()
    var_list = []
    cvar_list = []
    for name in names:
        var = data[name].quantile(a)
        var_list.append(var)
        cvar = data[name].filter(data[name].le(var)).mean()
        cvar_list.append(cvar)
    result = pl.DataFrame({
        "tickers": names,
        "var": var_list,
        "cvar": cvar_list
    })
    return result


def calc_max_dd(dfs: pl.DataFrame, ticker: str, date_col: str) -> Dict:   #@save
    """
    Calculate maximum drawdown for an asset
    
    Parameters
    ----------
    dfs: pl.DataFrame
        asset returns, columns ['Date', 'tk1', 'tk2', ...]
    ticker: str
        one single ticker to evaluate
    date_col: str
        column name that indicates time
    
    Returns
    -------
    Dict:
        - max_drawdown: amount
        - peak: day that reach peak of the max drawdown
        - bottom: day that reach bottom of the max drawdown
        - recover: day recover, if no then "nan"
        - duration: time length to recover, if no then "nan"
    """
    # create new df, calculate cumulative return, peak, bottom
    df = (
        dfs.select([pl.col(pl.Date), ticker])
        .with_columns(
            (pl.col(ticker) + 1).cum_prod().alias("cum_return")
        )
        .with_columns(
            pl.col("cum_return").cum_max().alias("peak")
        )
        .with_columns(
            (pl.col("cum_return") / pl.col("peak") - 1).alias("drawdown")
        )
    )
    # filter the period where max drawdown is realized
    dd_period = df.filter(
        pl.col("drawdown").eq(pl.col("drawdown").min())
    )
    max_dd = dd_period["drawdown"][0]
    peak_value = dd_period["peak"]
    bottom_day = dd_period[date_col][0]

    peak_day = df.filter(
        (pl.col("cum_return").eq(peak_value)) & 
        (pl.col(date_col).le(bottom_day))
    )[date_col][-1]  # Take the last occurrence before bottom
    

    recover_df = df.filter(
        pl.col("cum_return").ge(peak_value)
        & (pl.col(date_col).gt(bottom_day))
    )
    if recover_df.height > 0:
        recover_day = recover_df[0][date_col][0]
        duration = recover_day - peak_day
    else:
        recover_day = None
        duration = None
    
    result = {
        "max_drawdown": max_dd,
        "peak": peak_day,
        "bottom": bottom_day,
        "recover": recover_day,
        "duration": duration,
        "drawdown": df.select(pl.col([date_col, "drawdown"]))
    }
    return result


def calc_beta(dfs: pl.DataFrame, X: str|List[str], y: str):
    """
    Calculate the beta of y on X (y regressed on X)
    """
    y_df = dfs.select(pl.col(y)).to_numpy().reshape(-1, 1)
    if isinstance(X, str):
        X_df = dfs.select(pl.col(X)).to_numpy().reshape(-1, 1)
    else:
        X_df = dfs.select(pl.col(X)).to_numpy()
    
    model = LinearRegression()
    model.fit(X=X_df, y=y_df)
    beta = model.coef_[0]
    beta_df = pl.DataFrame({
        "Tickers": X, f"Beta ({y})": beta
    })
    return beta_df


def calc_reg_metrics(
    data: pl.DataFrame,
    X: str | List[str],
    y: str,
    freq: int,
    rf: float = 0,
    metrics: List[str] | Literal["all"] = "all"
) -> pl.DataFrame:
    """
    Calculate regression metrics for y regressed on X.
    
    For single X (bivariate): returns alpha, beta, info_ratio, r_squared
    For multiple X (multivariate): returns betas only (no alpha, IR, R²)
    
    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing return data
    X : str or List[str]
        Independent variable(s) - column name(s) for predictor(s)
    y : str
        Dependent variable - column name for outcome
    rf: float
        risk-free rate
    metrics : List[str] or "all", default "all"
        Metrics to calculate.
        Options: ["alpha", "beta", "info_ratio", "treynor", "r_squared"]
        Use "all" for all metrics (only works with single X)
    freq : int, default 52
        Number of periods per year for annualization (52 for weekly, 252 for daily)
    
    Returns
    -------
    pl.DataFrame
        DataFrame with requested regression metrics
        ["alpha", "beta", "info_ratio", "treynor", "r_squared"]
        - For single X: one row with all metrics
        - For multiple X: one row per X variable with beta values
    """
    data = only_numeric(data)

    # Extract y data
    y_data = data.select(pl.col(y)).to_numpy().reshape(-1, 1)
    y_name = y
    
    # Handle single vs multiple X
    is_multivariate = isinstance(X, list) and len(X) > 1
    
    if isinstance(X, str):
        # Single X (bivariate regression)
        X_data = data.select(pl.col(X)).to_numpy().reshape(-1, 1)
        X_names = [X]
    elif isinstance(X, list):
        # Multiple X
        X_data = data.select(X).to_numpy()
        X_names = X
    else:
        raise TypeError(f"X must be str or List[str], got {type(X)}")
    
    # Fit regression
    model = LinearRegression().fit(X_data, y_data)
    
    # Get coefficients
    if X_data.shape[1] == 1:    # bivariate
        betas = [model.coef_[0][0]]
    else:
        betas = model.coef_[0].tolist()
    
    # For multivariate, only return betas
    if is_multivariate:
        result_df = pl.DataFrame({
            "Ticker": X_names,
            f"Beta_{y}": betas
        })
        return result_df
    
    # For bivariate, calculate all requested metrics
    result_dict = {}
    
    # Determine which metrics to calculate
    if metrics == "all":
        metrics_to_calc = ["alpha", "beta", "info_ratio", "treynor", "r_squared"]
    else:
        metrics_to_calc = metrics
    
    # Calculate beta (always needed)
    beta = betas[0]
    
    if "beta" in metrics_to_calc:
        result_dict["beta"] = beta
    
    if any(m in metrics_to_calc for m in ["alpha", "info_ratio", "treynor", "r_squared"]):
        # Get predictions and residuals
        y_pred = model.predict(X_data)
        residuals = y_data - y_pred
        
        # Alpha
        alpha = model.intercept_[0]
        # alpha_annualized = alpha_per_period * freq

        # Treynor
        mean_ann = data.select(pl.col(y)).mean().to_numpy()[0] * freq
        rf_ann = rf * freq
        
        if "alpha" in metrics_to_calc:
            result_dict["alpha"] = alpha * freq
        
        # Information ratio
        if "info_ratio" in metrics_to_calc:
            residual_std = np.std(residuals, ddof=1)
            if residual_std:
                ir = alpha / residual_std
                ir_ann = ir * np.sqrt(freq)
            else:
                ir_ann = np.nan
            result_dict["info_ratio"] = ir_ann

        # Treynor ratio
        if "treynor" in metrics_to_calc:
            treynor = np.nan
            if beta != 0:
                treynor = (mean_ann - rf_ann) / beta
            result_dict["treynor"] = treynor
        
        # R-squared
        if "r_squared" in metrics_to_calc:
            y_mean = np.mean(y_data)
            ss_tot = np.sum((y_data - y_mean) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            result_dict["r_squared"] = r_squared
    
    # Add ticker name
    result_dict = {"X": X_names[0], "y": y_name, **result_dict}
    
    # Create single-row DataFrame
    result_df = pl.DataFrame([result_dict])
    
    return result_df


def calc_reg_metrics_batch(
    data: pl.DataFrame,
    X: str,
    y_cols: List[str],
    metrics: List[str] | Literal["all"] = "all",
    freq: int = 52
) -> pl.DataFrame:
    """
    Calculate regression metrics for multiple y variables against a single X.
    Useful for analyzing multiple assets against a benchmark.
    
    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing return data
    X : str
        Independent variable - column name for predictor (e.g., "SPY")
    y_cols : List[str]
        List of dependent variables - column names for outcomes (e.g., ["AAPL", "NVDA"])
    metrics : List[str] or "all", default "all"
        Metrics to calculate. Options: ["alpha", "beta", "info_ratio", "r_squared"]
    freq : int, default 52
        Number of periods per year for annualization
    
    Returns
    -------
    pl.DataFrame
        DataFrame with one row per y variable, containing all requested metrics
    """
    results = []
    
    for y_col in y_cols:
        result = calc_reg_metrics(
            data=data,
            X=X,
            y=y_col,
            metrics=metrics,
            freq=freq
        )
        results.append(result)
    
    # Concatenate all results
    final_df = pl.concat(results, how="vertical")
    
    return final_df


def calc_cov(data, columns=None):
    """
    Calculate covariance matrix from a Polars DataFrame.
    
    Parameters:
    -----------
    data : pl.DataFrame
        Input DataFrame
    columns : list, optional
        List of column names to include. If None, uses all numeric columns.
    
    Returns:
    --------
    cov_matrix : np.ndarray
        Covariance matrix
    column_names : list
        List of column names in the order they appear in the matrix
    """
    
    # Select numeric columns if not specified
    df_subset = data
    if columns is None:
        df_subset = only_numeric(data)

    df_numpy = df_subset.to_numpy()
    
    # Calculate covariance matrix
    cov_matrix = np.cov(df_numpy, rowvar=False)
    cov_df = pl.DataFrame({
        col: cov_matrix[i] for i, col in enumerate(columns)
    })
    
    return cov_df


if __name__ == "__main__":
    file_name = "proshares_analysis_data_ta.xlsx"
    file_path = load_path(file_name)
    mer_fac = pl.read_excel(file_path, sheet_name="merrill_factors")
    mer_fac = mer_fac[:-1]
    df_pandas = pd.read_excel(file_path, sheet_name="hedge_fund_series", index_col=0)
    df_polars = pl.from_pandas(df_pandas.reset_index()).rename({"index": "Date"}).drop_nulls()
    monthly_ret = df_polars.select(pl.col(pl.Float64))
    concat_df = pl.concat([monthly_ret, pl.DataFrame(mer_fac["SPY US Equity"])], how="horizontal")
    x = calc_reg_metrics(concat_df, X="SPY US Equity", y="MLEIFCTR Index", freq=12)