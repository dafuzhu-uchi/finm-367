import polars as pl
import numpy as np
from typing import List, Dict, Literal
from sklearn.linear_model import LinearRegression

def select_cols(dfs: pl.DataFrame, tickers: List[str]) -> pl.DataFrame:
    return dfs.select(pl.col(tickers))


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
    return dfs.select(
        pl.col(pl.Float64)
    ).with_columns(
        (
            (pl.all() - pl.all().mean()).pow(m) / (pl.len() - 1)
        ).truediv(
            pl.all().std().pow(m)
        )
    ).sum()


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
    elif isinstance(X, list):
        X_df = dfs.select(pl.col(X)).to_numpy()
    
    beta = LinearRegression().fit(X=X_df, y=y_df).coef_[0]
    beta_df = pl.DataFrame({
        "Tickers": X, f"Beta ({y})": beta
    })
    return beta_df


def calc_reg_metrics(
    data: pl.DataFrame,
    X: str | List[str],
    y: str,
    metrics: List[str] | Literal["all"] = "all",
    freq: int = 52
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
    metrics : List[str] or "all", default "all"
        Metrics to calculate. Options: ["alpha", "beta", "info_ratio", "r_squared"]
        Use "all" for all metrics (only works with single X)
    freq : int, default 52
        Number of periods per year for annualization (52 for weekly, 252 for daily)
    
    Returns
    -------
    pl.DataFrame
        DataFrame with requested regression metrics
        - For single X: one row with all metrics
        - For multiple X: one row per X variable with beta values
    """
    data = data.select(pl.col(pl.Float64))

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
        metrics_to_calc = ["alpha", "beta", "info_ratio", "r_squared"]
    else:
        metrics_to_calc = metrics
    
    # Calculate beta (always needed)
    beta = betas[0]
    
    if "beta" in metrics_to_calc:
        result_dict["beta"] = beta
    
    if any(m in metrics_to_calc for m in ["alpha", "info_ratio", "r_squared"]):
        # Get predictions and residuals
        y_pred = model.predict(X_data)
        residuals = y_data - y_pred
        
        # Alpha
        alpha_per_period = model.intercept_[0]
        alpha_annualized = alpha_per_period * freq
        
        if "alpha" in metrics_to_calc:
            result_dict["alpha"] = alpha_annualized
        
        # Information ratio
        if "info_ratio" in metrics_to_calc:
            residual_std = np.std(residuals, ddof=1)
            if residual_std:
                info_ratio = alpha_annualized / (residual_std * np.sqrt(freq))
            else:
                info_ratio = None
            result_dict["info_ratio"] = info_ratio
        
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
    periods_per_year: int = 52
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
    periods_per_year : int, default 52
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
            freq=periods_per_year
        )
        results.append(result)
    
    # Concatenate all results
    final_df = pl.concat(results, how="vertical")
    
    return final_df


def calc_cov(dfs, columns=None):
    """
    Calculate covariance matrix from a Polars DataFrame.
    
    Parameters:
    -----------
    dfs : pl.DataFrame
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
    if columns is None:
        columns = [col for col in dfs.columns if dfs[col].dtype in 
                   [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
    
    # Select only the specified columns
    df_subset = dfs.select(columns)
    data = df_subset.to_numpy()
    
    # Calculate covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    cov_df = pl.DataFrame({
        col: cov_matrix[i] for i, col in enumerate(columns)
    })
    
    return cov_df