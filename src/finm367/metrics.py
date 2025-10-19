import polars as pl
import numpy as np
from typing import List, Dict

def select_cols(data: pl.DataFrame, tickers: List[str]) -> pl.DataFrame:
    return data.select(pl.col(tickers))


def calc_moment(data: pl.DataFrame, m: int) -> pl.DataFrame:
    """
    Calculate moment for return dataframe
    Args
        - data: rows as timeseries returns, columns as ticker names
        - m: moment (m >= 3)
    """
    return data.with_columns(
        (
            (pl.all() - pl.all().mean()).pow(m) / (pl.len() - 1)
        ).truediv(
            pl.all().std().pow(m)
        )
    ).sum()


def calc_max_dd(data: pl.DataFrame, ticker: str) -> Dict:   #@save
    """
    Calculate maximum drawdown for an asset
    Args:
        - data: asset returns, columns ['Date', 'tk1', 'tk2', ...]
        - ticker: one single ticker to evaluate
    Returns:
        - Dict:
            - max_drawdown: amount
            - peak: day that reach peak of the max drawdown
            - bottom: day that reach bottom of the max drawdown
            - recover: day recover, if no then "nan"
            - duration: time length to recover, if no then "nan"
    """
    # create new df, calculate cumulative return, peak, bottom
    df = (
        data.select([pl.col(pl.Date), ticker])
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
    bottom_day = dd_period["Date"][0]

    peak_day = df.filter(
        (pl.col("cum_return").eq(peak_value)) & 
        (pl.col("Date").le(bottom_day))
    )["Date"][-1]  # Take the last occurrence before bottom

    recover_day = df.filter(
        pl.col("cum_return").ge(peak_value)
        & (pl.col("Date").gt(bottom_day))
    )[0]["Date"][0]
    
    result = {
        "max_drawdown": max_dd,
        "peak": peak_day,
        "bottom": bottom_day,
        "recover": recover_day,
        "duration": recover_day - peak_day,
        "drawdown": df.select(pl.col(["Date", "drawdown"]))
    }
    return result