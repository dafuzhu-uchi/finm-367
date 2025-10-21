"""
Contain data models for assets, portfolios, etc.
"""
import polars as pl
from src.finm367.utils import *
from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    raw: pl.DataFrame | pl.Series

    @property
    def name(self):
        return self.raw.columns

    @property
    def len(self):
        return self.raw.shape[0]

    def _long_table(self):
        names = self.name
        values = self.raw.to_numpy()[0]
        return pl.DataFrame({
            "Index": names,
            "Values": values
        })

    @property
    def table(self):
        return self._long_table()


@dataclass(frozen=True)
class Group:
    dfs: List[pl.DataFrame] | List[pl.Series]

    @property
    def concat(self):
        df_list = [to_frame(df) for df in self.dfs]
        lens = [df.shape[0] for df in df_list]
        if len(set(lens)) > 1:
            raise ValueError("Rows not aligned, must have same length")
        return pl.concat(df_list, how="horizontal")


@dataclass(frozen=True)
class TimeSeries:
    df: pl.DataFrame | pl.Series
    date: List[datetime]
