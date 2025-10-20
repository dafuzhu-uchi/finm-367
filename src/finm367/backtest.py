from dataclasses import dataclass
import polars as pl
from scipy import stats


@dataclass
class BacktestResults:
    """Container for backtest results"""
    hit_rate: float
    num_hits: int
    num_observations: int
    expected_hit_rate: float
    hit_series: pl.Series
    var_series: pl.Series

    def __repr__(self) -> str:
        return (
            f"BacktestResults(\n"
            f"  Hit Rate: {self.hit_rate:.2%}\n"
            f"  Expected: {self.expected_hit_rate:.2%}\n"
            f"  Hits: {self.num_hits} / {self.num_observations}\n"
            f")"
        )


class Backtest:

    def __init__(
            self,
            data: pl.DataFrame,
            ticker: str = "AAPL",
            loss_prob: float = 0.05
    ) -> None:

        self.data = data
        self.ticker = ticker
        self.q = loss_prob

        if ticker not in self.data.columns:
            raise ValueError(f"Ticker '{ticker}' not found in data")

    def calc_expanding_vol(self, min_periods: int = 26) -> pl.DataFrame:
        result = self.data.with_columns(
            pl.col(self.ticker).shift(1)
            .cumulative_eval(
                pl.element().std()
            )
            .alias("vol_expanding")
        )
        return result[min_periods:, :]

    def calc_rolling_vol(
            self,
            window_size: int = 26,
            min_periods: int = 26
    ) -> pl.DataFrame:
        result = self.data.with_columns(
            pl.col(self.ticker).shift(1)
            .rolling_std(window_size=window_size, min_samples=min_periods)
            .alias("vol_rolling")
        ).drop_nulls()
        return result

    def _calc_var(
            self,
            test_df: pl.DataFrame,
            vol_col: str = "vol_expanding",
            mean_return: float = 0.
    ) -> pl.Series:
        """
        Calculate conditional (normal) VaR value series
        Args:
            - test_df: DataFrame with volatility column
            - vol_col: column name for vol
            - mean_return: typical to ignore
        Return:
            - VaR value over time
        """
        z = stats.norm.ppf(self.q)
        var = test_df.select(
            (pl.lit(mean_return) + z * pl.col(vol_col))
            .alias("var")
        )["var"]
        return var

    def hit_test(
            self,
            method: str = "expanding",
            params: dict = {}
    ) -> BacktestResults:

        window_size, min_periods = params["window_size"], params["min_periods"]

        if method == "expanding":
            test_df = self.calc_expanding_vol(min_periods)
        elif method == "rolling":
            test_df = self.calc_rolling_vol(window_size, min_periods)
        else:
            raise ValueError("Can only enter 'expanding' or 'rolling' for method")

        vol_col = f"vol_{method}"
        var_series = self._calc_var(test_df, vol_col=vol_col, mean_return=0)
        test_df = test_df.with_columns(
            var_series.alias("var"),
        ).with_columns(
            (pl.col(self.ticker) < pl.col("var"))
            .cast(pl.Int32)
            .alias("hit")
        )

        n_hit = test_df.select("hit").sum().to_numpy()[0][0]
        n_obs = test_df.select(pl.len()).cast(pl.Int32).to_numpy()[0][0]
        hit_rate = n_hit / n_obs
        expd_hit_rate = self.q  # prob of loss exceeding q
        hit_series = test_df.select("hit")
        var_series = test_df.select(vol_col)

        return BacktestResults(
            hit_rate, n_hit, n_obs, expd_hit_rate, hit_series, var_series
        )

    def run(self, params):
        results = {}
        results["expanding"] = self.hit_test(method="expanding", params=params)
        results["rolling"] = self.hit_test(method="rolling", params=params)
        return results