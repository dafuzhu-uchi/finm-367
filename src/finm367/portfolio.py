from locale import normalize

import polars as pl
import numpy as np
from scipy.optimize import minimize
from typing import Literal
from dataclasses import dataclass
from src.finm367.utils import only_numeric, broadcast_rows
from src.finm367.metrics import calc_cov


@dataclass
class Portfolio:
    data: pl.DataFrame
    scale_cov: int=1
    mv_type: str='unconstrained'
    freq: int=52

    @staticmethod
    def normalize(data: pl.DataFrame) -> pl.DataFrame:
        """
        Parameters
        ----------
        data : shape (1, n_cols)
        """
        return data / data.sum_horizontal()

    @property
    def annual_cov_mat(self) -> np.ndarray:
        data = only_numeric(self.data).to_numpy()
        # Calculate covariance matrix
        cov_matrix = np.cov(data.T)
        annual_cov_matrix = cov_matrix * self.freq
        return annual_cov_matrix


    def mv_weights(self) -> pl.DataFrame:
        """
        Mean-variance optimization for weights
        """
        if self.mv_type == 'unconstrained':
            data = only_numeric(self.data)
            covmat_full = calc_cov(data)
            covmat_diag = np.diag(np.diag(covmat_full))
            covmat = self.scale_cov * covmat_full.to_numpy() + (1 - self.scale_cov) * covmat_diag

            weights = np.linalg.solve(covmat, data.mean().to_numpy().reshape(-1, 1))
            weights = weights / weights.sum()

            if data.mean() @ weights < 0:
                weights = -weights

            w = pl.DataFrame({
                col: weights[i] for i, col in enumerate(data.columns)
            })
            return Portfolio.normalize(w)
        elif self.mv_type == 'constrained':
            return self.con_optim_book()
        else:
            raise ValueError(f"Can only choose from 'unconstrained' and 'constrained', got {self.mv_type}")


    def equal_weights(self):
        data = self.data
        n = data.shape[0]
        w = pl.DataFrame({
            col: 1 / n for col in data.columns
        })
        return Portfolio.normalize(w)

    def rp_weights(self):
        """
        Risk parity weights
        """
        data = self.data
        variance = data.std().select(pl.all().pow(2))
        w = pl.DataFrame({
            col: 1 / variance[col] for col in data.columns
        })
        return Portfolio.normalize(w)

    def get_returns(
        self,
        weight_type: Literal["eq","rp","mv"]="eq"
    ) -> pl.Series:
        if weight_type == "eq":
            w = self.equal_weights()
        elif weight_type == "rp":
            w = self.rp_weights()
        else:
            w = self.mv_weights()

        p_rets = broadcast_rows(self.data, w).sum_horizontal().alias("p_rets")
        return p_rets

    def con_optim_book(
            self, target_annual_return: float = 0.20,
            min_weight: float = -0.20,
            max_weight: float = 0.35
    ) -> pl.DataFrame:
        """
        Maximize Sharpe ratio, then rescale to achieve target return.

        This approach:
        1. Optimizes for maximum Sharpe ratio with weights summing to 1
        2. Rescales the portfolio to hit the target return
        3. After rescaling, weights will NOT sum to 1 (leveraged/deleveraged)

        Parameters:
        -----------
        target_annual_return : float
            Target annualized return (e.g., 0.20 for 20%)
        min_weight : float
            Minimum weight for any asset before rescaling (e.g., -0.20 for max 20% short)
        max_weight : float
            Maximum weight for any asset before rescaling (e.g., 0.35 for max 35% long)

        Returns:
        --------
        pl.DataFrame with optimal portfolio weights (rescaled)
        """
        returns_df = only_numeric(self.data)

        # Extract asset columns
        asset_cols = returns_df.columns
        returns_array = returns_df.to_numpy()
        n_assets = len(asset_cols)

        # Calculate mean returns per period and covariance matrix
        mean_returns = np.mean(returns_array, axis=0)
        cov_matrix = np.cov(returns_array.T)

        # Risk-free rate
        rf = 0

        # Objective function: maximize Sharpe ratio (minimize negative Sharpe)
        def sharpe_ratio_neg(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - rf) / portfolio_vol
            return -sharpe

        # Constraint: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        # Set up bounds for each weight
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets

        # Optimize for maximum Sharpe ratio
        result = minimize(
            fun=sharpe_ratio_neg,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )

        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")

        # Get optimal weights from Sharpe optimization
        optimal_weights = result.x

        # Calculate portfolio return before rescaling
        portfolio_return_period = np.dot(optimal_weights, mean_returns)
        portfolio_return_annual = portfolio_return_period * self.freq

        # Rescale weights to achieve target return
        # This breaks the "weights sum to 1" constraint
        scale_factor = target_annual_return / portfolio_return_annual
        rescaled_weights = optimal_weights * scale_factor

        # Create weights dictionary
        weights_dict = {asset: weight for asset, weight in zip(asset_cols, rescaled_weights)}

        return pl.DataFrame(weights_dict)

    def con_optim_mine(
        self, target_annual_return: float = 0.20,
        min_weight: float = -0.20,
        max_weight: float = 0.35
    ) -> pl.DataFrame:
        """
        Solve mean-variance portfolio optimization with constraints.

        This approach:
        Optimizes for maximum Sharpe ratio with weights summing to 1, and
        mean annual return equals 20%


        Parameters:
        -----------
        returns_df : pl.DataFrame
            DataFrame with 'date' column and return columns for each asset
        target_annual_return : float
            Target annualized return (e.g., 0.20 for 20%)
        min_weight : float
            Minimum weight for any asset (e.g., -0.20 for max 20% short)
        max_weight : float
            Maximum weight for any asset (e.g., 0.35 for max 35% long)
        freq : int
            Number of periods per year (52 for weekly, 252 for daily)

        Returns:
        --------
        dict with keys:
            - 'weights': optimal portfolio weights
            - 'annual_return': achieved annualized return
            - 'annual_volatility': annualized portfolio volatility
            - 'sharpe_ratio': Sharpe ratio (assuming risk-free rate = 0)
        """
        returns_df = only_numeric(self.data)
        # Extract asset columns
        asset_cols = returns_df.columns
        returns_array = returns_df.to_numpy()
        n_assets = len(asset_cols)

        # Calculate mean returns (per period) and covariance matrix
        mean_returns = np.mean(returns_array, axis=0)

        # Annualize
        annual_mean_returns = mean_returns * self.freq
        annual_cov_matrix = self.annual_cov_mat

        # Objective function: minimize portfolio variance
        def portfolio_variance(weights):
            return weights.T @ annual_cov_matrix @ weights

        # Constraint: portfolio return equals target
        def return_constraint(weights):
            return weights @ annual_mean_returns - target_annual_return

        # Constraint: weights sum to 1
        def weight_sum_constraint(weights):
            return np.sum(weights) - 1.0

        # Set up constraints
        constraints = [
            {'type': 'eq', 'fun': return_constraint},
            {'type': 'eq', 'fun': weight_sum_constraint}
        ]

        # Set up bounds for each weight
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets

        # Solve optimization problem
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")

        optimal_weights = result.x
        weights_dict = {asset: weight for asset, weight in zip(asset_cols, optimal_weights)}

        return pl.DataFrame(weights_dict)


if __name__ == "__main__":
    TICKS = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'BRK/B', 'LLY']
    TICK_ETF = 'SPY'
    import pandas as pd

    INFILE = '../../data/spx_returns_weekly.xlsx'
    SHEET_INFO = 's&p500 names'
    SHEET_RETURNS = 's&p500 rets'
    SHEET_BENCH = 'benchmark rets'
    info = pd.read_excel(INFILE, sheet_name=SHEET_INFO)
    info.set_index('ticker', inplace=True)
    rets = pd.read_excel(INFILE, sheet_name=SHEET_RETURNS)
    rets.set_index('date', inplace=True)
    rets = rets[TICKS]
    bench = pd.read_excel(INFILE, sheet_name=SHEET_BENCH)
    bench.set_index('date', inplace=True)
    rets[TICK_ETF] = bench[TICK_ETF]
    import polars as pl
    import numpy as np
    from scipy.optimize import minimize

    rets_df = pl.from_pandas(rets.reset_index()).with_columns(pl.col("date").cast(pl.Date))
    FREQ = 52
    TARGET_MEAN = 0.20

    w = Portfolio(data=rets_df, freq=FREQ)
    optim_w = w.optimize_portfolio()
    print("mean", broadcast_rows(only_numeric(rets_df), optim_w).sum_horizontal())
    annual_cov = w.annual_cov_mat
    w_array = optim_w.to_numpy()[0].reshape(-1, 1)
    print(w_array.T @ annual_cov @ w_array)
