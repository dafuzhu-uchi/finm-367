import polars as pl
import numpy as np
from openpyxl import load_workbook
from pathlib import Path
from typing import List
import altair as alt

def print_sheetname(FILE_PATH: Path) -> List:  #@save
    wb = load_workbook(FILE_PATH, read_only=True)
    result = wb.sheetnames
    wb.close()
    return result


def get_tickers(data: pl.DataFrame) -> List[str]:
    return data.select(pl.col(pl.Float64)).columns


def plot_line(
        data: pl.DataFrame, 
        x: str, 
        y: str|List[str]
    ) -> alt.Chart:
    """
    Time series line plot 
    Args
        - data: columns ["Date", "var1", "var2", ...]
        - x: Column name for x
        - y: Column name or name list for y
    """
    df_long = data.unpivot(
        on=y, index=x, variable_name="Variable", value_name="Value"
    )
    chart = alt.Chart(df_long).mark_line().encode(
        x=f'{x}:T',
        y='Value:Q',
        color='Variable:N'
    ).properties(width=500)
    return chart