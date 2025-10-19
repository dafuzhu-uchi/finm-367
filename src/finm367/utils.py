import polars as pl
from openpyxl import load_workbook
from pathlib import Path
from typing import List, Dict
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


def concat_with_labels(
    dfs: Dict[str, pl.DataFrame] | List[pl.DataFrame],
    label_col: str = "statistic",
    labels: List[str] | None = None,
    label_position: str = "first"
) -> pl.DataFrame:
    """
    Concatenate DataFrames vertically and add a label column.
    
    Parameters
    ----------
    dfs : Dict[str, pl.DataFrame] or List[pl.DataFrame]
        DataFrames to concatenate. If dict, keys are used as labels.
        If list, labels must be provided separately.
    label_col : str, default "statistic"
        Name of the label column to add.
    labels : List[str] or None, default None
        Labels for each DataFrame. Only needed if dfs is a list.
        Ignored if dfs is a dict.
    label_position : str, default "first"
        Position of label column: "first" or "last".
    
    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame with label column.
    """
    # Convert to dict if list is provided
    if isinstance(dfs, list):
        if labels is None:
            raise ValueError("labels must be provided when dfs is a list")
        if len(dfs) != len(labels):
            raise ValueError(f"Number of DataFrames ({len(dfs)}) must match number of labels ({len(labels)})")
        dfs = dict(zip(labels, dfs))
    
    # Add label column to each DataFrame
    labeled_dfs = [
        df.with_columns(pl.lit(label).alias(label_col))
        for label, df in dfs.items()
    ]
    
    # Concatenate vertically
    result = pl.concat(labeled_dfs, how="vertical")
    
    # Reorder columns based on label_position
    if label_position == "first":
        result = result.select([label_col, pl.all().exclude(label_col)])
    elif label_position == "last":
        result = result.select([pl.all().exclude(label_col), label_col])
    else:
        raise ValueError(f"label_position must be 'first' or 'last', got '{label_position}'")
    
    return result