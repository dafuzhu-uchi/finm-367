from typing import List, Optional, Tuple
import altair as alt
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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



def plot_corr(
    corr_matrix: pl.DataFrame, 
    labels: Optional[List[str]], 
    figsize: Tuple[int]=(10, 8), 
    cmap: str='RdBu_r', 
    annot: bool=True, 
    fmt: str='.0%',
    title: str='Correlation matrix (lower triangle)'
):
    """
    Plot a correlation matrix with circles highlighting max and min correlations.
    
    Parameters:
    -----------
    corr_matrix : array-like
        Correlation matrix (can be full or lower triangular)
    labels : list, optional
        List of labels for the axes
    figsize : tuple, optional
        Figure size (width, height)
    cmap : str, optional
        Colormap name
    annot : bool, optional
        Whether to annotate cells with values
    fmt : str, optional
        Format string for annotations (e.g., '.0%' for percentages)
    title : str, optional
        Plot title
    """
    # Convert to numpy array
    corr = np.array(corr_matrix)
    n = corr.shape[0]
    
    # Create labels if not provided
    if labels is None:
        labels = [f'Var{i+1}' for i in range(n)]
    
    # Create a mask for upper triangle (keep lower triangle only)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.copy()
    corr_masked[mask] = np.nan
    
    # Find min and max correlations (excluding diagonal)
    lower_tri_mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
    lower_tri_values = corr[lower_tri_mask]
    
    min_val = np.min(lower_tri_values)
    max_val = np.max(lower_tri_values)
    
    # Find positions of min and max
    min_pos = np.where((corr == min_val) & lower_tri_mask)
    max_pos = np.where((corr == max_val) & lower_tri_mask)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(corr_masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add annotations
    if annot:
        for i in range(n):
            for j in range(n):
                if not mask[i, j] and i != j:  # Lower triangle, excluding diagonal
                    val = corr[i, j]
                    if fmt == '.0%':
                        text = f'{val*100:.0f}%'
                    else:
                        text = format(val, fmt.replace('%', ''))
                    ax.text(j, i, text, ha="center", va="center", 
                           color="black" if 0.3 < val < 0.7 else "white",
                           fontsize=10, fontweight='bold')
    
    # Draw rectangles around min and max values
    for i, j in zip(min_pos[0], min_pos[1]):
        rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                linewidth=3, edgecolor='blue', 
                                facecolor='none', zorder=10)
        ax.add_patch(rect)
    
    for i, j in zip(max_pos[0], max_pos[1]):
        rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                linewidth=3, edgecolor='red', 
                                facecolor='none', zorder=10)
        ax.add_patch(rect)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

