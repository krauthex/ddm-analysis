"""Plotting related helper functions."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt


def decorate_axes(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    legend: bool = True,
    grid: bool = True,
    grid_args: Optional[Dict[str, str]] = {"which": "both"},
    xscale: str = "log",
    yscale: str = "log",
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    title_fontsize: int = 9,
) -> plt.Axes:
    """Decorate the given plt.Axes instance with labels, title, legend, ..."""

    # labels & title
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # grid & legend
    if grid:
        if grid_args is None:
            ax.grid()
        else:
            # e.g. grid_args = {'which': 'both'} for grid lines on minor ticks as well
            ax.grid(**grid_args)

    if legend:
        ax.legend()

    # scale & limits
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    return ax


def finalize_and_save(fig: plt.Figure, figname: Path, dpi: int = 150) -> None:
    """Finalize figure, save it and close the figure object.

    Parameters
    ----------
    fig : plt.Figure
        The figure to be finalized.
    figname : Path
        Full path where the figure should be saved to.
    dpi : int, optional
        DPI value to be passed to plt.figsave, by default 150
    """
    # checking plot file ending
    if not figname.name.endswith(".png"):
        figname = figname.parent / (figname.name + ".png")

    # cosmetics
    fig.tight_layout()

    fig.savefig(figname, dpi=dpi)
    plt.close(fig)
