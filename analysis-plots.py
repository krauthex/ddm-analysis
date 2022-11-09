#!/usr/bin/env python

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional, Dict, Any, List

import numpy as np
from dfmtoolbox.io import read_data
from scipy.optimize import curve_fit


@dataclass
class AnalysisBlob:
    """Class for bundling analysis data with additional information."""

    data_source: Path
    lags: np.ndarray
    image_structure_function: np.ndarray
    azimuthal_average: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    stardist_details: Optional[List[np.ndarray]] = None


def general_exp(
    x: np.ndarray, amp: float, tau: float, offset: float, beta: float = 1.0
) -> np.ndarray:
    """The general exponential of the shape

        f(x) = amp * exp(-(x/tau)^beta) + offset

    Parameters
    ----------
    x : np.ndarray
        Input data
    amp : float
        The amplitude of the exponential.
    tau : float
        "half-life" of exponential process.
    offset : float
        Offset from zero
    beta : float, optional
        Compressing (< 1.0) or stretching (> 1.0) exponent, by default 1.0

    Returns
    -------
    np.ndarray
        The computed exponential.
    """


# default p0 values for curve_fit
default_p0 = [1.0, 1e3, 0.0]  # amplitude, tau, offset


parser = argparse.ArgumentParser()
parser.add_argument("src", help="The location of source file(s) to analyze.")
parser.add_argument(
    "--dest",
    default="plots/",
    help="The location where to put plots and plot-related data. Default value is 'plots/' relative to src directory.",
)
args = parser.parse_args()

if __name__ == "__main__":
    total_analysis_time = perf_counter()
    print(f":: Starting analysis plots in {args.src}")

    # reading data
    src = Path(args.src)
    if src.name.endswith(".pkl"):  # single file
        plots = src.parent / args.dest
        print(f":: Destination folder is {plots}")

        print(":: Reading data ...")
        blob = read_data(src)
        print(blob.data_source, blob.image_structure_function.shape, blob.notes)

    print(
        f":: Overall plotting & fitting took {perf_counter() - total_analysis_time: .2f} s"
    )
