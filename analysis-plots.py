#!/usr/bin/env python

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional, Dict, Any, List

import matplotlib.pyplot as plt
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


def from_u_to_q(u: np.ndarray, pars: Dict[str, Any]) -> np.ndarray:
    """Calculate wave vectors `q` from pixel unit values `u`. Values of u are assumed to be
    pixel counts of spatial frequencies.

    The supplied `pars` dict has to define the values for magnification and pixel size
    (i.e. physical size of one pixel), as well as image dimensions.

    Returns the array of pixel unit values as wave vectors.
    """
    if pars is None:
        raise RuntimeError("[ERR] Supplied parameters are non-existent.")

    if "magnification" not in pars.keys():
        pars["magnification"] = 1.0
        raise RuntimeWarning(
            "[WARN] Found no magnification, going to use magnification=1."
        )

    if "pixel_size" not in pars.keys():
        pars["pixel_size"] = 1.0
        raise RuntimeWarning(
            "[WARN] Found no pixel size specification, going to use pixel_size=1."
        )

    if "image_size" not in pars.keys():
        raise RuntimeError("[ERR] Supplied parameters don't contain image size.")

    u_min = 1 / (pars["image_size"] * pars["pixel_size"] / pars["magnification"])
    q_min = 2 * np.pi * u_min

    return u * q_min


def analyse_single(src: Path, plots: Path) -> None:

    info_template = (
        "[info] {filename} | shape {shape} | notes {notes} | lag fraction {lag_frac}"
    )

    binary_file_name = src.name.replace(".pkl", "")  # without file extension
    print(f":: Destination folder is {plots}")
    print(":: Reading data ...")
    blob = read_data(src)
    print(
        info_template.format(
            filename=blob.data_source.name,
            shape=blob.azimuthal_average.shape,
            notes=blob.notes,
            lag_frac=blob.metadata["fraction_total_lags"],
        )
    )

    lags = blob.lags
    azimuthal_avg = blob.azimuthal_average
    metadata = blob.metadata

    fps = metadata["fps"]
    pixel_size = metadata["pixel_size"]
    unit = metadata["pixel_size_unit"]

    # convert pixel-u-values to q wave vectors
    u = np.arange(1, metadata["image_size"] // 2 + 1)
    q = from_u_to_q(u, metadata)

    # get representative dt values:
    test_lags = np.linspace(1, len(lags) - 1, num=5, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(6, 6))
    for testlag in test_lags:
        ax.plot(q, azimuthal_avg[testlag], label=r"$\Delta t={}$".format(testlag))
        ax.set_xlabel(r"Wavevectors $q [{}^{{-1}}]$".format(unit))
        ax.set_ylabel("Azimuthal average [a.u.]")
        ax.set_title(f"Azimuthal averages for {binary_file_name}")
        ax.legend()
        ax.grid()
        ax.set_yscale("log")
    fig.savefig(plots / f"az-avg-{binary_file_name}.png", dpi=150)


# default p0 values for curve_fit
default_p0 = [1.0, 1e3, 0.0]  # amplitude, tau, offset

# argpares setup
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
    if src.is_file():  # single file
        plots = src.parent / args.dest
        analyse_single(src, plots)
    else:
        plots = src / args.dest
        for file in [file for file in sorted(src.iterdir()) if file.is_file()]:
            print(f"\n:: working on file {file}")
            analyse_single(file, plots)

    print(
        f":: Overall plotting & fitting took {perf_counter() - total_analysis_time: .2f} s"
    )
