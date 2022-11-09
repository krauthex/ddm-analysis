#!/usr/bin/env python

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from dfmtoolbox._dfm_python import (
    azimuthal_average,
    distance_array,
    reconstruct_full_spectrum,
)
from dfmtoolbox.io import read_data
from matplotlib.colors import TABLEAU_COLORS
from scipy.optimize import curve_fit


@dataclass
class AnalysisBlob:
    """Class for bundling analysis data with additional information."""

    data_source: Path
    rfft2: np.ndarray
    lags: np.ndarray
    image_structure_function: np.ndarray
    azimuthal_average: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    stardist_details: Optional[List[np.ndarray]] = None


def general_exp(
    t: np.ndarray, tau: float, amp: float = 1.0, offset: float = 0.0, beta: float = 1.0
) -> np.ndarray:
    """The general exponential of the shape

        f(t) = amp * exp(-(t/tau)^beta) + offset

    Parameters
    ----------
    t : np.ndarray
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
    return amp * np.exp(-((t / tau) ** beta)) + offset


def static_estimate_A_B(rfft2: np.ndarray) -> Tuple[np.ndarray, float]:
    """Estimate the values for A(q) and B from an average of the power spectra of ImageSet FFT2s.

    The values for A(q) and B can be used to determine the shape of the ISF, since

        D(q, dt) = A(q)[1 - ISF(q, dt)] + B

    with D(q, dt) being the image structure function. The value of B is taken as the average of the
    10 last values of two times the average power spectrum, and the sum of A and B is given by:

        A + B = 2 * <|FFT2(I)|^2>
    """

    power_spec = np.abs(rfft2) ** 2  # numpy understands complex numbers
    power_spec = np.array([reconstruct_full_spectrum(im) for im in power_spec])
    power_spec = power_spec.mean(axis=0)
    dist = distance_array(power_spec.shape)
    a_plus_b = azimuthal_average(power_spec, dist)
    a_plus_b *= 2

    B = a_plus_b[-10:].mean()
    A = a_plus_b - B

    return A, B


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


def intermediate_scattering_function(
    structure_function: np.ndarray, A: np.ndarray, B: float
) -> np.ndarray:
    """Return the intermediate scattering function."""
    A_masked = np.ma.masked_equal(A, 0)  # masking 0 values for A

    return 1 - (structure_function - B) / A_masked


def analyse_single(src: Path, plots: Path) -> None:

    colors = list(TABLEAU_COLORS.values())

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
    # pixel_size = metadata["pixel_size"]
    unit = metadata["pixel_size_unit"]

    # convert pixel-u-values to q wave vectors
    u = np.arange(1, metadata["image_size"] // 2 + 1)
    q = from_u_to_q(u, metadata)

    # get representative dt values:
    test_lags = np.linspace(1, len(lags) - 1, num=5, dtype=np.int64)
    time = lags / fps

    # representative u/q values (linspaced)
    test_wv_idc = np.linspace(
        int(len(u) * 0.1), int(len(u) * 0.5), num=5, dtype=np.int64
    )  # test indices for u and q; use only the first half of q range, rest is very noisy
    test_u = u[test_wv_idc]  # get some test u values
    test_q = q[test_wv_idc]

    # plotting azimuthal average
    fig, ax = plt.subplots(figsize=(6, 6))
    for testlag in test_lags:
        ax.plot(q, azimuthal_avg[testlag], label=r"$\Delta t={}$".format(testlag))
    ax.set_xlabel(r"Wavevectors $q\ [{}^{{-1}}]$".format(unit))
    ax.set_ylabel("Azimuthal average [a.u.]")
    ax.set_title(f"Azimuthal avg for {binary_file_name} | {blob.notes}", fontsize=9)
    ax.legend()
    ax.grid()
    ax.set_yscale("log")
    fig.savefig(plots / f"az-avg-{binary_file_name}.png", dpi=150)
    plt.close(fig)  # cleanup

    # calculating A, B, plotting
    A, B = static_estimate_A_B(blob.rfft2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(q, A, label=r"$A(q)$")
    ax.hlines(B, q[0], q[-1], linestyle="--", colors="k", label=r"$B={:.2f}$".format(B))
    ax.set_xlabel(r"Wavevectors $q\ [{}^{{-1}}]$".format(unit))
    ax.set_ylabel(r"$A(q),\ B$ [a.u.]")
    ax.set_title(f"$A(q),\ B$ for {binary_file_name} | {blob.notes}", fontsize=9)
    ax.legend()
    ax.grid()
    ax.set_yscale("log")
    fig.savefig(plots / f"static_AB-{binary_file_name}.png", dpi=150)
    plt.close(fig)

    # fitting exponential
    isf = np.zeros_like(azimuthal_avg)
    for i, avg in enumerate(azimuthal_avg):
        isf[i] = intermediate_scattering_function(avg, A, B)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, tu in enumerate(test_u):
        ax.plot(
            time, isf[:, tu], color=colors[i], label="$q = {:.3f}$".format(test_q[i])
        )

        # popt, pcov = curve_fit(general_exp, time, isf[:, tu], p0=default_p0)
        popt, pcov = curve_fit(general_exp, time, isf[:, tu], p0=default_p0)
        ax.plot(
            time,
            general_exp(time, *popt),
            linestyle=":",
            linewidth=0.8,
            color=colors[i],
        )
        print(popt, np.sqrt(np.diag(pcov)))
        # print(popt, np.sqrt(pcov))

    ax.set_xlabel(r"Time $t\ [s]$")
    ax.set_ylabel(r"Intermediate scattering function $f(q, \Delta t)$")
    ax.set_title(f"ISF for {binary_file_name} | {blob.notes}", fontsize=9)
    ax.legend()
    ax.grid()
    ax.set_xscale("log")
    fig.savefig(plots / f"isf-{binary_file_name}.png", dpi=150)
    plt.close(fig)


# default p0 values for curve_fit
default_p0 = [1e3, 1.0]  # amplitude, tau; offset is always very close to zero anyway

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
