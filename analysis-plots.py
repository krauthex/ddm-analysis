#!/usr/bin/env python

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

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
    t: np.ndarray, tau: float, amp: float = 1.0, beta: float = 1.0
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
    beta : float, optional
        Compressing (< 1.0) or stretching (> 1.0) exponent, by default 1.0

    Returns
    -------
    np.ndarray
        The computed exponential.
    """
    return amp * np.exp(-((t / tau) ** beta))


def tau_moment(tau: float, beta: float) -> float:
    """Calculate the first moment of the stretched exponential function."""
    from scipy.special import gamma

    return tau / beta * gamma(1 / beta)


def power_law(q: np.ndarray, a: float, eta: float) -> np.ndarray:
    """A general power law function of the shape f(q) = a * q^eta.

    Parameters
    ----------
    q : np.ndarray
        Input values.
    a : float
        Constant prefactor.
    eta : float
        Scaling exponent.

    Returns
    -------
    np.ndarray
        The computed power law.
    """
    return a * q**eta


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


def print_blob_info(blob: AnalysisBlob) -> None:
    info_template = (
        "[info] {filename} | shape {shape} | notes {notes} | lag fraction {lag_frac}"
    )

    metadata = blob.metadata if blob.metadata is not None else {}
    print(
        info_template.format(
            filename=blob.data_source.name,
            shape=blob.image_structure_function.shape,
            notes=blob.notes,
            lag_frac=metadata.get("fraction_total_lags"),
        )
    )


def plot_exp_fit_parameters(
    parameters: np.ndarray,
    fit_q: np.ndarray,
    colors: List[str],
    fit_parameter_labels: List[str],
    xunit: str,
) -> Tuple[plt.Figure, plt.Axes]:

    nparameters = len(parameters)

    fig, axes = plt.subplots(
        nparameters, 1, figsize=(6, 2.5 * nparameters), sharex=True
    )

    for i in range(nparameters):
        ax = axes[i]
        par = parameters[i]
        ax.scatter(fit_q, par, c=colors[i], s=20)
        ax.set_ylabel(fit_parameter_labels[i])
        ax.grid(which="both")
        ax.set_xscale("log")

        # fit parameter specific settings
        if i == 0:  # tau
            ax.set_yscale("log")
            ax.set_ylim((500, 1e4))

        elif i == 1:  # amplitude
            ax.set_ylim((0.6, 1.3))

        elif i == 2:  # beta
            ax.set_ylim((0.6, 2))
            # also plot the first moment of tau in the right axis
            axes[0].scatter(
                fit_q,
                tau_moment(parameters[0], par),
                c="k",
                marker="v",
                label=r"$\langle \tau(q) \rangle$",
            )
            axes[0].legend()

    # cosmetics
    ax.set_xlabel(r"Wavevector $q\ [{}^{{-1}}]$".format(xunit))

    return fig, axes


def analyse_single(
    src: Union[Path, List[Path]], plots: Path, ensemble_average: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray]]:

    colors = list(TABLEAU_COLORS.values())

    print(f":: Destination folder is {plots}")

    if ensemble_average and isinstance(src, list):
        # we average the image structure function over the files given and analyse that

        # assume same file naming for simplicity
        binary_file_name = src[0].name.replace(".pkl", "")  # without file extension

        ensemble = []
        print(":: Reading ensemble data ...")
        for path in src:
            blob = read_data(path)
            ensemble.append(blob)
            print_blob_info(blob)

        # use last blob to extract info that should be the same for all blobs
        lags = blob.lags
        metadata = blob.metadata
        notes = blob.notes

        print(":: Averaging ensemble data ...")
        # we will compute our own azimuthal average
        dqt = np.array([blob.image_structure_function for blob in ensemble]).mean(
            axis=0
        )
        rfft2 = np.array([blob.rfft2 for blob in ensemble])

        dqt_dist = distance_array(dqt.shape)
        azimuthal_avg = np.zeros((len(dqt), metadata["image_size"] // 2))
        for i in range(len(dqt)):
            azimuthal_avg[i] = azimuthal_average(dqt[i], dqt_dist)

        # memory cleanup
        del ensemble[:], blob

    else:
        print(":: Reading data ...")
        binary_file_name = src.name.replace(".pkl", "")  # type: ignore
        blob = read_data(src)
        print_blob_info(blob)

        lags = blob.lags
        metadata = blob.metadata
        azimuthal_avg = blob.azimuthal_average
        rfft2 = blob.rfft2
        notes = blob.notes

    fps = metadata["fps"]
    # pixel_size = metadata["pixel_size"]
    unit = metadata["pixel_size_unit"]

    # convert pixel-u-values to q wave vectors
    u = np.arange(1, metadata["image_size"] // 2 + 1)
    q = from_u_to_q(u, metadata)

    # get representative dt values:
    test_lags = np.linspace(1, len(lags), num=5, dtype=np.int64, endpoint=False)
    time = lags / fps

    # representative u/q values (linspaced)
    idx_range = (int(len(u) * 0.1), int(len(u) * 0.5))
    test_wv_idc = np.linspace(
        *idx_range, num=6, dtype=np.int64, endpoint=False
    )  # test indices for u and q; use only the first half of q range, rest is very noisy
    test_u = u[test_wv_idc]  # get some test u values
    test_q = q[test_wv_idc]

    # plotting azimuthal average ##################################################################
    print("::: plotting azimuthal average for some test-lags .. ")
    fig, ax = plt.subplots(figsize=(6, 6))
    for testlag in test_lags:
        ax.plot(q, azimuthal_avg[testlag], label=r"$\Delta t={}$".format(testlag))
    ax.set_xlabel(r"Wavevectors $q\ [{}^{{-1}}]$".format(unit))
    ax.set_ylabel("Azimuthal average [a.u.]")
    ax.set_title(f"Azimuthal avg | {binary_file_name} | {notes}", fontsize=9)
    ax.legend()
    ax.grid()
    ax.set_yscale("log")
    fig.savefig(plots / f"az-avg-{binary_file_name}.png", dpi=150)
    plt.close(fig)  # cleanup

    # calculating A, B, plotting ##################################################################
    print("::: plotting A(q), B ...")
    if ensemble_average:
        As, Bs = np.array([0]), 0.0
        for fov in rfft2:  # iterate over all fields of view
            _A, _B = static_estimate_A_B(fov)
            As += _A
            Bs += _B
        A = As / len(rfft2)  # normalization
        B = Bs / len(rfft2)
        del As, Bs, _A, _B, rfft2

    else:
        A, B = static_estimate_A_B(rfft2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(q, A, label=r"$A(q)$")
    ax.hlines(B, q[0], q[-1], linestyle="--", colors="k", label=r"$B={:.2f}$".format(B))
    ax.set_xlabel(r"Wavevectors $q\ [{}^{{-1}}]$".format(unit))
    ax.set_ylabel(r"$A(q),\ B$ [a.u.]")
    ax.set_title(f"$A(q),\ B$ | {binary_file_name} | {notes}", fontsize=9)
    ax.legend()
    ax.grid()
    ax.set_ylim((1e-2, 1e6))
    ax.set_yscale("log")
    fig.savefig(plots / f"static_AB-{binary_file_name}.png", dpi=150)
    plt.close(fig)

    # calculating ISF #############################################################################
    print("::: Plotting ISF with exponential fit ...")
    isf = np.zeros_like(azimuthal_avg)
    for i, avg in enumerate(azimuthal_avg):
        isf[i] = intermediate_scattering_function(avg, A, B)

    # fitting exponential to all q values within prepared range
    fit_u = np.arange(*idx_range)
    fit_q = from_u_to_q(fit_u, metadata)
    fit_params = {}

    for fu in fit_u:
        popt, pcov = curve_fit(general_exp, time, isf[:, fu], p0=default_p0)
        fit_params[fu] = (popt, np.sqrt(np.diag(pcov)))

    # plotting ISF with fits for test u/q values
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, tu in enumerate(test_u):
        ax.plot(
            time, isf[:, tu], color=colors[i], label="$q = {:.3f}$".format(test_q[i])
        )

        popt, _ = fit_params[tu]
        ax.plot(
            time,
            general_exp(time, *popt),
            linestyle=":",
            linewidth=0.8,
            color=colors[i],
        )
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlabel(r"Time $t\ [s]$")
    ax.set_ylabel(r"Intermediate scattering function $f(q, \Delta t)$")
    ax.set_title(f"ISF w/ {exp_type} fit | {binary_file_name} | {notes}", fontsize=9)
    ax.legend()
    ax.grid()
    ax.set_xscale("log")
    fig.savefig(plots / f"isf-{exp_type}-{binary_file_name}.png", dpi=150)
    plt.close(fig)

    # plotting fit results ########################################################################
    print("::: Fit results ...")

    parameters = np.array(
        [item[0] for item in fit_params.values()]
    )  # extracting parameters
    parameters = parameters.T

    fig, _ = plot_exp_fit_parameters(
        parameters, fit_q, colors, fit_parameter_labels, unit
    )

    fig.suptitle(f"Fitting parameters | {binary_file_name} | {notes}", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / f"{exp_type}-fit-pars-{binary_file_name}.png", dpi=150)
    plt.close(fig)

    # fitting power law and plotting it ###########################################################
    if args.fit_power_law:
        print("::: fitting power law to tau(q) ...")

        if stretched_exp:
            # this part needs some extra treatment
            print(
                "    --> Used stretching exponent; averaging beta(q), then redoing exp fits with "
                "fixed beta."
            )
            beta_avg, beta_std = parameters[2].mean(), parameters[2].std()
            print(f"    -->〈β(q)〉= {beta_avg:.2f}, std(β(q)) = {beta_std:.2f}")
            exp_fixed_beta = partial(general_exp, beta=beta_avg)

            # redoing fits, overwriting old values
            for fu in fit_u:  # ignore the last value in default_p0
                popt, pcov = curve_fit(
                    exp_fixed_beta, time, isf[:, fu], p0=default_p0[:-1]
                )
                fit_params[fu] = (popt, np.sqrt(np.diag(pcov)))

            # extract parameters array
            parameters = np.array([item[0] for item in fit_params.values()])
            parameters = parameters.T

            print("     --> Plotting exponential fit parameters of redone fits...")
            fig, axes = plot_exp_fit_parameters(
                parameters, fit_q, colors, fit_parameter_labels, unit
            )
            # manually plot the first moment as well
            tau_m = tau_moment(parameters[0], beta_avg)
            axes[0].scatter(
                fit_q,
                tau_m,
                c="k",
                marker="v",
                label=r"$\langle \tau(q) \rangle$",
            )
            axes[0].legend()

            fig.suptitle(
                f"Fitting parameters | β={beta_avg:.2f} | {binary_file_name} | {notes}",
                fontsize=8,
            )
            fig.tight_layout()
            fig.savefig(
                plots / f"{exp_type}-fixed-beta-fit-pars-{binary_file_name}.png",
                dpi=150,
            )
            plt.close(fig)

        print("::: Plotting power law ...")
        power_law_p0 = [1.0, -1.5]  # prefactor, exponent
        title_addendum = ""

        fig, ax = plt.subplots(figsize=(6, 4))
        # first plot tau(q)
        tau = parameters[0]
        ax.scatter(fit_q, tau, s=20)
        if stretched_exp:
            ax.scatter(
                fit_q, tau_m, c="k", marker="v", label=plot_labels["tau_moment_legend"]
            )
            popt, pcov = curve_fit(
                power_law, fit_q, tau_m, p0=power_law_p0
            )  # fit tau_m
            title_addendum = r"| $\beta = {:.2f}$".format(beta_avg)
        else:
            popt, pcov = curve_fit(power_law, fit_q, tau, p0=power_law_p0)

        popt_err = np.sqrt(np.diag(pcov))
        _, eta = popt
        ax.plot(
            fit_q,
            power_law(fit_q, *popt),
            c="tab:red",
            linestyle="--",
            label=plot_labels["power_law_legend"].format(eta=eta),
        )

        ax.set_ylim((500, 1e4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(plot_labels["wavevector_axis"].format(unit=unit))
        ax.set_ylabel(plot_labels["tau_axis"])
        ax.set_title(f"Power law fit {title_addendum}")
        ax.grid(which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / f"power-law-{exp_type}-{binary_file_name}.png")
        plt.close(fig)

        return popt, popt_err
    return None


# argpares setup
parser = argparse.ArgumentParser()
parser.add_argument(
    "src", metavar="SRC", nargs="+", help="The location of source file(s) to analyze."
)
parser.add_argument(
    "--dest",
    default="plots/",
    help="The location where to put plots and plot-related data. Default value is 'plots/' relative to src directory.",
)
parser.add_argument(
    "--fit-stretched-exp",
    action="store_true",
    help="Fit a stretched exponential instead of a regular exponential.",
)
parser.add_argument(
    "--average",
    action="store_true",
    help="Average over multiple input directories, keeping the chunked structure.",
)
parser.add_argument(
    "--fit-power-law",
    action="store_true",
    help="Additionally fit a power law to tau(q). If the stretching exponent is used, the average "
    "of the stretching exponent is used to redo all fits, and then the power law fit is performed.",
)
args = parser.parse_args()

# default p0 values for curve_fit
# default_p0 = [1e3, 1.0]  # amplitude, tau; offset is always very close to zero anyway
stretched_exp = args.fit_stretched_exp
exp_type = "stretched-exp" if stretched_exp else "exp"
default_p0 = [1e3, 1.0, 1.0] if stretched_exp else [1e3, 1.0]
fit_parameter_labels = [
    r"$\tau(q)\ [s]$",
    "amplitude of exponential",
    r"$\beta(q)$",
]
plot_labels = {
    "tau_moment_legend": r"$\langle \tau(q) \rangle$",
    "power_law_legend": r"$\sim q^{{ {eta:.2f} }}$",  # to be formatted
    "wavevector_axis": r"Wavevectors $q\ [{unit}^{{-1}}]$",  # to be formatted
    "tau_axis": r"$\tau(q)\ [s]$",
}

if __name__ == "__main__":
    total_analysis_time = perf_counter()
    # print(f":: Starting analysis plots in {args.src}")

    # reading data
    src = args.src
    if len(src) > 1:
        sources = [Path(path) for path in src]
        if all([path.is_dir() for path in sources]) and args.average:
            print(": Multiple directories to average over chunk-wise:")
            for path in sources:
                print(f"  --> {path.name}")

            sources_files = [
                [file for file in sorted(src.iterdir()) if file.is_file()]
                for src in sources
            ]

            plots = Path(args.dest)
            plots.mkdir(parents=True, exist_ok=True)  # ensure plotting path exists
            for ensemble in zip(*sources_files):
                result = analyse_single(list(ensemble), plots, ensemble_average=True)
                if result is not None:
                    print(result)

        else:
            raise RuntimeError(
                "Got mixed directories and single files and/or w/o average flag."
            )

    else:
        src = Path(*src)
        if src.is_file():  # single file
            plots = src.parent / args.dest if args.dest == "plots/" else args.dest
            analyse_single(src, plots)
        else:
            plots = src / args.dest if args.dest == "plots/" else args.dest
            for file in [file for file in sorted(src.iterdir()) if file.is_file()]:
                print(f"\n:: working on file {file}")
                result = analyse_single(file, plots)
                if result is not None:
                    print(result)

    print(
        f":: Overall plotting & fitting took {perf_counter() - total_analysis_time: .2f} s"
    )
