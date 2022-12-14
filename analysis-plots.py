#!/usr/bin/env python

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as scifft
from dfmtoolbox._dfm_python import azimuthal_average, spatial_frequency_grid
from dfmtoolbox.io import read_data, store_data
from matplotlib.colors import TABLEAU_COLORS

# external local modules
from models import (
    ExtractedResult,
    exp_model,
    extract_results,
    fit,
    intermediate_scattering_function,
    model_generator,
    power_law,
    tau_model,
    tau_moment,
)
from plotting import decorate_axes, finalize_and_save
from utils import AnalysisBlob, FitResults, from_u_to_q, static_estimate_A_B


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
    parameters: Dict[str, np.ndarray],
    fit_q: np.ndarray,
    colors: List[str],
    fit_parameter_labels: List[str],
    xunit: str,
) -> Tuple[plt.Figure, plt.Axes]:

    # size = len(parameters)
    pnames = list(parameters.keys())
    nparameters = len(pnames)

    fig, axes = plt.subplots(
        nparameters, 1, figsize=(6, 2.5 * nparameters), sharex=True
    )

    for i, pname in enumerate(pnames):
        ax = axes[i]
        par = parameters[pname]
        ax.scatter(fit_q, par, c=colors[i], s=20)
        ax.set_ylabel(fit_parameter_labels[i])
        ax.grid(which="both")
        ax.set_xscale("log")

        # fit parameter specific settings
        if pname == "tau":  # tau
            ax.set_yscale("log")
            ax.set_ylim((1e2, 6e4))

        elif pname == "amp":  # amplitude
            ax.set_ylim((0.6, 1.2))

        elif pname == "beta":  # beta
            ax.set_ylim((0.2, 1.9))
            # also plot the first moment of tau in the right axis
            axes[0].scatter(
                fit_q,
                tau_moment(parameters["tau"], par),
                c="k",
                marker="v",
                label=r"$\langle \tau(q) \rangle$",
            )
            axes[0].legend()

    # cosmetics
    ax.set_xlabel(r"Wavevector $q\ [{}^{{-1}}]$".format(xunit))

    return fig, axes


def fit_isf(
    model: lm.Model,
    fit_u: Iterable[int],
    *,
    ydata: np.ndarray,
    alt_weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[int, ExtractedResult]:
    """Fit the intermediate scattering function with the given model in a given range of u values.

    Parameters
    ----------
    model : lm.Model
        The lmfit model instance used for fitting
    fit_u : Iterable[int]
        The iterable of u values the fit should be produced for.
    ydata : np.ndarray
        The 2D data we want to fit the model to; the last axis will be used for iteration.
    alt_weights: Optional[np.ndarray]
        Alternative weights if fit fails at first try, by default None

    Returns
    -------
    _type_
        _description_
    """
    fit_params = {}
    alt_kwargs = kwargs.copy()
    alt_kwargs["weights"] = alt_weights

    for fu in fit_u:
        result = fit(model, ydata=ydata[:, fu], **kwargs)
        if not result.success and (alt_weights is not None):
            print(f"    --> retrying fit for u={fu} with different weights.. ")
            result = fit(model, ydata=ydata[:, fu], **alt_kwargs)

        fit_params[fu] = extract_results(result)

    return fit_params


def transpose_extracted_results(
    fit_params: Dict[int, ExtractedResult]
) -> Dict[str, Any]:

    parameters, param_errors, uncertainty_bands = [], [], []
    fit_u = list(fit_params.keys())
    for (par, perr, uncert) in fit_params.values():
        parameters.append(par)
        param_errors.append(perr)
        uncertainty_bands.append(uncert)

    size = len(parameters)
    pnames = list(parameters[0].keys())

    pars = {key: np.zeros(size) for key in pnames}
    for i, pdict in enumerate(parameters):
        for pname, value in pdict.items():
            pars[pname][i] = value

    transposed = {
        "u": fit_u,
        "parameters": pars,
        "param_errors": param_errors,
        "uncertainty_bands": uncertainty_bands,
    }
    return transposed


def find_tau_fit_range(tau: np.ndarray, fit_q: np.ndarray) -> Tuple[int, int]:
    # tau can be tau or tau_m
    # the value of maximum tau is given by chunksize / fps
    # for now hardcoded
    max_tau_val = 200 * 60  # 200 / (1 / 60)
    (idx,) = np.where(tau <= max_tau_val)
    left_idx = idx.min()

    right_idx = max(
        np.argmin(np.abs(tau[:-1] - tau[1:])),
        np.argmin(np.abs(fit_q - 1)),
    )

    return int(left_idx), int(right_idx)


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
        rfft2_sqmod = np.array([blob.rfft2_sqmod for blob in ensemble])

        *_, ny, nx = dqt.shape
        bigside = max(nx, ny)
        kx = scifft.fftfreq(nx)
        ky = scifft.fftfreq(ny)
        spatial_freq = scifft.fftshift(
            spatial_frequency_grid(kx, ky)
        )  # equiv of old distance array
        dqt_dist = spatial_freq  # distance_array(dqt.shape)
        azimuthal_avg = np.zeros(
            (len(dqt), bigside // 2)
        )  # metadata["image_size"] // 2))
        for i in range(len(dqt)):
            azimuthal_avg[i] = azimuthal_average(dqt[i], dqt_dist)

        # memory cleanup
        del ensemble[:], blob

    else:
        print(":: Reading data ...")
        binary_file_name = src.name.replace(".pkl", "")  # type: ignore
        blob = read_data(src)
        print_blob_info(blob)
        *_, ny, nx = blob.image_structure_function.shape
        bigside = max(nx, ny)

        lags = blob.lags
        metadata = blob.metadata
        azimuthal_avg = blob.azimuthal_average
        rfft2_sqmod = blob.rfft2_sqmod
        notes = blob.notes

    fps = metadata["fps"]
    # pixel_size = metadata["pixel_size"]
    unit = metadata["pixel_size_unit"]

    # convert pixel-u-values to q wave vectors
    u = np.arange(1, bigside // 2 + 1)
    q = from_u_to_q(u, metadata)

    # get representative dt values:
    test_lags = np.linspace(1, len(lags), num=5, dtype=np.int64, endpoint=False)
    time = lags / fps

    # representative u/q values (linspaced)
    idx_range = (9, int(len(u) * 0.8))
    test_wv_idc = (
        np.linspace(*idx_range, num=6, dtype=np.int64, endpoint=False) - 1
    )  # test indices for u and q; use only the first half of q range, rest is very noisy
    test_u = u[test_wv_idc]  # get some test u values
    test_q = q[test_wv_idc]

    # plotting azimuthal average ##################################################################
    print("::: plotting azimuthal average for some test-lags .. ")
    fig, ax = plt.subplots(figsize=(6, 6))
    for testlag in test_lags:
        ax.plot(q, azimuthal_avg[testlag - 1], label=r"$\Delta t={}$".format(testlag))

    ax = decorate_axes(
        ax,
        xlabel=plot_labels["wavevector_axis"].format(unit=unit),
        ylabel="Azimuthal average [a.u.]",
        title=f"Azimuthal avg | {binary_file_name} | {notes}",
        grid_args=None,
        xscale="linear",
    )
    finalize_and_save(fig, plots / f"az-avg-{binary_file_name}")

    # calculating A, B, plotting ##################################################################
    print("::: plotting A(q), B ...")
    if ensemble_average:
        As, Bs = np.array([0.0]), 0.0
        for fov in rfft2_sqmod:  # iterate over all fields of view
            _A, _B = static_estimate_A_B(fov, (ny, nx))
            As = _A + As  # shorthand notation doesn't work
            Bs += _B
        A = As / len(rfft2_sqmod)  # normalization
        B = Bs / len(rfft2_sqmod)
        del As, Bs, _A, _B, rfft2_sqmod

    else:
        A, B = static_estimate_A_B(rfft2_sqmod, (ny, nx))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(q[1:], A[1:], label=r"$A(q)$")
    ax.hlines(B, q[1], q[-1], linestyle="--", colors="k", label=r"$B={:.2f}$".format(B))

    ax = decorate_axes(
        ax,
        xlabel=plot_labels["wavevector_axis"].format(unit=unit),
        ylabel=r"$A(q),\ B$ [a.u.]",
        title=f"$A(q),\ B$ | {binary_file_name} | {notes}",
        grid_args=None,
        xscale="linear",
        ylim=(1e-2, 1e6),
    )

    finalize_and_save(fig, plots / f"static_AB-{binary_file_name}")

    # calculating ISF #############################################################################
    print("::: Plotting ISF with exponential fit ...")
    isf = np.zeros_like(azimuthal_avg)
    for i, avg in enumerate(azimuthal_avg):
        isf[i] = intermediate_scattering_function(avg, A, B)

    # fitting (stretched) exponential to all q values within prepared range
    fit_u = np.arange(*idx_range)
    fit_q = from_u_to_q(fit_u, metadata)
    # weights = np.sqrt(lags.max() / lags)
    weights = 1 / np.sqrt(lags)

    # fitting
    fit_params = fit_isf(
        exp_model, fit_u, ydata=isf, xdata=time, weights=weights, alt_weights=1 / lags
    )

    # store fit parameters
    fit_results = FitResults(
        data_source=src, isf=fit_params, notes=f"{notes}; {exp_type}"
    )

    # plotting ISF with fits for test u/q values
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, tu in enumerate(test_u):
        # datapoints
        ax.scatter(
            time,
            isf[:, tu],
            s=3,
            color=colors[i],
            label="$q = {:.3f}$".format(test_q[i]),
        )

        # fit
        popt, _, uncertainty_band = fit_params[tu]
        ax.plot(
            time,
            # general_exp(time, *popt),
            exp_model.eval(t=time, **popt),
            linestyle="--",
            linewidth=1,
            color=colors[i],
        )
        # uncertainty band
        ax.fill_between(time, *uncertainty_band, color=colors[i], alpha=0.5)

    ax = decorate_axes(
        ax,
        xlabel=r"Lag times $\Delta t\ [s]$",
        ylabel=r"Intermediate scattering function $f(q, \Delta t)$",
        title=f"ISF w/ {exp_type} fit | {binary_file_name} | {notes}",
        yscale="linear",
        grid_args=None,
        ylim=(-0.1, 1.1),
    )
    finalize_and_save(fig, plots / f"isf-{exp_type}-{binary_file_name}")

    # plotting fit results ########################################################################
    print("::: Fit results ...")

    transposed_results = transpose_extracted_results(fit_params)
    parameters = transposed_results["parameters"]

    fig, _ = plot_exp_fit_parameters(
        parameters, fit_q, colors, fit_parameter_labels, unit
    )

    fig.suptitle(f"Fitting parameters | {binary_file_name} | {notes}", fontsize=8)
    finalize_and_save(fig, plots / f"{exp_type}-fit-pars-{binary_file_name}")

    # fitting power law and plotting it ###########################################################
    if args.fit_power_law:
        print("::: fitting power law to tau(q) ...")

        if stretched_exp:
            # this part needs some extra treatment
            print(
                "    --> Used stretching exponent; averaging beta(q), then redoing exp fits with "
                "fixed beta."
            )
            beta_avg, beta_std = parameters["beta"].mean(), parameters["beta"].std()
            beta_std_err = beta_std / np.sqrt(len(parameters["beta"]))

            print(f"    -->?????(q)???= {beta_avg:.2f}, std_err(??(q)) = {beta_std_err:.2f}")

            # exp_fixed_beta = partial(general_exp, beta=beta_avg)
            exp_fixed_beta = model_generator("exp")
            exp_fixed_beta.set_param_hint("beta", value=beta_avg, vary=False)

            # redoing fits, overwriting old values
            fit_params = fit_isf(
                exp_fixed_beta, fit_u, ydata=isf, xdata=time, weights=weights
            )

            # extract parameters list of dicts
            transposed_results = transpose_extracted_results(fit_params)
            parameters = transposed_results["parameters"]
            # reduce to tau and amplitude, since beta is constant
            del parameters["beta"]

            print("    --> Plotting exponential fit parameters of redone fits...")
            fig, axes = plot_exp_fit_parameters(
                parameters, fit_q, colors, fit_parameter_labels, unit
            )
            # manually plot the first moment as well
            tau_m = tau_moment(parameters["tau"], beta_avg)
            axes[0].scatter(
                fit_q,
                tau_m,
                c="k",
                marker="v",
                label=r"$\langle \tau(q) \rangle$",
            )
            axes[0].legend()

            fig.suptitle(
                f"Fitting parameters | ??={beta_avg:.2f} | {binary_file_name} | {notes}",
                fontsize=8,
            )
            finalize_and_save(
                fig, plots / f"{exp_type}-fixed-beta-fit-pars-{binary_file_name}"
            )

        print("::: Plotting power law ...")
        title_addendum = ""

        fig, ax = plt.subplots(figsize=(6, 4))
        # first plot tau(q)
        tau = parameters["tau"]
        ax.scatter(fit_q, tau, s=20)
        if stretched_exp:
            ax.scatter(
                fit_q, tau_m, c="k", marker="v", label=plot_labels["tau_moment_legend"]
            )
            title_addendum = r"| $\beta = {:.2f}$".format(beta_avg)

            left, right = find_tau_fit_range(tau_m, fit_q)

        else:
            left, right = find_tau_fit_range(tau, fit_q)

        tau_fit_q = fit_q[left:right]
        result = fit(tau_model, xdata=tau_fit_q, ydata=tau_m[left:right])
        popt, popt_err, uncertainty_band = extract_results(result)

        ax.plot(
            tau_fit_q,
            tau_model.eval(q=tau_fit_q, **popt),
            c="tab:red",
            linestyle="--",
            label=plot_labels["power_law_legend"].format(eta=popt["eta"]),
        )
        ax.fill_between(tau_fit_q, *uncertainty_band, color="tab:red", alpha=0.4)

        ax = decorate_axes(
            ax,
            xlabel=plot_labels["wavevector_axis"].format(unit=unit),
            ylabel=plot_labels["tau_axis"],
            title=f"Power law fit {title_addendum} | {binary_file_name}",
            grid_args={"which": "both"},
            ylim=(3e2, 6e4),
        )
        finalize_and_save(fig, plots / f"power-law-{exp_type}-{binary_file_name}")

        fit_results.power_law = (popt, popt_err)
        store_data(fit_results, path=plots, name=f"fit-results-{binary_file_name}")

        return np.array(list(popt.values())), popt_err

    store_data(fit_results, path=plots, name=f"fit-results-{binary_file_name}")

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

# set initial guess for tau
exp_model.set_param_hint("tau", value=1e3)

# bounds2-tuple of array_like, optional
if not stretched_exp:
    exp_model.set_param_hint("beta", vary=False)  # set beta to be a constant

fit_parameter_labels = [
    r"$\tau(q)\ [s]$",
    "amplitude of exponential",
    r"$\beta(q)$",
]
plot_labels = {
    "tau_moment_legend": r"$\langle \tau(q) \rangle$",
    "power_law_legend": r"$\sim q^{{ -{eta:.2f} }}$",  # to be formatted
    "wavevector_axis": r"Wavevectors $q\ [{unit}^{{-1}}]$",  # to be formatted
    "tau_axis": r"$\tau(q)\ [s]$",
}

if __name__ == "__main__":
    total_analysis_time = perf_counter()
    # print(f":: Starting analysis plots in {args.src}")

    result_pars = []
    result_errs = []
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
                    pars, errs = result
                    result_pars.append(pars)
                    result_errs.append(errs)

        else:
            raise RuntimeError(
                "Got mixed directories and single files and/or w/o average flag."
            )

    else:
        src = Path(*src)
        if src.is_file():  # single file
            plots = src.parent / args.dest if args.dest == "plots/" else args.dest
            result = analyse_single(src, plots)
            if result is not None:
                print(result)
        else:
            plots = src / args.dest if args.dest == "plots/" else args.dest
            for file in [file for file in sorted(src.iterdir()) if file.is_file()]:
                print(f"\n:: working on file {file}")
                result = analyse_single(file, plots)
                if result is not None:
                    pars, errs = result
                    result_pars.append(pars)
                    result_errs.append(errs)

    if len(result_pars):
        print(":: Plotting power law evolution ... ")
        result_pars = np.array(result_pars).T
        result_errs = np.array(result_errs).T
        amplitude, eta = result_pars
        amplitude_err, eta_err = result_errs
        age = np.arange(len(eta))

        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        ax = axes[0]
        ax.errorbar(age, eta, eta_err, fmt=".", capthick=1, capsize=2, ecolor="tab:red")

        ax = decorate_axes(
            ax,
            xlabel="Age [chunk]",
            ylabel="Scaling exponent $\eta$",
            title=r"{}| Scaling exponent $\eta$ evolution for $\tau = q^{{-\eta}}/v_0$".format(
                exp_type
            ),
            ylim=(1.0, 2.0),
            grid_args=None,
            yscale="linear",
            xscale="linear",
            legend=False,
        )

        ax = axes[1]
        ax.errorbar(
            age,
            amplitude,
            amplitude_err,
            fmt=".",
            capthick=1,
            capsize=2,
            ecolor="tab:red",
        )

        ax = decorate_axes(
            ax,
            xlabel="Age [chunk]",
            ylabel="Amplitude $1/v_0$",
            title=r"{}| Amplitude $1/v_0$ evolution for $\tau(q) = q^{{-\eta}}/v_0$".format(
                exp_type
            ),
            grid_args=None,
            yscale="linear",
            xscale="linear",
            legend=False,
        )
        fig.suptitle(
            "Error bars from power law fit of the ensemble averaged data", fontsize=9
        )
        fig.savefig(plots / f"power-law-evo-{exp_type}.png")

    print(
        f":: Overall plotting & fitting took {perf_counter() - total_analysis_time: .2f} s"
    )
