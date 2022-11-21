"""A collection of helper tools for fitting."""
from typing import Union, Optional, Any

import lmfit as lm
import numpy as np
from scipy.special import gamma


def general_exp(
    t: np.ndarray, tau: float, amp: float = 1.0, beta: float = 1.0
) -> np.ndarray:
    """The general exponential of the shape

        f(t) = amp * exp(-(t/tau)^beta)

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


exp_model = lm.Model(general_exp)
exp_model.set_param_hint("tau", min=0.0, max=np.inf)
exp_model.set_param_hint(
    "amp", min=0.0, max=2.0
)  # max value of 1 possibly causes artifacts
exp_model.set_param_hint("beta", min=0.0, max=np.inf)


def tau_moment(tau: float, beta: float) -> float:
    """Calculate the first moment of the stretched exponential function."""

    return tau / beta * gamma(1 / beta)


def power_law(q: np.ndarray, a: float, eta: float) -> np.ndarray:
    """A general power law function of the shape f(q) = a * q^-eta.

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
    return a * q ** (-eta)


tau_model = lm.Model(power_law)
tau_model.set_param_hint("a", min=0, max=np.inf)
tau_model.set_param_hint("eta", min=0, max=np.inf)


def fit(
    model: lm.Model,
    xdata: np.ndarray,
    ydata: np.ndarray,
    params: Optional[Union[lm.Parameters, lm.Parameter]] = None,
    verbose: bool = False,
    **fitargs: Any,
) -> lm.model.ModelResult:
    """A wrapper for fitting a given model to given data.

    It is highly recommended to pass the `weights` argument for very noisy data.

    Parameters
    ----------
    model : lm.Model
        The model to be used for the fit.
    xdata : np.ndarray
        The data of the independent variable.
    ydata : np.ndarray
        The data we want to fit the model to.
    params : Optional[Union[lm.Parameters, lm.Parameter]], optional
        Either a single lm.Parameter or lm.Parameters, as the Model expects, by default None
    verbose : bool, optional
        Pretty prints the parameters before fitting and the fit report afterwards, by default False

    Returns
    -------
    lm.model.ModelResult
        The results of the fit.
    """
    if verbose:
        print(":: Model parameters:")
        p = model.make_params() if params is None else params
        p.pretty_print()

    # we will assume the models only have one indep. variable
    indep_var = model.independent_vars[0]
    fitargs[indep_var] = xdata  # mapping xdata to independent variable name

    # data for "x" should be in fitargs with the correct naming of the independent
    result = model.fit(ydata, params=params, **fitargs)

    if verbose:
        print(":: Fit report:")
        print(result.fit_report())

    return result
