"""A collection of helper tools for fitting."""
from typing import Union, Optional, Dict, Any

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
exp_model.set_param_hint("amp", min=0.0, max=1.0)
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
    ydata: np.ndarray,
    xdata: np.ndarray,
    params: Optional[Union[lm.Parameters, lm.Parameter]] = None,
    verbose: bool = False,
    **fitargs: Any,
) -> lm.model.ModelResult:

    if verbose:
        p = model.make_params() if params is None else params
        p.pretty_print()

    # we will assume the models only have one indep. variable
    indep_var = model.independent_vars[0]
    fitargs[indep_var] = xdata  # mapping xdata to independent variable name

    # data for "x" should be in fitargs with the correct naming of the independent
    result = model.fit(ydata, params=params, **fitargs)

    return result
