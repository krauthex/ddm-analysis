"""A collection of helper tools for fitting."""
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
