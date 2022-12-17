"""Helper functions/computations."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dfmtoolbox._dfm_python import (
    reconstruct_full_spectrum,
    azimuthal_average,
)


@dataclass
class AnalysisBlob:
    """Class for bundling analysis data with additional information."""

    data_source: Path
    rfft2_sqmod: np.ndarray  # the square modulus of the rfft2
    lags: np.ndarray
    image_structure_function: np.ndarray
    azimuthal_average: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    stardist_details: Optional[List[np.ndarray]] = None


@dataclass
class FitResults:
    """Class for storing fit results for ISF and power law of tau."""

    data_source: Union[Path, List[Path]]
    isf: Optional[Dict[int, Tuple[Dict[str, float], np.ndarray, List[Any]]]] = None
    power_law: Optional[Tuple[np.ndarray, np.ndarray]] = None
    notes: Optional[str] = None


@dataclass
class StructureFactor:
    """Class for storing structure factor results."""

    sf: np.ndarray
    q: np.ndarray  # wavevectors
    sf_std: Optional[np.ndarray] = None
    size: Optional[int] = None  # sample size
    notes: Optional[str] = None


@dataclass
class ImageCellStats:
    """Class for storing statistics about cells in images."""

    data_source: Path  # folder or file
    images: np.ndarray  # image indices
    stats: List[Dict[str, Any]]  # list of stats dictionary on a per-image basis


def static_estimate_A_B(
    rfft2_sqmod: np.ndarray, shape: Optional[Tuple[int, ...]]
) -> Tuple[np.ndarray, float]:
    """Estimate the values for A(q) and B from an average of the power spectra of ImageSet FFT2s.

    The values for A(q) and B can be used to determine the shape of the ISF, since

        D(q, dt) = A(q)[1 - ISF(q, dt)] + B

    with D(q, dt) being the image structure function. The value of B is taken as the average of the
    10 last values of two times the average power spectrum, and the sum of A and B is given by:

        A + B = 2 * <|FFT2(I)|^2>
    """
    # assume we already receive the square modulus of the rfft2
    power_spec = np.array([reconstruct_full_spectrum(im, shape) for im in rfft2_sqmod])
    power_spec = power_spec.mean(axis=0)
    a_plus_b = azimuthal_average(power_spec)
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
        if "dim_x" in pars.keys() and "dim_y" in pars.keys():
            pars["image_size"] = max(pars["dim_x"], pars["dim_y"])
        else:
            raise RuntimeError("[ERR] Supplied parameters don't contain image size.")

    u_min = 1 / (pars["image_size"] * pars["pixel_size"] / pars["magnification"])
    q_min = 2 * np.pi * u_min

    return u * q_min


def chunkify(data: np.ndarray, chunksize: int, overlap: int = 0) -> List[np.ndarray]:
    """Takes a dataset and chunks it into smaller portions of size `chunksize`, with a given `overlap` with the previous chunk.

    The last chunk may not be of the right size. The chunking will happen along the __first__ axis.
    """
    size = len(data)
    nchunks = size // chunksize
    if nchunks == 0:  # nothing to do here
        return [data]

    left, right, diff = 0, chunksize, chunksize - overlap
    chunks = []

    # main chunks
    while right < size:
        chunks.append(data[left:right])
        left += diff
        right += diff

    # rest chunk if any
    if len(data[left:]) > 0:
        chunks.append(data[left:])

    return chunks
