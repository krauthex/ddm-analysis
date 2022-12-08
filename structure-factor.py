#!/usr/bin/env python
"""Perform real space analysis of fluorescence cell nuclei to get the structure factor."""

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Tuple, Union, Optional, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as scifft
from csbdeep.utils import normalize
from dfmtoolbox._dfm_python import azimuthal_average, spatial_frequency_grid

# import matplotlib.pyplot as plt
from dfmtoolbox.utils import tiff_to_numpy
from stardist.models import StarDist2D
from sympy import Point, Polygon

# from stardist.plot import render_label


# stardist setup
# prints a list of available models
# StarDist2D.from_pretrained()

Details = Dict[str, np.ndarray]
Stats = Dict[str, Union[int, Dict[str, float]]]


def stardist_single(image: np.ndarray, model: StarDist2D, norm: bool = True) -> Any:
    """Apply stardist to a single image and return the labels & details objects."""
    if norm:
        image = normalize(image)
    labels, details = model.predict_instances(image)
    return labels, details


def images(path: Path, seq: Optional[Sequence[int]] = None) -> np.ndarray:
    return tiff_to_numpy(path, seq=seq)


def multicore_dispatch(
    func: Callable,
    func_iterable: np.ndarray,
    *,
    cpus: int = 2,
) -> Any:
    pool = Pool(cpus)
    chunksize = len(func_iterable) // cpus  # rough chunksize setting
    results = pool.imap(func, func_iterable, chunksize=chunksize)

    return results


def single_cell_stat(coord: np.ndarray) -> Tuple[float, float]:
    points = [Point(*pos) for pos in coord.T]
    polygon = Polygon(*points)

    return abs(float(polygon.area)), abs(float(polygon.perimeter))


def stats(details: Details, *, cpus: int = 1) -> Stats:
    # calc average cell area & std, average cell number & std
    coord = details["coord"]
    cell_number = len(coord)
    areas = np.zeros(cell_number)
    perimeters = np.zeros(cell_number)

    for i, result in enumerate(multicore_dispatch(single_cell_stat, coord, cpus=cpus)):
        area, perim = result
        areas[i] = area
        perimeters[i] = perim

    stat = {
        "number": cell_number,
        "area": {"mean": areas.mean(), "std": areas.std()},
        "perimeter": {"mean": perimeters.mean(), "std": perimeters.std()},
    }
    return stat


def structure_factor_single(
    points: np.ndarray,
    shape: Tuple[int, int],
    *,
    workers: int = 2,
    spatial_freqs: Optional[np.ndarray] = None,
) -> np.ndarray:

    sf = np.zeros(shape)
    y, x = points.T  # points is expected to be `details["points"]`
    norm = len(y)  # number of points
    sf[y, x] = 1
    fft2 = scifft.fftshift(scifft.fft2(sf, workers=workers))

    return azimuthal_average(np.abs(fft2) ** 2, dist=spatial_freqs) / norm


def structure_factor(
    images: np.ndarray,
    *,
    model: StarDist2D,
    cpus: int = 1,
) -> np.ndarray:
    stardist_fixed_model = partial(stardist_single, model=model)
    shape = images[0].shape
    full_sf = []
    kx = scifft.fftfreq(shape[-1])
    ky = scifft.fftfreq(shape[-2])
    spatial_freq = spatial_frequency_grid(kx, ky)

    # for i, result in enumerate(
    #     multicore_dispatch(stardist_fixed_model, images, cpus=cpus)
    # ):
    for im in images:
        _, details = stardist_fixed_model(im)
        full_sf.append(
            structure_factor_single(
                details["points"], shape=shape, workers=cpus, spatial_freqs=spatial_freq
            )
        )

    sf = np.array(full_sf).mean(axis=0)

    return sf


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


parser = argparse.ArgumentParser()
parser.add_argument("src", help="The source TIFF file.")
parser.add_argument(
    "--stats", action="store_true", help="Compute the stats for each image."
)
parser.add_argument(
    "--cpus", type=int, default=1, help="Number of CPUs to use for polygon computation."
)
parser.add_argument(
    "--chunksize",
    type=int,
    default=0,
    help="Splits the analysis and averaging int o slices of this chunksize.",
)
parser.add_argument("--overlap", type=int, default=0, help="Overlap of chunks.")

args = parser.parse_args()
overlap, chunksize = args.overlap, args.chunksize


if __name__ == "__main__":
    # timing
    start = perf_counter()

    # creates a pretrained model
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    # read images
    imgs = images(args.src)

    # setup chunks
    length = len(imgs)
    if chunksize == 0:
        chunks = [np.arange(length)]
    else:
        print(f":: Chunksize set to {chunksize}")
        chunks = chunkify(np.arange(length), chunksize=chunksize, overlap=overlap)

    for chunk in chunks:
        data = imgs[chunk]  # select images based on indices
        if args.stats:
            pass

        else:
            sf_time = perf_counter()

            # l, d = stardist_single(img[0], model)
            print(":: structure factor")
            sf = structure_factor(data, model=model, cpus=args.cpus)

            # sf = structure_factor_single(
            #     d["points"], shape=img[0].shape, workers=args.cpus
            # )
            sf_time = perf_counter() - sf_time
            plt.scatter(range(len(sf) - 1), sf[1:], s=3)
            plt.title(f"Average structure function for {chunksize} images")
            plt.show()

            # results = stats(d, cpus=args.cpus)
            # area = results["area"]
            # perimeter = results["perimeter"]
            # n = results["number"]

            # print(f"Average area: {area['mean']:.2f} \pm {area['std']/np.sqrt(n):.2f}")
            # print(
            #     f"Average perimeter: {perimeter['mean']:.2f} \pm {perimeter['std']/np.sqrt(n):.2f}"
            # )

            print(f":: Overall runtime: {perf_counter()-start:.2f}s")
            # keys: coord, points, prob
