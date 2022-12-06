#!/usr/bin/env python
"""Perform real space analysis of fluorescence cell nuclei to get the structure factor."""

import argparse
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
from csbdeep.utils import normalize

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


def images(path: Path) -> np.ndarray:
    return tiff_to_numpy(path)


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


parser = argparse.ArgumentParser()
parser.add_argument("src", help="The source TIFF file.")
parser.add_argument(
    "--cpus", type=int, default=1, help="Number of CPUs to use for polygon computation."
)

args = parser.parse_args()

if __name__ == "__main__":
    # timing
    start = perf_counter()

    # creates a pretrained model
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    img = images(args.src)
    stardist_time = perf_counter()
    l, d = stardist_single(img[0], model)
    stardist_time = perf_counter() - stardist_time
    print(f":: Stardist took {stardist_time:.2f}s")

    results = stats(d, cpus=args.cpus)
    area = results["area"]
    perimeter = results["perimeter"]
    n = results["number"]

    print(f"Average area: {area['mean']:.2f} \pm {area['std']/np.sqrt(n):.2f}")
    print(
        f"Average perimeter: {perimeter['mean']:.2f} \pm {perimeter['std']/np.sqrt(n):.2f}"
    )

    print(f":: Overall runtime: {perf_counter()-start:.2f}s")
    # keys: coord, points, prob
