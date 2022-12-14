#!/usr/bin/env python
"""Perform real space analysis of fluorescence cell nuclei to get the structure factor."""

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as scifft
from csbdeep.utils import normalize
from cv2 import fitEllipse
from dfmtoolbox._dfm_python import azimuthal_average, spatial_frequency_grid
from dfmtoolbox.io import store_data
from dfmtoolbox.utils import tiff_to_numpy
from shapely import Polygon
from stardist.models import StarDist2D

from plotting import decorate_axes, finalize_and_save
from utils import ImageCellStats, StructureFactor, chunkify, from_u_to_q, read_metadata

Details = Dict[str, np.ndarray]
Stats = Dict[str, Any]


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


def single_cell_stat(coord: np.ndarray) -> Tuple[float, float, float]:
    polygon = Polygon(coord.T)

    return abs(polygon.area), abs(polygon.length), aspect_ratio(polygon)


def aspect_ratio(poly: Polygon) -> float:
    """Fit an ellipse and return its aspect ratio."""
    coords = np.array(poly.boundary.coords, dtype=np.float32)
    *_, axes, _ = fitEllipse(
        coords
    )  # other return values are x,y center and orientation

    return max(axes) / min(axes)


def stats(details: Details, *, cpus: int = 1) -> Stats:
    # calc average cell area & std, average cell number & std
    coord = details["coord"]
    cell_number = len(coord)
    areas = np.zeros(cell_number)
    perimeters = np.zeros(cell_number)
    aspect_ratios = np.zeros(cell_number)

    for i, result in enumerate(multicore_dispatch(single_cell_stat, coord, cpus=cpus)):
        area, perim, ar = result
        areas[i] = area
        perimeters[i] = perim
        aspect_ratios[i] = ar

    stat = {
        "number": cell_number,
        "area": {"mean": areas.mean(), "std": areas.std()},
        "perimeter": {"mean": perimeters.mean(), "std": perimeters.std()},
        "aspect_ratio": {"mean": aspect_ratios.mean(), "std": aspect_ratios.std()},
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

    # doing the fft2
    fft2 = scifft.fftshift(scifft.fft2(sf, workers=workers))

    return azimuthal_average(np.abs(fft2) ** 2, dist=spatial_freqs) / norm


def structure_factor(
    images: np.ndarray,
    *,
    model: StarDist2D,
    cpus: int = 1,
    cellstats: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:

    stardist_fixed_model = partial(stardist_single, model=model)
    shape = images[0].shape
    full_sf = []
    kx = scifft.fftfreq(shape[-1])
    ky = scifft.fftfreq(shape[-2])
    spatial_freq = spatial_frequency_grid(kx, ky)

    statistics = []

    for im in images:
        _, details = stardist_fixed_model(im)
        full_sf.append(
            structure_factor_single(
                details["points"], shape=shape, workers=cpus, spatial_freqs=spatial_freq
            )
        )
        if cellstats:
            statistics.append(stats(details, cpus=cpus))

    sf = np.array(full_sf)

    return sf.mean(axis=0), sf.std(axis=0), statistics


def plot_sf(
    sf: np.ndarray,
    pars: Dict[str, Any],
    title: str,
    sf_std: Optional[np.ndarray] = None,
    figname: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 6),
) -> None:

    # square empty marker
    # marker = {"marker": "o", "facecolor": "none", "edgecolor": "k", "s": 5}
    lines = {"linestyle": "-", "color": "k"}

    # setup data
    sf = sf[1:]  # ignore q=0 value
    u = np.arange(1, len(sf) + 1)
    q = from_u_to_q(u, pars)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(q, sf, label=r"average $S(q)$", **lines)
    # ax.scatter(q, sf, **marker)
    if sf_std is not None:
        sf_std = sf_std[1:]
        ax.fill_between(q, sf - sf_std, sf + sf_std, label=r"$\sigma [S](q)$")

    ax = decorate_axes(
        ax,
        xlabel="Wave vector $q$",
        ylabel="Static structure factor $S(q)$",
        title=title,
        yscale="linear",
        xscale="linear",
        ylim=(0.0, 1.6),
    )

    if figname is None:
        figname = title.replace(" ", "_")
    finalize_and_save(fig, Path(figname))


parser = argparse.ArgumentParser()
parser.add_argument("src", help="The source TIFF file.")
parser.add_argument("--dest", help="The destination folder to store data in.")
parser.add_argument("--range", help="Comma separated image index (!) range, e.g. 1,10")
parser.add_argument(
    "--stats-only", action="store_true", help="Compute only the stats for each image."
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
parser.add_argument("--metadata", help="path to metadata config file in json format.")

args = parser.parse_args()
overlap, chunksize = args.overlap, args.chunksize

if args.metadata is not None:
    metadata = read_metadata(args.metadata)
    chunksize = metadata["chunksize"] if "chunksize" in metadata.keys() else chunksize
    overlap = metadata["overlap"] if "overlap" in metadata.keys() else overlap

else:
    chunksize = 200
    overlap = 100
    metadata = {
        "fps": 1 / 60,  # 1 frame per minute
        "magnification": 1,  # the magnification is already accounted for in the given voxel size
        "pixel_size": 1.29,
        "pixel_size_unit": "??m",
        "image_size": 672,
        "chunksize": chunksize,
        "overlap": overlap,
    }


if __name__ == "__main__":
    # timing
    start = perf_counter()

    # creates a pretrained model
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    # read images
    if args.range is not None:
        image_range: Optional[range] = range(*[int(i) for i in args.range.split(",")])
    else:
        image_range = None

    imgs = images(args.src, seq=image_range)
    nimages = len(imgs)

    # setup store path
    store_path = Path(args.dest) if args.dest is not None else Path("plots/")

    # full analysis or just stats
    if args.stats_only:
        print(":: Only performing cell statistics")
        cellstats = ImageCellStats(args.src, np.array(range(nimages)), [])

        for i, data in enumerate(imgs):
            img_time = perf_counter()
            print(f"::: working on image {i+1}/{nimages}")

            _, details = stardist_single(data, model)
            stat = stats(details, cpus=args.cpus)
            cellstats.stats.append(stat)

            print(f"    --> took {perf_counter() - img_time:.1f} s")

        print(":: Storing stats data ..")
        store_data(cellstats, path=store_path, name="cellstats.pkl")

    else:
        # setup chunks
        length = len(imgs)
        if chunksize == 0:
            chunks = [np.arange(length)]
        else:
            print(f":: Chunksize set to {chunksize}")
            chunks = chunkify(np.arange(length), chunksize=chunksize, overlap=overlap)

        for i, chunk in enumerate(chunks):
            data = imgs[chunk]  # select images based on indices

            sf_time = perf_counter()

            print(":: structure factor")
            N = len(data)
            sf, sf_std, statistics = structure_factor(
                data, model=model, cpus=args.cpus, cellstats=True
            )

            sf_time = perf_counter() - sf_time
            print(f"   --> structure factor for chunk {i} took {sf_time:.1f} s")
            plot_sf(
                sf,
                metadata,
                sf_std=sf_std,
                title=f"chunk-{i:02d} | avg. structure factor | chunksize={chunksize}",
                figname=f"plots/{i:02d}-structure_factor",
            )

            print(":: Storing data now ...")
            if len(statistics) > 0:
                store_data(
                    ImageCellStats(args.src, chunk, statistics),
                    path=store_path,
                    name=f"{i:02d}-cellstats.pkl",
                )

            u = np.arange(len(sf))
            q = from_u_to_q(u, metadata)
            store_data(
                StructureFactor(sf, q, sf_std=sf_std, size=N, notes=f"chunk {i:02d}"),
                path=store_path,
                name=f"{i:02d}-structure-factor.pkl",
            )

    print(f":: Overall runtime: {perf_counter()-start:.2f}s")
    # keys: coord, points, prob
