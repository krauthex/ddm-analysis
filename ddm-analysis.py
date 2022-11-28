#!/usr/bin/env python
"""Perform DDM analysis using dfmtoolbox on cell movies and store the results in a binary format."""

import argparse
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Iterator, List, Tuple

import numpy as np
import tifffile
from dfmtoolbox._dfm_python import run
from dfmtoolbox.io import store_data
from dfmtoolbox.utils import tiff_to_numpy

from utils import AnalysisBlob


def get_fluorescence_tiff_paths(folder: Path) -> Iterator[Path]:
    """Return the 2nd, 4th, .. file in a given folder.

    In this peculiar case this corresponds to the fluorescence microscopy movies of MDCK cells.
    """
    return islice(
        (file for file in sorted(folder.iterdir()) if file.is_file()), 1, None, 2
    )


def create_folderstructure(location: Path, name: str) -> Tuple[Path, Path]:
    """Create a folderstructure for analysis files at `location` with containing folder name `name`, if it not already exists.

    Returns the Path objects for the datastore and the plots directory.
    """
    datastore = location / name
    datastore.mkdir(parents=True, exist_ok=True)
    plots = datastore / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    return datastore, plots


def prepare_analysis(folder: Path) -> None:
    """Create folderstructure for all TIFF files in a given folder."""
    for tiffpath in get_fluorescence_tiff_paths(folder):
        name = tiffpath.name.replace(".tif", "")
        create_folderstructure(tiffpath.parent, "{}-analysis".format(name))


def get_dataset_dims(path: Path) -> Tuple[int, int, int]:
    """Return the dataset dimensions without loading it into RAM."""
    with tifffile.TiffFile(path) as tif:
        length: int = len(tif.pages)
        shape: Tuple[int, int] = tif.pages[0].shape

    return (length, *shape)


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


def process_single_tiff(src: Path) -> None:
    """Process a single tiff file.

    Parameters
    ----------
    src : Path
        The path to the tiff file.
    """

    process_single_time = perf_counter()
    print(f"\n:: Working on file: {src}")

    # folders ...
    print(":: Creating folder structure ...")
    if not isinstance(src, Path):
        src = Path(src)

    datastore, plots = create_folderstructure(
        src.parent, f"{src.name.replace('.tif', '')}-analysis"
    )

    # chunking setup
    length, y, x = get_dataset_dims(src)
    tiff_indices = np.arange(length)

    for i, chunk in enumerate(chunkify(tiff_indices, chunksize, overlap)):
        chunk_calc_time = perf_counter()
        print(f":: [chunk={i}] Processing image range [{chunk[0]}-{chunk[-1]}]")

        data = tiff_to_numpy(src, seq=chunk)
        """
        if y != x:
            square_dim = min(y, x)
            print(
                f"::: dimensions not square, cropping to {square_dim}x{square_dim}..."
            )
            data = data[:, :square_dim, :square_dim]
        """
        print("::: Performing analysis now ...")
        rfft2_sqmod, azimuthal_avg, dqt = run(data, lags, keep_full_structure=True)
        if rfft2_sqmod.dtype != np.float64:
            rfft2_sqmod = rfft2_sqmod.astype(np.float64)

        print(f"::: Analysis took {perf_counter() - chunk_calc_time:.2f} s")

        print("::: Creating data structure ... ")
        blob = AnalysisBlob(
            data_source=src,
            rfft2_sqmod=rfft2_sqmod,
            lags=lags,
            image_structure_function=dqt,
            azimuthal_average=azimuthal_avg,
            metadata=metadata,
            notes=notes_template.format(chunk=i, normalized=True, windowed=False),
        )

        print("::: Writing datastructure to binary file ... ")
        store_data(blob, path=datastore, name=f"chunk-{i:03d}")

    print(
        f":: processing duration for {src.name}: {perf_counter() - process_single_time:.2f} s."
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "src", metavar="SRC", nargs="+", help="Location of TIFF file(s) to be processed."
)

args = parser.parse_args()

chunksize = 200
overlap = 100
lag_pct = 0.6
lags = np.arange(1, int(lag_pct * chunksize))

notes_template = "chunk: {chunk}; normalized: {normalized}; windowed: {windowed}"

metadata = {
    "fps": 1 / 60,  # 1 frame per minute
    "magnification": 1,  # the magnification is already accounted for in the given voxel size
    "pixel_size": 1.29,
    "pixel_size_unit": "µm",
    "image_size": 512,  # always assume square images; see preparation below
    "chunksize": chunksize,
    "overlap": overlap,
    "fraction_total_lags": lag_pct,  # the fraction of total lags to use
}


if __name__ == "__main__":
    total_time_start = perf_counter()

    src = args.src
    tiff_paths = [Path(tiff) for tiff in args.src]

    # info output
    print(":: Processing the following files:")
    for tiffpath in tiff_paths:
        print(f":: --> {tiffpath.name}")

    for tiff in tiff_paths:
        process_single_tiff(tiff)

    overall_time = perf_counter() - total_time_start
    print(
        f"\n:: Overall processing took {int(overall_time // 60)} min {overall_time % 60:.2f} s."
    )
