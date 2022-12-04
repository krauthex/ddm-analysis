#!/usr/bin/env python
"""Perform real space analysis of fluorescence cell nuclei to get the structure factor."""

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
from csbdeep.utils import normalize

# import matplotlib.pyplot as plt
from dfmtoolbox.utils import tiff_to_numpy
from stardist.models import StarDist2D

# from stardist.plot import render_label


# stardist setup
# prints a list of available models
# StarDist2D.from_pretrained()

Details = Dict[str, np.ndarray]


def stardist_single(image: np.ndarray, model: StarDist2D, norm: bool = True) -> Any:
    """Apply stardist to a single image and return the labels & details objects."""
    if norm:
        image = normalize(image)
    labels, details = model.predict_instances(image)
    return labels, details


def images(path: Path) -> np.ndarray:
    return tiff_to_numpy(path)


def area(coord: np.ndarray) -> float:
    pass


def stats(details: Details) -> Any:
    # calc average cell area & std, average cell number & std
    pass


parser = argparse.ArgumentParser()
parser.add_argument("src", help="The source TIFF file.")

args = parser.parse_args()

if __name__ == "__main__":
    # creates a pretrained model
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    img = images(args.src)
    l, d = stardist_single(img[0], model)
    print(d, d.keys())
