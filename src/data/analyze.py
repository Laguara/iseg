"""Loading utilities for Analyze image pairs."""

from os import PathLike
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np


def load_analyze_volume(header_path: Union[str, PathLike]) -> np.ndarray:
    """Load an Analyze ``.hdr``/``.img`` pair without preprocessing.

    Parameters
    ----------
    header_path:
        Path to the ``.hdr`` file. The paired ``.img`` file must be next to it
        and have the same stem.

    Returns
    -------
    numpy.ndarray
        The volume as represented by the Analyze image, retaining its original
        shape and values.
    """

    path = Path(header_path)
    if path.suffix.lower() != ".hdr":
        raise ValueError(f"Expected an Analyze .hdr path, received: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Analyze header file not found: {path}")

    image_path = path.with_suffix(".img")
    if not image_path.is_file():
        raise FileNotFoundError(f"Analyze image file not found: {image_path}")

    try:
        image = nib.load(str(path))
    except nib.filebasedimages.ImageFileError as error:
        raise ValueError(f"Could not read Analyze pair: {path}") from error

    return np.asanyarray(image.dataobj)
