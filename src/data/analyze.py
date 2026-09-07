"""Loading utilities for Analyze image pairs."""

from os import PathLike
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np


def load_analyze_volume(header_path: Union[str, PathLike]) -> np.ndarray:
    """Load an Analyze ``.hdr``/``.img`` pair without value preprocessing.

    Parameters
    ----------
    header_path:
        Path to the ``.hdr`` file. The paired ``.img`` file must be next to it
        and have the same stem.

    Returns
    -------
    numpy.ndarray
        The volume values from the Analyze image, with only a trailing
        singleton dimension removed.
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

    volume = np.asanyarray(image.dataobj)
    return remove_trailing_singleton_dimension(volume)


def remove_trailing_singleton_dimension(volume: np.ndarray) -> np.ndarray:
    """Remove only a final dimension when its size is one.

    All other dimensions and voxel values are left unchanged.
    """

    if volume.ndim == 0 or volume.shape[-1] != 1:
        return volume

    return np.squeeze(volume, axis=-1)
