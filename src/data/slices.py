"""2.5D slice construction utilities."""

import numpy as np

from .validation import validate_matching_shapes


def build_25d_input(t1: np.ndarray, t2: np.ndarray, z: int) -> np.ndarray:
    """Build a six-channel 2.5D input around slice ``z``.

    Channels are ordered as T1(z-1), T1(z), T1(z+1), followed by
    T2(z-1), T2(z), T2(z+1). At a volume boundary, the closest available
    slice is repeated.
    """

    validate_matching_shapes(t1, t2)
    _validate_3d_volume(t1, "T1")
    _validate_slice_index(z, t1.shape[-1])

    previous_index = max(z - 1, 0)
    next_index = min(z + 1, t1.shape[-1] - 1)
    return np.stack(
        (
            t1[..., previous_index],
            t1[..., z],
            t1[..., next_index],
            t2[..., previous_index],
            t2[..., z],
            t2[..., next_index],
        ),
        axis=0,
    )


def extract_central_label_slice(labels: np.ndarray, z: int) -> np.ndarray:
    """Extract the target label slice at position ``z``."""

    _validate_3d_volume(labels, "labels")
    _validate_slice_index(z, labels.shape[-1])
    return labels[..., z]


def _validate_3d_volume(volume: np.ndarray, name: str) -> None:
    if volume.ndim != 3:
        raise ValueError(
            f"{name} volume must be 3-dimensional, received shape {volume.shape}"
        )


def _validate_slice_index(z: int, depth: int) -> None:
    if not isinstance(z, (int, np.integer)):
        raise TypeError(f"Slice index must be an integer, received: {z}")
    if z < 0 or z >= depth:
        raise IndexError(f"Slice index out of range: {z} for depth {depth}")
