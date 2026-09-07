"""Validation utilities for subject volumes."""

from typing import Optional

import numpy as np


def validate_matching_shapes(
    t1: np.ndarray,
    t2: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> None:
    """Validate that subject volumes share the same shape.

    Labels are optional because test subjects do not have annotations.
    """

    shapes = {"T1": t1.shape, "T2": t2.shape}
    if labels is not None:
        shapes["labels"] = labels.shape

    if len({shape for shape in shapes.values()}) != 1:
        details = ", ".join(f"{name}={shape}" for name, shape in shapes.items())
        raise ValueError(f"Volume shapes do not match: {details}")
