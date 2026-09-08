"""Fonctions minimales pour entraîner le modèle de segmentation."""

from .losses import cross_entropy_loss
from .steps import training_step

__all__ = ["cross_entropy_loss", "training_step"]
