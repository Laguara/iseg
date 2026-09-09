"""Fonctions qui mesurent l'écart entre une prédiction et la vérité terrain."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calcule la Cross-Entropy pixel par pixel pour la segmentation.

    ``logits`` contient quatre scores bruts par pixel, de forme
    ``[batch, classes, hauteur, largeur]``. ``targets`` contient la bonne
    classe entière de chaque pixel, de forme ``[batch, hauteur, largeur]``.

    Cette première baseline utilise seulement la Cross-Entropy. Elle est plus
    simple à comprendre et à tester qu'une combinaison de pertes. Une perte
    Dice sera ajoutée séparément après validation de cette boucle complète.
    """

    _validate_segmentation_tensors(logits, targets)
    return F.cross_entropy(logits, targets)


def _validate_segmentation_tensors(logits: torch.Tensor, targets: torch.Tensor) -> None:
    """Rend les erreurs de forme et de labels explicites avant PyTorch."""

    if logits.ndim != 4:
        raise ValueError(
            "Les logits doivent avoir la forme [batch, classes, hauteur, largeur], "
            f"forme reçue : {tuple(logits.shape)}"
        )
    if targets.ndim != 3:
        raise ValueError(
            "Les labels doivent avoir la forme [batch, hauteur, largeur], "
            f"forme reçue : {tuple(targets.shape)}"
        )
    if logits.shape[0] != targets.shape[0] or logits.shape[-2:] != targets.shape[-2:]:
        raise ValueError(
            "Les dimensions batch et spatiales des logits et des labels doivent "
            f"correspondre, reçu logits={tuple(logits.shape)}, "
            f"labels={tuple(targets.shape)}"
        )
    if targets.dtype != torch.long:
        raise TypeError(
            "Les labels doivent être de type torch.long : ils représentent des "
            f"indices de classes, reçu {targets.dtype}"
        )
    if targets.numel() == 0:
        raise ValueError("Les labels ne peuvent pas être vides")

    minimum_class = int(targets.min().item())
    maximum_class = int(targets.max().item())
    if minimum_class < 0 or maximum_class >= logits.shape[1]:
        raise ValueError(
            "Les labels doivent être compris entre 0 et "
            f"{logits.shape[1] - 1}, reçu [{minimum_class}, {maximum_class}]"
        )
