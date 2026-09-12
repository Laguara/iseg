"""Fonctions pures pour compter et calculer le Dice de segmentation."""

from __future__ import annotations

import torch


def segmentation_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    number_of_classes: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compte intersection, prediction et cible pour chaque classe.

    Les comptes peuvent etre additionnes batch par batch. Leur Dice final est
    donc exactement le Dice obtenu sur le volume entier, sans conserver toutes
    les coupes en memoire.
    """

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions et labels doivent avoir la meme forme, recu "
            f"{tuple(predictions.shape)} et {tuple(targets.shape)}."
        )

    intersections = torch.zeros(number_of_classes, dtype=torch.long)
    prediction_counts = torch.zeros(number_of_classes, dtype=torch.long)
    target_counts = torch.zeros(number_of_classes, dtype=torch.long)

    for class_index in range(number_of_classes):
        predicted = predictions == class_index
        target = targets == class_index
        intersections[class_index] = torch.count_nonzero(predicted & target)
        prediction_counts[class_index] = torch.count_nonzero(predicted)
        target_counts[class_index] = torch.count_nonzero(target)

    return intersections, prediction_counts, target_counts


def dice_from_counts(
    intersections: torch.Tensor,
    prediction_counts: torch.Tensor,
    target_counts: torch.Tensor,
) -> torch.Tensor:
    """Calcule ``2 * intersection / (prediction + cible)`` par classe.

    Si une classe est absente a la fois de la prediction et du label, elle ne
    cree pas d'erreur artificielle et recoit Dice=1.
    """

    denominator = prediction_counts + target_counts
    dice = torch.ones_like(denominator, dtype=torch.float32)
    present = denominator > 0
    dice[present] = 2.0 * intersections[present].float() / denominator[present]
    return dice
