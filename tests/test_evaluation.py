"""Tests des comptes utilises pour le Dice volumique."""

import torch

from src.evaluation import dice_from_counts, segmentation_counts


def test_counts_and_dice_match_a_small_known_example() -> None:
    predictions = torch.tensor([[0, 1], [2, 3]])
    targets = torch.tensor([[0, 1], [3, 3]])

    intersections, prediction_counts, target_counts = segmentation_counts(predictions, targets)
    dices = dice_from_counts(intersections, prediction_counts, target_counts)

    torch.testing.assert_close(intersections, torch.tensor([1, 1, 0, 1]))
    torch.testing.assert_close(prediction_counts, torch.tensor([1, 1, 1, 1]))
    torch.testing.assert_close(target_counts, torch.tensor([1, 1, 0, 2]))
    torch.testing.assert_close(dices, torch.tensor([1.0, 1.0, 0.0, 2.0 / 3.0]))


def test_empty_class_has_dice_one() -> None:
    counts = torch.zeros(4, dtype=torch.long)
    dices = dice_from_counts(counts, counts, counts)

    torch.testing.assert_close(dices, torch.ones(4))
