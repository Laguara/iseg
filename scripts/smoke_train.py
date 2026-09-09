"""Vérifier la boucle d'apprentissage sur des données artificielles.

Ce script ne mesure pas la qualité médicale du modèle. Il construit une tâche
triviale, dont la réponse est volontairement encodée dans l'entrée, puis vérifie
que les kernels peuvent apprendre à faire baisser la Cross-Entropy.

Exemple :
    python scripts/smoke_train.py --steps 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models import TinyUNet25D  # noqa: E402
from src.training import training_step  # noqa: E402


def create_synthetic_batch(
    batch_size: int = 2, height: int = 32, width: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crée une tâche facile pour tester l'algorithme, pas les données iSeg.

    Chaque pixel reçoit une classe selon son quadrant. Les quatre premiers
    canaux indiquent directement cette classe, avec un peu de bruit. Le modèle
    doit donc pouvoir mémoriser cette relation si ``backward`` et l'optimiseur
    fonctionnent correctement.
    """

    rows = torch.arange(height).view(height, 1)
    columns = torch.arange(width).view(1, width)
    targets_one_image = (rows >= height // 2).long() * 2 + (
        columns >= width // 2
    ).long()
    targets = targets_one_image.expand(batch_size, -1, -1).clone()

    images = torch.zeros(batch_size, 6, height, width)
    for class_index in range(4):
        images[:, class_index] = (targets == class_index).float()

    # Les canaux 4 et 5 imitent des informations supplémentaires, sans être
    # nécessaires pour résoudre cette tâche de vérification.
    images[:, 4:] = 0.05 * torch.randn(batch_size, 2, height, width)
    return images, targets


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.steps <= 0:
        raise ValueError("--steps doit être strictement positif")

    torch.manual_seed(0)
    images, targets = create_synthetic_batch()
    model = TinyUNet25D()
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)

    first_loss = None
    last_loss = None
    for step in range(1, arguments.steps + 1):
        last_loss = training_step(model, images, targets, optimizer)
        if first_loss is None:
            first_loss = last_loss
        if step == 1 or step == arguments.steps or step % 10 == 0:
            print(f"step={step:03d} loss={last_loss:.4f}")

    assert first_loss is not None and last_loss is not None
    print(f"loss_initiale={first_loss:.4f} loss_finale={last_loss:.4f}")


if __name__ == "__main__":
    main()
