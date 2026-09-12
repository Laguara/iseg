"""Entraîner la baseline 2.5D iSeg sur de vraies IRM.

Les sujets 1 a 8 servent a apprendre. Les sujets 9 et 10 servent uniquement
a choisir le meilleur checkpoint : ils ne participent jamais a ``backward``.

Exemple :
    .venv/bin/python scripts/train_baseline.py \
        --data /Users/foqker/Downloads/iSeg-2017-Training \
        --epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

# Permet de lancer ce fichier depuis ``scripts/`` sans installer le projet.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.dataset import ISeg25DSliceDataset  # noqa: E402
from src.data.splits import TRAIN_SUBJECT_IDS, VALIDATION_SUBJECT_IDS  # noqa: E402
from src.models import TinyUNet25D, count_trainable_parameters  # noqa: E402
from src.training import cross_entropy_loss, training_step  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dossier iSeg-2017-Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="auto choisit MPS sur Mac si disponible, sinon CPU.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/baseline_2p5d.pt"),
        help="Checkpoint du meilleur epoch selon la loss de validation.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="0 = toutes les coupes. Utile avec 1 pour un test technique rapide.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=0,
        help="0 = toutes les coupes de validation.",
    )
    return parser.parse_args()


def choose_device(requested_device: str) -> torch.device:
    """Choisit un appareil disponible sans masquer un mauvais choix explicite."""

    mps_available = torch.backends.mps.is_available()
    if requested_device == "mps":
        if not mps_available:
            raise RuntimeError("MPS a ete demande mais n'est pas disponible sur cette machine.")
        return torch.device("mps")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if mps_available else "cpu")


def limited_batches(loader: DataLoader, maximum: int) -> Iterable:
    """Retourne toutes les batches, ou seulement les ``maximum`` premieres."""

    for batch_number, batch in enumerate(loader, start=1):
        if maximum and batch_number > maximum:
            break
        yield batch


def mean_validation_loss(
    model: TinyUNet25D,
    loader: DataLoader,
    device: torch.device,
    maximum_batches: int,
) -> float:
    """Mesure l'erreur sur 9-10 sans gradients ni modification des poids."""

    model.eval()
    total_loss = 0.0
    total_images = 0

    with torch.no_grad():
        for images, labels in limited_batches(loader, maximum_batches):
            images = images.to(device)
            labels = labels.to(device)
            loss = cross_entropy_loss(model(images), labels)
            total_loss += loss.item() * images.shape[0]
            total_images += images.shape[0]

    if total_images == 0:
        raise ValueError("La validation ne contient aucune batch.")
    return total_loss / total_images


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("--epochs et --batch-size doivent etre strictement positifs.")
    if arguments.max_train_batches < 0 or arguments.max_validation_batches < 0:
        raise ValueError("Les limites de batches doivent etre positives ou nulles.")

    torch.manual_seed(arguments.seed)
    device = choose_device(arguments.device)

    # Ces deux datasets restent separes par patient : aucune coupe de 9 ou 10
    # ne peut entrer dans l'optimiseur pendant l'entrainement.
    train_dataset = ISeg25DSliceDataset(arguments.data, TRAIN_SUBJECT_IDS)
    validation_dataset = ISeg25DSliceDataset(arguments.data, VALIDATION_SUBJECT_IDS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = TinyUNet25D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"appareil={device} | poids={count_trainable_parameters(model):,}")
    print(
        f"train: sujets {TRAIN_SUBJECT_IDS}, {len(train_dataset)} coupes | "
        f"validation: sujets {VALIDATION_SUBJECT_IDS}, {len(validation_dataset)} coupes"
    )

    best_validation_loss = float("inf")
    for epoch in range(1, arguments.epochs + 1):
        total_train_loss = 0.0
        total_train_images = 0

        # ``training_step`` contient le cycle zero_grad -> forward -> loss ->
        # backward -> optimizer.step. Ici, il est applique a chaque vraie batch.
        for images, labels in limited_batches(train_loader, arguments.max_train_batches):
            images = images.to(device)
            labels = labels.to(device)
            loss = training_step(model, images, labels, optimizer)
            total_train_loss += loss * images.shape[0]
            total_train_images += images.shape[0]

        if total_train_images == 0:
            raise ValueError("L'entrainement ne contient aucune batch.")

        train_loss = total_train_loss / total_train_images
        validation_loss = mean_validation_loss(
            model,
            validation_loader,
            device,
            arguments.max_validation_batches,
        )
        print(
            f"epoch={epoch:02d}/{arguments.epochs} "
            f"train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}"
        )

        # On conserve le checkpoint qui generalise le mieux sur les patients 9-10.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "train_subject_ids": TRAIN_SUBJECT_IDS,
                    "validation_subject_ids": VALIDATION_SUBJECT_IDS,
                    "model": "TinyUNet25D",
                    "input_channels": 6,
                    "number_of_classes": 4,
                },
                arguments.output,
            )
            print(f"checkpoint sauvegarde : {arguments.output}")

    print(f"meilleure validation_loss={best_validation_loss:.4f}")
    print("Prochaine etape : calculer le Dice complet par patient avec ce checkpoint.")


if __name__ == "__main__":
    main()
