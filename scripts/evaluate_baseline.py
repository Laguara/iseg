"""Evaluer un checkpoint iSeg avec un Dice complet par patient.

Exemple :
    .venv/bin/python scripts/evaluate_baseline.py \
        --data /Users/foqker/Downloads/iSeg-2017-Training \
        --checkpoint outputs/baseline_2p5d.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.dataset import ISeg25DSliceDataset  # noqa: E402
from src.data.splits import VALIDATION_SUBJECT_IDS  # noqa: E402
from src.evaluation import dice_from_counts, segmentation_counts  # noqa: E402
from src.models import TinyUNet25D  # noqa: E402

CLASS_NAMES = ("fond", "LCR", "SG", "SB")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dossier iSeg-2017-Training")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation_dice.json"),
        help="Resultats JSON, ignore par Git avec le dossier outputs/.",
    )
    return parser.parse_args()


def choose_device(requested_device: str) -> torch.device:
    mps_available = torch.backends.mps.is_available()
    if requested_device == "mps" and not mps_available:
        raise RuntimeError("MPS a ete demande mais n'est pas disponible.")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if requested_device == "auto" and mps_available else "cpu")


def main() -> None:
    arguments = parse_arguments()
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size doit etre strictement positif.")
    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {arguments.checkpoint}")

    device = choose_device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model = TinyUNet25D().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ISeg25DSliceDataset(arguments.data, VALIDATION_SUBJECT_IDS)
    loader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False, num_workers=0)

    # Une entree est ajoutee pour chaque patient. On accumule les comptes de
    # toutes ses coupes : le Dice final porte donc sur son volume entier.
    counts = {
        subject_id: {
            "intersection": torch.zeros(4, dtype=torch.long),
            "prediction": torch.zeros(4, dtype=torch.long),
            "target": torch.zeros(4, dtype=torch.long),
        }
        for subject_id in VALIDATION_SUBJECT_IDS
    }

    offset = 0
    with torch.no_grad():
        for images, labels in loader:
            batch_size = images.shape[0]
            subject_ids = [dataset.sample_index[offset + index][0] for index in range(batch_size)]
            offset += batch_size

            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for index, subject_id in enumerate(subject_ids):
                intersection, prediction, target = segmentation_counts(
                    predictions[index], labels[index]
                )
                counts[subject_id]["intersection"] += intersection
                counts[subject_id]["prediction"] += prediction
                counts[subject_id]["target"] += target

    results: dict[str, object] = {
        "checkpoint": str(arguments.checkpoint),
        "patients": {},
    }
    foreground_dices = []
    for subject_id in VALIDATION_SUBJECT_IDS:
        subject_counts = counts[subject_id]
        dices = dice_from_counts(
            subject_counts["intersection"],
            subject_counts["prediction"],
            subject_counts["target"],
        )
        subject_result = {
            class_name: round(float(dice), 6)
            for class_name, dice in zip(CLASS_NAMES, dices)
        }
        subject_result["mean_foreground_dice"] = round(float(dices[1:].mean()), 6)
        results["patients"][str(subject_id)] = subject_result
        foreground_dices.append(dices[1:].mean())
        formatted = " | ".join(f"{name}={value:.4f}" for name, value in subject_result.items())
        print(f"patient {subject_id} : {formatted}")

    results["mean_foreground_dice"] = round(
        float(torch.stack(foreground_dices).mean()), 6
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"moyenne patients 9-10 : {results['mean_foreground_dice']:.4f}")
    print(f"resultats ecrits dans : {arguments.output}")


if __name__ == "__main__":
    main()
