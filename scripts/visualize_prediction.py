"""Créer une figure : IRM, label réel et prédiction d'une coupe de validation.

Exemple :
    .venv/bin/python scripts/visualize_prediction.py \
        --data /Users/foqker/Downloads/iSeg-2017-Training \
        --checkpoint outputs/baseline_2p5d.pt \
        --subject 9 --slice 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.dataset import ISeg25DSliceDataset  # noqa: E402
from src.data.splits import VALIDATION_SUBJECT_IDS  # noqa: E402
from src.models import TinyUNet25D  # noqa: E402

CLASS_NAMES = ("Fond", "LCR", "SG", "SB")
CLASS_COLORS = ("black", "deepskyblue", "orange", "limegreen")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subject", type=int, default=9, choices=VALIDATION_SUBJECT_IDS)
    parser.add_argument("--slice", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/validation_example.png"))
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
    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {arguments.checkpoint}")

    # Le Dataset retourne les coupes du sujet dans l'ordre z=0, 1, ..., Z-1.
    dataset = ISeg25DSliceDataset(arguments.data, subject_ids=(arguments.subject,))
    if not 0 <= arguments.slice < len(dataset):
        raise ValueError(f"--slice doit etre entre 0 et {len(dataset) - 1}.")

    image, label = dataset[arguments.slice]
    device = choose_device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model = TinyUNet25D().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu()

    figure, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    axes[0].imshow(image[1], cmap="gray")  # Canal 1 : coupe T1 centrale.
    axes[0].set_title(f"T1, patient {arguments.subject}, coupe {arguments.slice}")

    color_map = ListedColormap(CLASS_COLORS)
    axes[1].imshow(label, cmap=color_map, vmin=0, vmax=3)
    axes[1].set_title("Label reel")
    axes[2].imshow(prediction, cmap=color_map, vmin=0, vmax=3)
    axes[2].set_title("Prediction")

    for axis in axes:
        axis.axis("off")

    figure.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=color) for color in CLASS_COLORS],
        labels=CLASS_NAMES,
        loc="lower center",
        ncol=4,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=150)
    print(f"figure ecrite dans : {arguments.output}")


if __name__ == "__main__":
    main()
