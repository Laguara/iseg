"""Produire des segmentations iSeg pour les sujets test 11 a 23.

Les donnees test n'ont pas de labels : ce script ecrit donc des predictions,
mais ne peut pas annoncer de Dice.

Exemple :
    .venv/bin/python scripts/predict_test.py \
        --data /Users/foqker/Downloads/iSeg-2017-Testing \
        --checkpoint outputs/baseline_2p5d.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.analyze import load_analyze_volume  # noqa: E402
from src.data.normalization import normalize_nonzero_intensity  # noqa: E402
from src.data.slices import build_25d_input  # noqa: E402
from src.data.validation import validate_matching_shapes  # noqa: E402
from src.models import TinyUNet25D  # noqa: E402

TEST_SUBJECT_IDS = tuple(range(11, 24))
# Le modele apprend 0, 1, 2, 3 ; ces valeurs sont celles attendues par iSeg.
INTERNAL_TO_RAW = np.array([0, 10, 150, 250], dtype=np.int16)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dossier iSeg-2017-Testing")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=TEST_SUBJECT_IDS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/test_predictions"))
    return parser.parse_args()


def choose_device(requested_device: str) -> torch.device:
    mps_available = torch.backends.mps.is_available()
    if requested_device == "mps" and not mps_available:
        raise RuntimeError("MPS a ete demande mais n'est pas disponible.")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if requested_device == "auto" and mps_available else "cpu")


def predict_subject(
    model: TinyUNet25D,
    t1: np.ndarray,
    t2: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predire toutes les coupes d'un sujet et reconstruire son volume 3D."""

    predictions = np.empty(t1.shape, dtype=np.uint8)
    with torch.no_grad():
        for first_slice in range(0, t1.shape[-1], batch_size):
            slice_indices = range(first_slice, min(first_slice + batch_size, t1.shape[-1]))
            images = np.stack([build_25d_input(t1, t2, z) for z in slice_indices])
            logits = model(torch.from_numpy(images).to(dtype=torch.float32, device=device))
            predicted_batch = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            for index, z in enumerate(slice_indices):
                predictions[..., z] = predicted_batch[index]
    return predictions


def main() -> None:
    arguments = parse_arguments()
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size doit etre strictement positif.")
    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {arguments.checkpoint}")
    invalid_subjects = set(arguments.subjects).difference(TEST_SUBJECT_IDS)
    if invalid_subjects:
        raise ValueError(f"Sujets test attendus : {TEST_SUBJECT_IDS}, recu {sorted(invalid_subjects)}")

    device = choose_device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model = TinyUNet25D().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    arguments.output.mkdir(parents=True, exist_ok=True)

    for subject_id in arguments.subjects:
        prefix = arguments.data / f"subject-{subject_id}"
        t1 = normalize_nonzero_intensity(load_analyze_volume(Path(f"{prefix}-T1.hdr")))
        t2 = normalize_nonzero_intensity(load_analyze_volume(Path(f"{prefix}-T2.hdr")))
        validate_matching_shapes(t1, t2)

        internal_prediction = predict_subject(model, t1, t2, arguments.batch_size, device)
        raw_prediction = INTERNAL_TO_RAW[internal_prediction]

        # .npy est pratique pour recharger vite ; .hdr/.img reste dans le format
        # Analyze des donnees iSeg. Le nom final peut etre adapte si le hackathon
        # impose une convention de soumission specifique.
        np.save(arguments.output / f"subject-{subject_id}-prediction-internal.npy", internal_prediction)
        nib.AnalyzeImage(raw_prediction, affine=np.eye(4)).to_filename(
            str(arguments.output / f"subject-{subject_id}-prediction.hdr")
        )
        print(f"sujet {subject_id} predit : forme={internal_prediction.shape}")

    print(f"predictions ecrites dans : {arguments.output}")
    print("Aucun Dice n'est disponible pour les sujets test sans labels.")


if __name__ == "__main__":
    main()
