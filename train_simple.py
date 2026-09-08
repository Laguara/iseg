"""Premier entraînement iSeg, volontairement réuni dans un seul fichier.

Lancer depuis la racine du dépôt :

    .venv/bin/python train_simple.py \
        --data /Users/foqker/Downloads/iSeg-2017-Training

Ce premier lancement utilise le vrai sujet 1, cinq petits batchs et ne cherche
pas encore à produire une performance médicale. Il vérifie le cycle complet :
vraies IRM -> modèle -> perte -> gradients -> mise à jour des poids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Ce Dataset est le travail de la partie données. Il charge les fichiers .hdr/.img,
# normalise T1/T2, transforme les labels 0/10/150/250 en 0/1/2/3 et fabrique
# les six canaux 2.5D. On ne réécrit pas cette logique ici.
from src.data.dataset import ISeg25DSliceDataset


class SimpleUNet25D(nn.Module):
    """Très petit U-Net : une descente, une remontée et un seul skip.

    Entrée : [batch, 6, hauteur, largeur]
    Sortie : [batch, 4, hauteur, largeur] : un score par classe et par pixel.

    Les paramètres des Conv2d sont automatiquement enregistrés comme poids du
    modèle : par exemple ``self.encoder_conv_1.weight``. C'est pour cela que le
    réseau reste une classe, même si l'entraînement tient dans ``main``.
    """

    def __init__(self) -> None:
        super().__init__()

        # Encodeur : 6 images d'entrée deviennent 8 cartes apprises.
        self.encoder_conv_1 = nn.Conv2d(6, 8, kernel_size=3, padding=1)
        self.encoder_conv_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Partie compacte : H et W sont divisés par deux, avec 16 cartes.
        self.bottleneck_conv_1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bottleneck_conv_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        # Décodeur : après le skip, il y a 16 + 8 = 24 cartes à combiner.
        self.decoder_conv_1 = nn.Conv2d(24, 8, kernel_size=3, padding=1)
        self.decoder_conv_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        # 1x1 : chaque pixel reçoit quatre scores (fond, LCR, SG, SB).
        self.classifier = nn.Conv2d(8, 4, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Exécute le chemin complet d'une image vers ses quatre scores."""

        # [B, 6, H, W] -> [B, 8, H, W]
        encoder = F.relu(self.encoder_conv_1(images))
        encoder = F.relu(self.encoder_conv_2(encoder))

        # MaxPool diminue seulement la hauteur et la largeur : H/2, W/2.
        compact = self.pool(encoder)
        compact = F.relu(self.bottleneck_conv_1(compact))
        compact = F.relu(self.bottleneck_conv_2(compact))  # [B, 16, H/2, W/2]

        # On réagrandit l'information globale, puis on lui redonne les détails
        # locaux conservés dans ``encoder``. cat ajoute des canaux, pas des pixels.
        enlarged = F.interpolate(
            compact,
            size=encoder.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        joined = torch.cat((enlarged, encoder), dim=1)  # [B, 24, H, W]
        decoded = F.relu(self.decoder_conv_1(joined))
        decoded = F.relu(self.decoder_conv_2(decoded))

        return self.classifier(decoded)  # [B, 4, H, W], des logits


def main() -> None:
    """Lit les vrais batchs et effectue les étapes d'apprentissage dans l'ordre."""

    parser = argparse.ArgumentParser(description="Premier entraînement iSeg simple.")
    parser.add_argument("--data", type=Path, required=True, help="Dossier iSeg-2017-Training")
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1],
        help="Sujets utilisés. Par défaut : 1, pour comprendre avant de généraliser.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=5,
        help="0 = toutes les coupes. Par défaut : 5 vrais batchs, pour un premier essai rapide.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.max_batches < 0:
        raise ValueError("epochs et batch-size doivent être positifs ; max-batches doit être >= 0.")

    # 1. Le Dataset charge les volumes réels ; le DataLoader rassemble plusieurs
    # coupes en un batch. images=[B,6,H,W], labels=[B,H,W].
    dataset = ISeg25DSliceDataset(args.data, subject_ids=args.subjects)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 2. Le modèle contient les poids ; Adam sait comment les modifier.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleUNet25D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(weight.numel() for weight in model.parameters())

    print(f"Données : {len(dataset)} coupes des sujets {list(args.subjects)}")
    print(f"Appareil : {device} | poids à apprendre : {parameter_count:,}")

    # 3. Voici la boucle entière d'apprentissage. Chaque passage dans cette
    # boucle est exactement le cycle expliqué : batch -> forward -> loss ->
    # backward -> modification des poids.
    for epoch in range(1, args.epochs + 1):
        model.train()

        for batch_number, (images, labels) in enumerate(loader, start=1):
            if args.max_batches and batch_number > args.max_batches:
                break

            # Les images et labels doivent être sur le même appareil que les poids.
            images = images.to(device)
            labels = labels.to(device)

            # A. Oublier les gradients du batch précédent.
            optimizer.zero_grad(set_to_none=True)

            # B. Forward : les poids actuels fabriquent quatre scores par pixel.
            logits = model(images)

            # C. Les labels 0=fond, 1=LCR, 2=SG, 3=SB disent quelle classe est vraie.
            # Cross-Entropy compare les scores et les labels sur tous les pixels.
            loss = F.cross_entropy(logits, labels)

            # D. Backward : chaque poids reçoit maintenant un gradient dans .grad.
            loss.backward()

            # E. Adam lit ces gradients et modifie réellement les poids.
            optimizer.step()

            print(
                f"epoch {epoch}/{args.epochs} | batch {batch_number} | "
                f"images {tuple(images.shape)} | loss {loss.item():.4f}"
            )

            # Une fois, rendre visible l'endroit exact où PyTorch range un gradient.
            if epoch == 1 and batch_number == 1:
                print(
                    "exemple : encoder_conv_1.weight = "
                    f"{tuple(model.encoder_conv_1.weight.shape)}, gradient = "
                    f"{tuple(model.encoder_conv_1.weight.grad.shape)}"
                )


if __name__ == "__main__":
    main()
