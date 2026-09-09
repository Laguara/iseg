"""Un pas élémentaire d'apprentissage pour le modèle iSeg."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer

from .losses import cross_entropy_loss


def training_step(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    optimizer: Optimizer,
) -> float:
    """Exécute une prédiction, mesure l'erreur et met à jour les poids.

    L'ordre est important :

    1. ``zero_grad`` efface les gradients de l'itération précédente ;
    2. ``model(images)`` produit les logits ;
    3. la loss compare logits et labels ;
    4. ``backward`` calcule l'influence de chaque poids sur cette loss ;
    5. ``step`` modifie les poids avec ces informations.

    La valeur retournée est détachée du graphe de calcul. Elle sert uniquement
    à suivre l'apprentissage dans la console ou un fichier de résultats.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    loss = cross_entropy_loss(logits, targets)
    if not torch.isfinite(loss):
        raise ValueError(f"La loss doit être finie, reçu {loss.item()}")

    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
