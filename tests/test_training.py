"""Tests de la Cross-Entropy et du pas d'apprentissage minimal."""

import pytest
import torch

from src.models import TinyUNet25D
from src.training import cross_entropy_loss, training_step


def test_cross_entropy_returns_a_finite_scalar() -> None:
    """Une batch valide doit produire une unique erreur finie."""

    logits = torch.randn(2, 4, 8, 10)
    targets = torch.randint(0, 4, (2, 8, 10), dtype=torch.long)

    loss = cross_entropy_loss(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_cross_entropy_rejects_non_integer_targets() -> None:
    """Les classes ne sont pas des valeurs continues ou flottantes."""

    logits = torch.randn(1, 4, 8, 8)
    targets = torch.zeros(1, 8, 8, dtype=torch.float32)

    with pytest.raises(TypeError, match="torch.long"):
        cross_entropy_loss(logits, targets)


def test_cross_entropy_rejects_out_of_range_targets() -> None:
    """Avec quatre classes, la valeur 4 ne peut pas représenter un label."""

    logits = torch.randn(1, 4, 8, 8)
    targets = torch.full((1, 8, 8), 4, dtype=torch.long)

    with pytest.raises(ValueError, match="compris entre 0 et 3"):
        cross_entropy_loss(logits, targets)


def test_training_step_updates_at_least_one_parameter() -> None:
    """Backward puis optimizer.step doivent réellement modifier le modèle."""

    torch.manual_seed(0)
    model = TinyUNet25D(base_channels=2)
    images = torch.randn(2, 6, 32, 32)
    targets = torch.randint(0, 4, (2, 32, 32), dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = training_step(model, images, targets, optimizer)

    assert loss > 0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters())
    )
