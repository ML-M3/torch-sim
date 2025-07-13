import urllib.request
from enum import StrEnum
from pathlib import Path

import pytest
import torch

from tests.models.conftest import make_model_calculator_consistency_test


try:
    from nequip.ase import NequIPCalculator

    from torch_sim.models.nequip_framework import (
        NequIPFrameworkModel,
        from_compiled_model,
    )
except ImportError:
    pytest.skip("nequip not installed", allow_module_level=True)


class NequIPUrls(StrEnum):
    """Checkpoint download URLs for NequIP models."""

    Si = "https://github.com/abhijeetgangan/pt_model_checkpoints/raw/refs/heads/main/nequip/Si.nequip.pth"


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture(scope="session")
def model_path_nequip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("nequip_checkpoints")
    model_name = "Si.nequip.pth"
    model_path = Path(tmp_path) / model_name

    if not model_path.exists():
        urllib.request.urlretrieve(NequIPUrls.Si, model_path)  # noqa: S310

    return Path(model_path)


@pytest.fixture
def nequip_model(model_path_nequip: Path, device: torch.device) -> NequIPFrameworkModel:
    """Create an NequIPModel wrapper for the pretrained model."""
    compiled_model, (r_max, type_names) = from_compiled_model(
        model_path_nequip, device=device
    )
    return NequIPFrameworkModel(
        model=compiled_model,
        r_max=r_max,
        type_names=type_names,
        device=device,
    )


@pytest.fixture
def nequip_calculator(model_path_nequip: Path, device: torch.device) -> NequIPCalculator:
    """Create an NequIPCalculator for the pretrained model."""
    return NequIPCalculator.from_compiled_model(model_path_nequip, device=device)


def test_nequip_initialization(model_path_nequip: Path, device: torch.device) -> None:
    """Test that the NequIP model initializes correctly."""
    compiled_model, (r_max, type_names) = from_compiled_model(
        model_path_nequip, device=device
    )
    model = NequIPFrameworkModel(
        model=compiled_model,
        r_max=r_max,
        type_names=type_names,
        device=device,
    )
    assert model._device == device  # noqa: SLF001


test_nequip_consistency = make_model_calculator_consistency_test(
    test_name="nequip",
    model_fixture_name="nequip_model",
    calculator_fixture_name="nequip_calculator",
    sim_state_names=("si_sim_state", "rattled_si_sim_state"),
)

# TODO (AG): Test multi element models
