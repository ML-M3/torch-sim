"""Wrapper for NequIP-Allegro models in TorchSim.

This module provides a TorchSim wrapper of the NequIP-Allegro models for computing
energies, forces, and stresses for atomistic systems. It integrates the NequIP-Allegro
models with TorchSim's simulation framework, handling batched computations for multiple
systems simultaneously.

The implementation supports various features including:

* Computing energies, forces, and stresses
* Batched calculations for multiple systems

References:
    - NequIP Package: https://github.com/mir-group/nequip
"""

import traceback
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ase.data
import torch

import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.typing import StateDict


try:
    from nequip.model.inference_models import load_compiled_model
    from nequip.nn import graph_model
    from nequip.scripts._compile_utils import ASE_OUTPUTS, PAIR_NEQUIP_INPUTS
except ImportError as exc:
    warnings.warn(f"NequIP import failed: {traceback.format_exc()}", stacklevel=2)

    class NequIPFrameworkModel(torch.nn.Module, ModelInterface):
        """NequIP model wrapper for torch_sim.

        This class is a placeholder for the NequIPModel class.
        It raises an ImportError if NequIP is not installed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err


class ChemicalSpeciesToAtomTypeMapper:
    """Maps atomic numbers to model-specific atom type indices.

    This class provides functionality to map atomic numbers to the corresponding atom type
    indices used by the model. It handles cases where the model's internal representation
    of atom types may differ from conventional chemical species, such as when modeling
    different charge states of the same element.

    The mapping is created using a lookup table that converts atomic numbers to
    zero-based indices based on the provided list of chemical symbols. The order of
    chemical symbols must match the order of atom types expected by the model.

    NOTE: This is adapted from the NequIP package.

    Attributes:
        lookup_table (torch.Tensor): Tensor mapping atomic numbers to model type indices.
            Contains -1 for unmapped atomic numbers.

    Args:
        chemical_symbols (list[str]): List of chemical symbols in the order matching
            the model's internal type ordering. Each symbol must be a valid chemical
            element symbol.

    Raises:
        AssertionError: If an invalid chemical symbol is provided.
    """

    def __init__(self, chemical_symbols: list[str]) -> None:  # noqa: D107
        # Create lookup table mapping atomic numbers to model type indices
        self.lookup_table = torch.full(
            (max(ase.data.atomic_numbers.values()),), -1, dtype=torch.long
        )
        for idx, sym in enumerate(chemical_symbols):
            assert sym in ase.data.atomic_numbers, f"Invalid chemical symbol {sym}"  # noqa: S101
            self.lookup_table[ase.data.atomic_numbers[sym]] = idx

    def __call__(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Convert atomic numbers to model-specific atom type indices.

        Args:
            atomic_numbers (torch.Tensor): Tensor of atomic numbers to convert.

        Returns:
            torch.Tensor: Tensor of atom type indices used by the model.
        """
        return torch.index_select(self.lookup_table, 0, atomic_numbers)


def from_compiled_model(
    compile_path: str, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, tuple[float, list[str]]]:
    """Load a compiled NequIP model from a file.

    Loads a compiled NequIP model from a file and extracts the necessary metadata
    for using it in TorchSim. The model must have been compiled using nequip-compile.

    Args:
        compile_path (str): Path to the compiled model file. The file should have been
            created using nequip-compile.
        device (str | torch.device): Device to load the model on. Can be either a string
            like 'cpu' or 'cuda', or a torch.device object. Defaults to 'cpu'.

    Returns:
        tuple[torch.nn.Module, tuple[float, list[str]]]: A tuple containing:
            - The loaded NequIP model as a torch.nn.Module
            - A tuple with:
                - r_max (float): Cutoff radius used by the model
                - type_names (list[str]): List of chemical symbols supported by the model

    Example:
        >>> model, (r_max, type_names) = from_compiled_model("model.pth", device="cuda")
        >>> print(f"Model cutoff: {r_max:.2f}")
        >>> print(f"Supported elements: {type_names}")

    References:
        For model compilation please refer to the NequIP documentation:
            https://nequip.readthedocs.io/en/latest/guide/getting-started/workflow.html#compilation
    """
    model, metadata = load_compiled_model(
        compile_path, device, PAIR_NEQUIP_INPUTS, ASE_OUTPUTS
    )

    # extract r_max and type_names for transforms
    r_max = metadata[graph_model.R_MAX_KEY]
    type_names = metadata[graph_model.TYPE_NAMES_KEY]

    return model, (r_max, type_names)


class NequIPFrameworkModel(torch.nn.Module, ModelInterface):
    """NequIP model for energy, force and stress calculations.

    This class wraps a NequIP model to compute energies, forces and stresses
    for atomic systems.

    Args:
        model (torch.nn.Module): The NequIP model to use. Must be a torch.nn.Module.
        r_max (float): Cutoff radius for neighbor list construction.
        type_names (list[str]): List of chemical symbols supported by the model.
        device (torch.device | None): Device to run calculations on.
            Defaults to CUDA if available, otherwise CPU.
        dtype (torch.dtype): Data type for calculations.
            Defaults to torch.float64.
        neighbor_list_fn (Callable): Function to compute neighbor lists.
            Defaults to vesin_nl_ts.
        atomic_numbers (torch.Tensor | None): Atomic numbers with shape [n_atoms].
            If provided at initialization, cannot be provided again during forward pass.
        batch (torch.Tensor | None): Batch indices with shape [n_atoms] indicating
            which system each atom belongs to. If not provided with atomic_numbers,
            all atoms are assumed to be in the same system.
    """

    def __init__(
        self,
        model: str | Path | torch.nn.Module | None = None,
        *,
        r_max: float,
        type_names: list[str],
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        neighbor_list_fn: Callable = vesin_nl_ts,
        atomic_numbers: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> None:
        """Initialize the NequIP model.

        Args:
            model: The NequIP model to use. Must be a torch.nn.Module.
            r_max: Cutoff radius for neighbor list construction.
            type_names: List of chemical symbols supported by the model.
            device: Device to run calculations on.
                Defaults to CUDA if available, otherwise CPU.
            dtype: Data type for calculations. Defaults to torch.float64.
            neighbor_list_fn: Function to compute neighbor lists. Defaults to vesin_nl_ts.
            atomic_numbers: Atomic numbers with shape [n_atoms]. If provided at
                initialization, cannot be provided again during forward pass.
            batch: Batch indices with shape [n_atoms] indicating which system
                each atom belongs to. If not provided with atomic_numbers, all atoms
                are assumed to be in the same system. If provided, must be a tensor
                of long integers.

        Raises:
            TypeError: If model is not a torch.nn.Module.
        """
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self.neighbor_list_fn = neighbor_list_fn
        self._memory_scales_with = "n_atoms_x_density"
        self._compute_forces = True
        self._compute_stress = True

        if isinstance(model, torch.nn.Module):
            self.model = model
        else:
            raise TypeError("Invalid model type. Must be a torch.nn.Module.")

        # Set model properties
        self.r_max = torch.tensor(r_max, dtype=self.dtype, device=self.device)
        self.type_names = type_names

        # Store flag to track if atomic numbers were provided at init
        self.atomic_numbers_in_init = atomic_numbers is not None
        self.n_systems = batch.max().item() + 1 if batch is not None else 1

        # Set up batch information if atomic numbers are provided
        if atomic_numbers is not None:
            if batch is None:
                # If batch is not provided, assume all atoms belong to same system
                batch = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self.device
                )

            self.setup_from_batch(atomic_numbers, batch)

    def setup_from_batch(self, atomic_numbers: torch.Tensor, batch: torch.Tensor) -> None:
        """Set up internal state from atomic numbers and batch indices.

        Processes the atomic numbers and batch indices to prepare the model for
        forward pass calculations. Creates the necessary data structures for
        batched processing of multiple systems.

        Args:
            atomic_numbers (torch.Tensor): Atomic numbers tensor with shape [n_atoms].
            batch (torch.Tensor): Batch indices tensor with shape [n_atoms] indicating
                which system each atom belongs to.
        """
        self.atomic_numbers = atomic_numbers
        self.batch = batch
        self.atomic_types = ChemicalSpeciesToAtomTypeMapper(self.type_names)(
            atomic_numbers
        )

        # Determine number of systems and atoms per system
        self.n_systems = batch.max().item() + 1
        self.total_atoms = atomic_numbers.shape[0]

    def forward(  # noqa: C901
        self,
        state: ts.SimState | StateDict,
    ) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the given atomic systems.

        Processes the provided state information and computes energies, forces, and
        stresses using the underlying MACE model. Handles batched calculations for
        multiple systems and constructs the necessary neighbor lists.

        Args:
            state (SimState | StateDict): State object containing positions, cell,
                and other system information. Can be either a SimState object or a
                dictionary with the relevant fields.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - 'energy': System energies with shape [n_systems]
                - 'forces': Atomic forces with shape [n_atoms, 3] if compute_forces=True
                - 'stress': System stresses with shape [n_systems, 3, 3] if
                    compute_stress=True

        Raises:
            ValueError: If atomic numbers are not provided either in the constructor
                or in the forward pass, or if provided in both places.
            ValueError: If batch indices are not provided when needed.
        """
        # Extract required data from input
        if isinstance(state, dict):
            state = ts.SimState(**state, masses=torch.ones_like(state["positions"]))

        # Handle input validation for atomic numbers
        if state.atomic_numbers is None and not self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers must be provided in either the constructor or forward."
            )
        if state.atomic_numbers is not None and self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers cannot be provided in both the constructor and forward."
            )

        # Use batch from init if not provided
        if state.batch is None:
            if not hasattr(self, "batch"):
                raise ValueError(
                    "Batch indices must be provided if not set during initialization"
                )
            state.batch = self.batch

        # Update batch information if new atomic numbers are provided
        if (
            state.atomic_numbers is not None
            and not self.atomic_numbers_in_init
            and not torch.equal(
                state.atomic_numbers,
                getattr(self, "atomic_numbers", torch.zeros(0, device=self.device)),
            )
        ):
            self.setup_from_batch(state.atomic_numbers, state.batch)

        # Process each system's neighbor list separately
        edge_indices = []
        shifts_list = []
        unit_shifts_list = []
        offset = 0

        # TODO (AG): Currently doesn't work for batched neighbor lists
        for b in range(self.n_systems):
            batch_mask = state.batch == b
            # Calculate neighbor list for this system
            edge_idx, shifts_idx = self.neighbor_list_fn(
                positions=state.positions[batch_mask],
                cell=state.row_vector_cell[b],
                pbc=state.pbc,
                cutoff=self.r_max,
            )

            # Adjust indices for the batch
            edge_idx = edge_idx + offset
            shifts = torch.mm(shifts_idx, state.row_vector_cell[b])

            edge_indices.append(edge_idx)
            unit_shifts_list.append(shifts_idx)
            shifts_list.append(shifts)

            offset += len(state.positions[batch_mask])

        # Combine all neighbor lists
        edge_index = torch.cat(edge_indices, dim=1)
        unit_shifts = torch.cat(unit_shifts_list, dim=0)
        shifts = torch.cat(shifts_list, dim=0)
        atomic_types = ChemicalSpeciesToAtomTypeMapper(self.type_names)(
            state.atomic_numbers
        )

        # Get model output
        data: dict[str, torch.Tensor] = {
            "pos": state.positions,
            "cell": state.row_vector_cell,
            "batch": state.batch,
            "pbc": torch.tensor(
                [state.pbc, state.pbc, state.pbc], dtype=torch.bool, device=self.device
            ),
            "atomic_numbers": state.atomic_numbers,
            "atom_types": atomic_types,
            "edge_index": edge_index,
            "edge_cell_shift": unit_shifts,
        }
        out = self.model(data)
        results = {}

        # Process energy
        energy = out["total_energy"]
        if energy is not None:
            results["energy"] = energy.detach()
        else:
            results["energy"] = torch.zeros(self.n_systems, device=self.device)

        # Process forces
        if self.compute_forces:
            forces = out["forces"]
            if forces is not None:
                results["forces"] = forces.detach()

        # Process stress
        if self.compute_stress:
            stress = out["stress"]
            if stress is not None:
                results["stress"] = stress.detach()

        return results
