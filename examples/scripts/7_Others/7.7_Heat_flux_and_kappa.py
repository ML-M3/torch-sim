"""Heat flux and thermal conductivity example with Lennard-Jones potential."""

# /// script
# dependencies = [
#     "ase>=3.24",
#     "matplotlib",
#     "numpy",
# ]
# ///

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.elastic import full_3x3_to_voigt_6_stress
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.properties.correlations import HeatFluxAutoCorrelation
from torch_sim.units import MetalUnits as Units
from torch_sim.units import uc


SMOKE_TEST = os.getenv("CI") is not None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# Using solid Ar w/ LJ for ease
atoms = bulk("Ar", crystalstructure="fcc", a=5.376, cubic=True)
N_repeats = 3 if SMOKE_TEST else 4
atoms = atoms.repeat((N_repeats, N_repeats, N_repeats))
state = ts.io.atoms_to_state(atoms, device=device, dtype=dtype)

# Simulation parameters
# See https://docs.lammps.org/compute_heat_flux.html for more details
epsilon = 0.0104  # eV
sigma = 3.405  # Å
cutoff = 13  # Å
temperature = 70.0  # Kelvin
timestep = 0.004  # ps (4 fs)
num_steps_equilibration = 800 if SMOKE_TEST else 8000
num_steps_production = 10000 if SMOKE_TEST else 100000
window_size = 200  # Length of correlation: dt * correlation_dt * window_size
correlation_dt = 10  # Step delta between correlations

# Lennard-Jones model
lj_model = LennardJonesModel(
    sigma=sigma,
    epsilon=epsilon,
    cutoff=cutoff,
    device=device,
    dtype=dtype,
    compute_forces=True,
    compute_stress=True,
    per_atom_energies=True,
    per_atom_stresses=True,
)

dt = torch.tensor(timestep * Units.time, device=device, dtype=dtype)
kT = torch.tensor(temperature * Units.temperature, device=device, dtype=dtype)
nvt_init, nvt_update = ts.integrators.nvt.nvt_langevin(model=lj_model, dt=dt, kT=kT)
state = nvt_init(state)

# Short equilibration run
# Shape: (num_steps, batch, dim)
heat_flux = torch.zeros((num_steps_equilibration, 3), device=device, dtype=dtype)

for step in range(num_steps_equilibration):
    state = nvt_update(state)  # type: ignore[call-arg]
    results = lj_model(state)
    J = ts.quantities.calc_heat_flux(
        momenta=state.momenta,
        masses=state.masses,
        velocities=None,
        energies=results["energies"],
        stresses=full_3x3_to_voigt_6_stress(results["stresses"]),
        batch=state.batch,
        is_centroid_stress=False,
        is_virial_only=False,
    )
    # NVT so volume is the same for both atoms
    heat_flux[step] = J / state.volume
    if step % 1000 == 0:
        print(f"Step {step} | {state.energy.item():.4f} eV")


# Convert to SI units
conv = uc.eV_to_J / uc.ps_to_s / uc.Ang2_to_met2
heat_flux_conv = (heat_flux * conv).detach().cpu().numpy()

hfacf_calc = HeatFluxAutoCorrelation(
    model=lj_model,
    window_size=window_size,
    device=device,
    use_running_average=True,
    normalize=False,
)

# Sampling freq is controlled by prop_calculators
# trajectory = "kappa_example.h5"

reporter = ts.TrajectoryReporter(
    None,  # add trajectory name here if you want to save the trajectory to disk
    state_frequency=10,
    prop_calculators={correlation_dt: {"hfacf": hfacf_calc}},
)

# Short production run
for step in range(num_steps_production):
    state = nvt_update(state)
    reporter.report(state, step)
    if step % 1000 == 0:
        print(f"Step {step} | {state.energy.item():.4f} eV")

reporter.close()

# HFACF results and plot
# Timesteps -> Time in fs
time_steps = np.arange(window_size)
time = time_steps * correlation_dt * timestep * 1000
hface_numpy = hfacf_calc.hfacf.cpu().numpy()

# Calculate kappa
integral = np.trapezoid(hface_numpy, x=time)
kappa = integral / (3 * temperature * temperature * Units.temperature)
kappa_conv = kappa * (uc.eV_to_J / uc.Ang_to_met / uc.ps_to_s)
print(f"kappa: {kappa_conv:.4f} (W/mK)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(heat_flux_conv[:, 0], "b-", linewidth=2, label=r"$J_x$")
ax1.plot(heat_flux_conv[:, 1], "r-", linewidth=2, label=r"$J_y$")
ax1.plot(heat_flux_conv[:, 2], "g-", linewidth=2, label=r"$J_z$")
ax1.set_xlabel("Time (fs)", fontsize=12)
ax1.set_ylabel(r"$J$ (W/m$^2$)", fontsize=12)
ax1.set_title("Heat Flux for Ar (LJ)", fontsize=14)
ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
ax1.legend(fontsize=12)

ax2.plot(time, hface_numpy, "b-", linewidth=2)
ax2.set_xlabel("Time (fs)", fontsize=12)
ax2.set_ylabel(r"$\langle \vec{J}(0) \cdot \vec{J}(t) \rangle$", fontsize=12)
ax2.set_title(
    rf"$\kappa$ = {kappa_conv:.4f} (W/mK) (Average of {hfacf_calc._window_count} windows)",  # noqa: E501, SLF001
    fontsize=14,
)
ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("heat_flux_and_kappa.pdf")
