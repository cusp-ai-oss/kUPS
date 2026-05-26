# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end equivalence test for the kUPS MACE adapter vs. ase MACECalculator.

Runs the same MACE checkpoint on the same atomic configuration twice:

1. Through the full kUPS pipeline: ``relax_state_from_ase`` → kUPS neighborlist
   → ``AtomGraphInput`` → ``MACEModule`` → energy/gradients.
2. Through ``mace.calculators.MACECalculator``, which builds its own graph.

Energies and forces must agree up to numerical tolerances. The two pipelines
build their neighbor lists independently — for a small enough cell the edge
sets agree, so the comparison is meaningful.

The module is skipped at collection time when ``mace-torch`` isn't installed
(default kUPS dev environment), so dedicated CI is required: install the
``mace`` extra and run the file directly, e.g.
``uv run --extra mace pytest test/potential/test_torch_mace_e2e.py``.
Also skipped when JAX is on a CPU backend — kUPS's torch wrapper uses
``torch.cuda.ExternalStream`` and only runs on GPU.

If ``KUPS_MACE_MODEL`` is set (or ``examples/mace-mpa-0-medium.model`` exists),
that checkpoint is used. Otherwise a tiny ``ScaleShiftMACE`` is constructed
with random weights and saved to a temporary ``.model`` file, which both
pipelines load — equivalence still holds by construction.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# Skip the whole module if mace isn't installed.
pytest.importorskip("mace.calculators")
torch = pytest.importorskip("torch", minversion="2.0.0")

import jax  # noqa: E402

# The kUPS torch wrapper drives the model via ``torch.cuda.ExternalStream``
# and a GPU XLA FFI callback, so the full kUPS-side pipeline only works when
# JAX is on a CUDA backend. CPU pytest runs will skip this module.
_REQUIRES_GPU = jax.default_backend() != "gpu"


def _resolve_mace_model_path() -> Path | None:
    """Look up a local MACE .model checkpoint via env var or known examples path."""
    env = os.environ.get("KUPS_MACE_MODEL")
    if env:
        p = Path(env)
        return p if p.exists() else None
    candidate = (
        Path(__file__).resolve().parents[2] / "examples" / "mace-mpa-0-medium.model"
    )
    return candidate if candidate.exists() else None


def _build_random_mace(
    atomic_numbers: list[int],
    cutoff: float,
    dtype,
    seed: int,
):
    """Construct a tiny ``ScaleShiftMACE`` with randomly initialised weights.

    The model is intentionally small (32 channels, 2 interaction layers) to
    keep the e2e test fast. It supports the species in ``atomic_numbers`` only.
    """
    import torch.nn.functional as F
    from e3nn import o3
    from mace import modules

    torch.manual_seed(seed)
    num_elements = len(atomic_numbers)
    model = modules.ScaleShiftMACE(
        r_max=float(cutoff),
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=num_elements,
        hidden_irreps=o3.Irreps("32x0e + 32x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=F.silu,
        atomic_energies=np.zeros(num_elements, dtype=float),
        avg_num_neighbors=8.0,
        atomic_numbers=list(atomic_numbers),
        correlation=2,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )
    return model.to(dtype=dtype).eval()


def _resolve_or_create_mace_checkpoint(tmp_path: Path) -> tuple[Path, list[int]]:
    """Return a path to a MACE checkpoint and the list of supported Z values.

    Uses ``KUPS_MACE_MODEL`` / ``examples/`` when available, otherwise builds a
    tiny random ``ScaleShiftMACE`` and persists it to ``tmp_path``.
    """
    real = _resolve_mace_model_path()
    if real is not None:
        # We don't know the real model's species offhand. Cu is in every common
        # MACE foundation model (mace-mpa, mace-omat) so we keep using it.
        return real, [29]

    atomic_numbers = [29]  # Cu — matches the triclinic test cell below.
    model = _build_random_mace(
        atomic_numbers=atomic_numbers,
        cutoff=5.0,
        dtype=torch.float32,
        seed=0,
    )
    path = tmp_path / "mace_random.model"
    torch.save(model, path)
    return path, atomic_numbers


@pytest.mark.skipif(
    _REQUIRES_GPU,
    reason="kUPS torch wrapper requires JAX on CUDA",
)
def test_kups_pipeline_matches_mace_calculator(tmp_path):
    """kUPS-driven MACE matches ase MACECalculator on the same point cloud.

    Both pipelines load the same checkpoint at float32. For a small bulk cell
    with cutoff 6 Å, the two graphs cover the same edges (modulo ordering),
    so energy, forces and stress should agree to numerical tolerance.
    """
    import ase
    import jax.numpy as jnp
    from mace.calculators import MACECalculator

    from kups.application.relaxation.data import relax_state_from_ase
    from kups.application.simulations.relax_torch import RelaxTorchState
    from kups.core.cell import to_lower_triangular
    from kups.core.lens import identity_lens
    from kups.core.neighborlist import UniversalNeighborlistParameters
    from kups.core.result import as_result_function
    from kups.observables.stress import _stress_via_virial_theorem
    from kups.potential.mliap.torch import load_mace, make_torch_mliap_from_state

    model_path, _ = _resolve_or_create_mace_checkpoint(tmp_path)

    # Same triclinic Cu cell as the UMA e2e test — lower-triangular with
    # positive diagonal so QR is trivial (Q = I), exercising the non-diagonal
    # h_kups path through the lower-tri stress recovery in
    # ``stress_via_virial_theorem``.
    rng = np.random.default_rng(0)
    atoms = ase.Atoms(
        "Cu",
        positions=[[0.0, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [1.5, 4.5, 0.0], [0.7, 0.4, 5.2]],
        pbc=True,
    ).repeat((2, 2, 2))
    atoms.positions += 0.05 * rng.standard_normal(atoms.positions.shape)

    # --- Reference: ase MACECalculator ---
    atoms.calc = MACECalculator(
        model_paths=str(model_path), device="cuda", default_dtype="float32"
    )
    mc_energy = float(atoms.get_potential_energy())
    mc_forces = np.asarray(atoms.get_forces())
    mc_stress = np.asarray(atoms.get_stress(voigt=False))  # (3, 3), eV/Å³

    # --- kUPS pipeline ---
    mliap = load_mace(
        model_path,
        device="cuda",
        dtype="float32",
        compute_cell_gradients=True,
    )
    particles, systems = relax_state_from_ase(atoms)
    nl_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, mliap.cutoff
    )
    state = RelaxTorchState(
        particles=particles,
        systems=systems,
        neighborlist_params=nl_params,
        opt_state=jnp.array(0),  # unused by the potential's forward pass
        step=jnp.array([0]),
        torch_mliap_model=mliap,
    )
    potential = make_torch_mliap_from_state(
        identity_lens(RelaxTorchState), compute_position_and_cell_gradients=True
    )
    potential_fn = as_result_function(potential)
    state = potential_fn(state, None).fix_or_raise(state)
    out = potential_fn(state, None)
    out.raise_assertion()
    pot_out = out.value.data

    kups_energy = float(np.asarray(pot_out.total_energies.data).sum())
    kups_pos_grad = np.asarray(pot_out.gradients.positions.data)  # (N, 3)
    kups_forces_triclinic = -kups_pos_grad

    # ``relax_state_from_ase`` rotates positions into a lower-triangular cell
    # frame via QR. ``uc_transform`` is ``x ↦ Q^T @ x`` (covariant), so
    # forces obey ``f_kups = Q^T @ f_ase`` ⇒ ``f_ase = Q @ f_kups``. Extract Q
    # by sending each basis vector through ``uc_transform``: row i of the
    # result is row i of Q.
    _, uc_transform = to_lower_triangular(jnp.asarray(atoms.cell.array))
    Q = np.asarray(jnp.stack([uc_transform(jnp.eye(3)[i]) for i in range(3)], axis=0))
    kups_forces = kups_forces_triclinic @ Q.T  # batched f_ase = Q @ f_kups

    # Stress in the kUPS (lower-triangular) frame, recovered from
    # ``tril(∂E/∂h)`` and ``h`` via ``_stress_via_virial_theorem`` (lower
    # triangle of ``S = h^T·∂E/∂h``, symmetry fills the upper). The gradients
    # are pulled from ``pot_out`` because ``torch_mliap_model_fn`` returns an
    # ``IdPatch`` and does not write them onto ``state``. Rotate to the ASE
    # frame (σ_ase = Q·σ_kups·Q^T) and flip the sign: kUPS uses
    # σ = -(sym(pos⊗∂U/∂r) + h^T·∂U/∂h)/V, while ASE/MACE report -σ_kUPS
    # (cf. ``test_stress_matches_ase`` in ``test/md/test_integrators.py``).
    kups_stress_triclinic = np.asarray(
        _stress_via_virial_theorem(
            position_gradients=pot_out.gradients.positions.data,
            vector_gradients=pot_out.gradients.cell.data.vectors,
            positions=state.particles.data.positions,
            vectors=state.systems.data.cell.vectors,
            system=state.particles.data.system,
        )[0]
    )
    kups_stress = -Q @ kups_stress_triclinic @ Q.T

    # Both pipelines call the same float32 model; tolerances loosened ~10× over
    # the observed gap.
    np.testing.assert_allclose(kups_energy, mc_energy, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(kups_forces, mc_forces, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(kups_stress, mc_stress, atol=1e-5, rtol=1e-4)
