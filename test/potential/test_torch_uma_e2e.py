# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end equivalence test for the kUPS UMA adapter vs. fairchem UMA.

Runs UMA on the same atomic configuration twice:

1. Through the full kUPS pipeline: ``relax_state_from_ase`` → kUPS neighborlist
   → ``AtomGraphInput`` → ``UMAModule`` → energy/gradients.
2. Through fairchem's own ``FAIRChemCalculator``, which builds its own graph.

Energies and forces must agree up to numerical tolerances. The two pipelines
build their neighbor lists independently — for a small enough cell the edge
sets agree, so the comparison is meaningful.

The module is skipped at collection time when ``fairchem-core`` isn't
installed (default kUPS dev environment), so dedicated CI is required:
install the ``uma`` extra and run the file directly, e.g.
``uv run --extra uma pytest test/potential/test_torch_uma_e2e.py``.
Also skipped when JAX is on a CPU backend — kUPS's torch wrapper uses
``torch.cuda.ExternalStream`` and only runs on GPU.

If ``KUPS_UMA_MODEL`` is set (or ``examples/uma-s-1p2.pt`` exists), that
checkpoint is used. Otherwise a tiny non-MoE ``eSCNMDBackbone`` is constructed
with random weights and persisted to a temporary ``MLIPInferenceCheckpoint``
``.pt`` file — both pipelines load that fresh checkpoint and equivalence
still holds by construction.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# Skip the whole module if fairchem (real, not stub) isn't installed.
pytest.importorskip("fairchem.core")
torch = pytest.importorskip("torch", minversion="2.0.0")

import jax  # noqa: E402

# The kUPS torch wrapper drives the model via ``torch.cuda.ExternalStream``
# and a GPU XLA FFI callback, so the full kUPS-side pipeline only works when
# JAX is on a CUDA backend. CPU pytest runs will skip this module.
_REQUIRES_GPU = jax.default_backend() != "gpu"


def _resolve_uma_model_path() -> Path | None:
    """Look up a local UMA .pt checkpoint via env var or known examples path."""
    env = os.environ.get("KUPS_UMA_MODEL")
    if env:
        p = Path(env)
        return p if p.exists() else None
    candidate = Path(__file__).resolve().parents[2] / "examples" / "uma-s-1p2.pt"
    return candidate if candidate.exists() else None


def _build_random_uma_checkpoint(
    out_path: Path,
    task_name: str,
    cutoff: float,
    seed: int,
) -> Path:
    """Synthesise a tiny non-MoE eSCNMDBackbone checkpoint with random weights.

    Constructs a minimal HydraModel (eSCNMDBackbone + MLP_EFS_Head) end-to-end
    compatible with ``MLIPPredictUnit``, wraps the weights in an
    ``AveragedModel`` to populate ``ema_state_dict`` (which is what
    ``load_inference_model(use_ema=True)`` consumes), and writes an
    ``MLIPInferenceCheckpoint`` to ``out_path``.
    """
    # Trigger registry registration for ``escnmd_backbone`` and
    # ``MLP_EFS_Head`` — these classes only land in the fairchem registry as
    # a side effect of importing the module.
    import fairchem.core.models.uma.escn_md  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import hydra
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint

    torch.manual_seed(seed)

    model_config = {
        "_target_": "fairchem.core.models.base.HydraModel",
        # pass_through=True so the EFS head's per-property dicts land under
        # the top-level keys ``energy`` / ``forces`` / ``stress``, which is
        # where ``_process_outputs`` indexes by ``task.name``.
        "pass_through_head_outputs": True,
        "supports_single_atoms": True,
        "otf_graph": False,
        "model_id": "TestRandomUMA-0.1",
        "backbone": {
            "model": "escnmd_backbone",
            "max_num_elements": 100,
            "sphere_channels": 16,
            "hidden_channels": 16,
            "edge_channels": 16,
            "num_distance_basis": 16,
            "num_layers": 1,
            "lmax": 1,
            "mmax": 1,
            "cutoff": float(cutoff),
            "max_neighbors": 50,
            "otf_graph": False,
            "use_pbc": True,
            "use_pbc_single": True,
            "regress_forces": True,
            "regress_stress": True,
            "direct_forces": False,
            "direct_stress": False,
            "act_type": "gate",
            "ff_type": "spectral",
            "norm_type": "rms_norm_sh",
            "chg_spin_emb_type": "rand_emb",
            "cs_emb_grad": True,
            "dataset_emb_grad": True,
            "dataset_mapping": {task_name: task_name},
            "use_dataset_embedding": True,
            "charge_balanced_channels": [0],
            "spin_balanced_channels": [1],
        },
        "heads": {
            "efs": {
                "module": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "wrap_property": True,
            },
        },
    }

    # Tasks: name == output key from MLP_EFS_Head; property == name; one dataset.
    # The Normalizer with mean=0, rmsd=1 is a no-op denormalize.
    _task_levels = {"energy": "system", "forces": "atom", "stress": "system"}
    _task_dims = {"energy": [1], "forces": [3], "stress": [3, 3]}
    tasks_config = [
        {
            "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
            "name": prop,
            "level": _task_levels[prop],
            "property": prop,
            "out_spec": {
                "_target_": "fairchem.core.units.mlip_unit.mlip_unit.OutputSpec",
                "dim": _task_dims[prop],
                "dtype": "float32",
            },
            "normalizer": {
                "_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
                "mean": 0.0,
                "rmsd": 1.0,
            },
            "datasets": [task_name],
        }
        for prop in ("energy", "forces", "stress")
    ]

    # Build the model via hydra.utils.instantiate so the saved state_dict keys
    # match what ``load_inference_model`` will create on the receiving side.
    model = hydra.utils.instantiate(model_config)
    model.eval()
    model_state_dict = model.state_dict()

    # ``load_inference_model(use_ema=True)`` wraps with AveragedModel and reads
    # an ``ema_state_dict`` with a ``module.`` prefix and an ``n_averaged``
    # field. Mirror that shape here.
    averaged = torch.optim.swa_utils.AveragedModel(model)
    ema_state_dict = averaged.state_dict()

    # ``MLIPInferenceCheckpoint`` annotates ``tasks_config`` as ``dict`` but
    # real checkpoints store a list (one task config per entry) and the rest
    # of fairchem treats it as iterable — pyright catches the discrepancy.
    checkpoint = MLIPInferenceCheckpoint(
        model_config=model_config,
        model_state_dict=model_state_dict,
        ema_state_dict=ema_state_dict,
        tasks_config=tasks_config,  # pyright: ignore[reportArgumentType]
    )
    torch.save(checkpoint, out_path)
    return out_path


def _resolve_or_create_uma_checkpoint(tmp_path: Path, task_name: str) -> Path:
    """Return a path to a UMA-compatible checkpoint, building one if needed."""
    real = _resolve_uma_model_path()
    if real is not None:
        return real
    return _build_random_uma_checkpoint(
        tmp_path / "uma_random.pt", task_name=task_name, cutoff=6.0, seed=0
    )


@pytest.mark.skipif(
    _REQUIRES_GPU,
    reason="kUPS torch wrapper requires JAX on CUDA",
)
def test_kups_pipeline_matches_fairchem_uma(tmp_path):
    """kUPS-driven UMA matches fairchem's UMA on the same point cloud.

    The kUPS adapter forces ``external_graph_gen=True`` and feeds UMA the
    kUPS-built graph, while fairchem's calculator builds its own. For a small
    bulk cell with cutoff 6 Å, the two graphs cover the same edges (modulo
    ordering), so energy and forces should agree to numerical tolerance.
    """
    import ase
    import jax.numpy as jnp
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    from kups.application.relaxation.data import relax_state_from_ase
    from kups.application.simulations.relax_torch import RelaxTorchState
    from kups.core.cell import to_lower_triangular
    from kups.core.lens import identity_lens
    from kups.core.neighborlist import UniversalNeighborlistParameters
    from kups.core.result import as_result_function
    from kups.observables.stress import _stress_via_virial_theorem
    from kups.potential.mliap.torch import load_uma, make_torch_mliap_from_state

    task_name = "omat"
    model_path = _resolve_or_create_uma_checkpoint(tmp_path, task_name=task_name)

    # Triclinic test cell: same shape as ``test_neighborlist_ase.py``'s
    # ``_make_triclinic`` (a lower-triangular non-diagonal cell that's already
    # validated against ASE's neighbour list), tiled to 8 atoms so forces are
    # non-zero. The cell is already lower-triangular with positive diagonal,
    # so QR is trivial (Q = I) and ``to_lower_triangular``'s sign fix is
    # vacuously satisfied. The off-diagonal entries in the lower triangle
    # exercise the lower-triangle stress recovery in
    # ``_stress_via_virial_theorem``.
    rng = np.random.default_rng(0)
    atoms = ase.Atoms(
        "Cu",
        positions=[[0.0, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [1.5, 4.5, 0.0], [0.7, 0.4, 5.2]],
        pbc=True,
    ).repeat((2, 2, 2))
    atoms.positions += 0.05 * rng.standard_normal(atoms.positions.shape)

    # --- Reference: fairchem's FAIRChemCalculator ---
    pu_ref = load_predict_unit(
        path=str(model_path), device="cuda", inference_settings="default"
    )
    atoms.calc = FAIRChemCalculator(pu_ref, task_name=task_name)
    fc_energy = float(atoms.get_potential_energy())
    fc_forces = np.asarray(atoms.get_forces())
    fc_stress = np.asarray(atoms.get_stress(voigt=False))  # (3, 3), eV/Å³

    # --- kUPS pipeline ---
    mliap = load_uma(
        model_path,
        device="cuda",
        task_name=task_name,
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
    # frame via QR. The cell above is already lower-triangular, so Q = I, but
    # we apply the rotation explicitly so the test is robust to cell
    # orientation. ``uc_transform`` is ``x ↦ Q^T @ x`` (covariant), so forces
    # obey ``f_kups = Q^T @ f_ase`` ⇒ ``f_ase = Q @ f_kups``. Extract Q by
    # sending each basis vector through ``uc_transform``: row i of the result
    # is row i of Q.
    _, uc_transform = to_lower_triangular(jnp.asarray(atoms.cell.array))
    Q = np.asarray(jnp.stack([uc_transform(jnp.eye(3)[i]) for i in range(3)], axis=0))
    kups_forces = kups_forces_triclinic @ Q.T  # batched f_ase = Q @ f_kups

    # Stress in the kUPS (lower-triangular) frame, recovered from
    # ``tril(∂E/∂h)`` and ``h`` via ``_stress_via_virial_theorem`` (lower
    # triangle of ``S = h^T·∂E/∂h``, symmetry fills the upper). The gradients
    # are pulled from ``pot_out`` because ``torch_mliap_model_fn`` returns an
    # ``IdPatch`` and does not write them onto ``state``. Rotate to the ASE
    # frame (σ_ase = Q·σ_kups·Q^T) and flip the sign: kUPS uses
    # σ = -(sym(pos⊗∂U/∂r) + h^T·∂U/∂h)/V, while ASE/UMA report -σ_kUPS
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

    # Both pipelines call the same float32 model; agreement should be at
    # float32 precision (~1e-7 energy, ~1e-6 forces, ~1e-6 stress empirically).
    # Tolerances are loosened ~10× over the observed gap.
    np.testing.assert_allclose(kups_energy, fc_energy, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(kups_forces, fc_forces, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(kups_stress, fc_stress, atol=1e-5, rtol=1e-4)
