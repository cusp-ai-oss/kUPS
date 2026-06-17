# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""``lattice_gradient_from_virial`` recovers ``∂E/∂h`` from a symmetric virial.

A torch MLFF backend returns only the symmetric strain virial (= stress·volume).
For a rotationally-invariant energy the lattice gradient is still recoverable in
full: the position virial is known from forces and positions, and in kUPS's row
convention (``r = frac @ h``) the cell virial is ``(∂E/∂h)^T @ h``, so
``∂E/∂h = h^-T @ (virial - pos_virial^T)``. Pairing the virial with the
transposed ``cell^T @ ∂E/∂h`` instead (and symmetrising the position virial)
biases the antisymmetric part — invisible to the symmetric stress but corrupting
cell relaxation on sheared cells. This is a CPU check; the GPU e2e tests only
exercise stress.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

torch = pytest.importorskip("torch")
ase_build = pytest.importorskip("ase.build")

from kups.application.potential.classical.lennard_jones import (  # noqa: E402
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import POSITIONS_AND_CELL  # noqa: E402
from kups.application.relaxation.data import relax_state_from_ase  # noqa: E402
from kups.application.simulations.relax_lj import RelaxLjState  # noqa: E402
from kups.core.lens import identity_lens  # noqa: E402
from kups.core.neighborlist import UniversalNeighborlistParameters  # noqa: E402
from kups.potential.classical.lennard_jones import LennardJonesParameters  # noqa: E402
from kups.potential.mliap.torch.interface import (  # noqa: E402
    lattice_gradient_from_virial,
)

SIGMA, EPS, RC = 3.405, 0.010326, 4.5


def _sheared_lj():
    """Rattled, lower-triangular triclinic Ar cell — genuine shear stress."""
    atoms = ase_build.bulk("Ar", "fcc", a=5.3, cubic=True) * (2, 2, 2)
    atoms.rattle(0.2, seed=1)
    atoms.set_cell(
        np.array([[10.6, 0.0, 0.0], [0.3, 10.6, 0.0], [0.2, 0.25, 10.6]]),
        scale_atoms=True,
    )
    return atoms


def test_recovers_partial_lattice_gradient_under_shear():
    atoms = _sheared_lj()
    particles, systems = relax_state_from_ase(atoms)
    lj = LennardJonesParameters.from_dict(
        cutoff=RC, parameters={"Ar": (SIGMA, EPS)}, mixing_rule="lorentz_berthelot"
    )
    nlp = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, lj.cutoff
    )
    state = RelaxLjState(particles, systems, nlp, jnp.zeros(()), jnp.array([0]), lj)

    # Direct autodiff partial gradient (atoms pinned): the reference ∂E/∂h and ∂E/∂r.
    out = make_lennard_jones_from_state(
        identity_lens(RelaxLjState), gradient=POSITIONS_AND_CELL
    )(state).data
    g_r = np.asarray(out.gradients.positions.data)
    frame = systems.data.cell.frame
    dEdh_true = np.asarray(frame.vectors_gradient(out.gradients.cell.data.frame)[0])
    r = np.asarray(particles.data.positions)
    h = np.asarray(systems.data.cell.vectors[0])

    # Symmetric total virial a backend would report (stress · volume).
    total = g_r.T @ r + dEdh_true.T @ h
    assert np.abs(total - total.T).max() < 1e-10  # rotational invariance
    virial = 0.5 * (total + total.T)

    recovered = np.asarray(
        lattice_gradient_from_virial(
            forces=torch.tensor(-g_r),
            positions=torch.tensor(r),
            batch=torch.zeros(len(r), dtype=torch.long),
            cell=torch.tensor(h)[None],
            virial=torch.tensor(virial)[None],
        )[0]
    )
    np.testing.assert_allclose(recovered, dEdh_true, atol=1e-10)
