# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared check that torch MLFF backends ignore padding rows and idle slots."""

from __future__ import annotations

import ase
import numpy as np

from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.potential.mliap.torch import make_torch_mliap_from_state
from kups.application.relaxation.streaming import (
    StreamingRelaxState,
    estimate_neighborlist_params,
    idle_payload,
    make_streaming_relax_state,
    payload_from_structures,
    slot_payload_lens,
)
from kups.application.utils.particles import particles_from_ase
from kups.core import units
from kups.core.lens import identity_lens
from kups.core.result import as_result_function
from kups.core.stream import reserve_slots
from kups.core.utils.jax import tree_concat
from kups.potential.mliap.torch import TorchMliap
from test.units import EV, FORCE


def assert_padding_and_idle_slots_do_not_change_results(
    mliap: TorchMliap, atoms: ase.Atoms
) -> None:
    """``atoms`` in a padded slot next to an idle slot evaluates as it does alone.

    Compares energy, position gradients and cell gradients of one exactly sized
    slot with a slot of twice the rows placed after an idle slot, then
    evaluates two idle slots.
    """
    particles, cell, _ = particles_from_ase(atoms)
    structure = (particles, cell)
    species = tuple(particles.data.labels.keys)
    n_atoms = len(atoms)
    potential_fn = as_result_function(
        make_torch_mliap_from_state(
            identity_lens(StreamingRelaxState[tuple[()]]),
            parameters=mliap,
            gradient=POSITIONS_AND_CELL,
        )
    )

    def evaluate(
        n_slots: int, capacity: int, slot: int | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Energies, occupied rows' position gradients and cell gradients."""
        slots = reserve_slots(n_slots, capacity)
        idle = idle_payload(
            capacity,
            species,
            cell.periodic,
            units.max(mliap.cutoff.data),
            particles.data.positions.dtype,
        )
        state = make_streaming_relax_state(
            slots,
            idle,
            lambda particles, systems: (),
            estimate_neighborlist_params(
                [structure], capacity, species, mliap.cutoff, multiplier=2.0
            ),
        )
        if slot is not None:
            rows = [idle] * n_slots
            rows[slot] = payload_from_structures([structure], capacity, species, [0])
            state = slot_payload_lens(slots).set(state, tree_concat(*rows))
        state = potential_fn(state, None).fix_or_raise(state)
        out = potential_fn(state, None)
        out.raise_assertion()
        gradients = out.value.data.gradients
        return (
            np.asarray(out.value.data.total_energies.data.to(EV)),
            np.asarray(gradients.positions.data.to(FORCE))[
                np.asarray(state.particles.data.system.valid_mask)
            ],
            np.asarray(
                state.systems.data.cell.frame.vectors_gradient(
                    gradients.cell.data.frame
                ).to(FORCE)
            ),
        )

    energy, position_gradients, cell_gradients = evaluate(1, n_atoms, 0)
    padded = evaluate(2, 2 * n_atoms, 1)
    np.testing.assert_allclose(padded[0], [0.0, energy[0]], atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(padded[1], position_gradients, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(padded[2][1], cell_gradients[0], atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(padded[2][0], 0.0)

    idle_energy, _, idle_cell_gradients = evaluate(2, 2 * n_atoms, None)
    np.testing.assert_array_equal(idle_energy, 0.0)
    np.testing.assert_array_equal(idle_cell_gradients, 0.0)
