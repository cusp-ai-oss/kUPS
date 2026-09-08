# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.potential.common.geometry import PositionsAndCell, PositionsAndCellIndex
from kups.relaxation.convergence import converged_per_system, max_dof_per_system


def _cubic(diagonal: jnp.ndarray) -> PeriodicCell:
    return PeriodicCell(
        TriclinicFrame.from_matrix(diagonal[:, None, None] * jnp.eye(3)[None])
    )


def _gradients(
    n_systems: int, rows: jnp.ndarray, system: list[int], cell_grad: jnp.ndarray
) -> tuple[PositionsAndCell, PositionsAndCellIndex]:
    system_index = Index.integer(jnp.array(system), n=n_systems, label=SystemId)
    positions = Table.arange(rows, label=ParticleId)
    cells = Table.arange(_cubic(cell_grad), label=SystemId)
    prefix = PositionsAndCellIndex(system_index, cells.index)
    return PositionsAndCell(positions, cells), prefix


class TestMaxDofPerSystem:
    def test_per_system_maximum_of_row_norms(self):
        rows = jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.5]])
        grads, prefix = _gradients(2, rows, [0, 0, 1], jnp.zeros(2))
        npt.assert_allclose(
            max_dof_per_system(grads, prefix, include_cell=False).data, [5.0, 0.5]
        )

    def test_cell_gradients_take_part(self):
        rows = jnp.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.2]])
        grads, prefix = _gradients(2, rows, [0, 1], jnp.array([-7.0, 0.05]))
        with_cell = max_dof_per_system(grads, prefix).data
        without = max_dof_per_system(grads, prefix, include_cell=False).data
        npt.assert_allclose(with_cell, [7.0, 0.2])
        npt.assert_allclose(without, [0.1, 0.2])

    def test_padded_rows_and_empty_systems(self):
        # Row 2 is padding (OOB system); system 1 has no particles.
        rows = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        grads, prefix = _gradients(2, rows, [0, 0, 2], jnp.zeros(2))
        result = max_dof_per_system(grads, prefix, include_cell=False).data
        npt.assert_allclose(result[0], 2.0)
        assert result[1] == -jnp.inf

    def test_converged_matches_global_criterion_for_one_system(self):
        rows = jnp.array([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
        grads, prefix = _gradients(1, rows, [0, 0], jnp.zeros(1))
        assert bool(converged_per_system(grads, prefix, 0.05).data[0])
        assert not bool(converged_per_system(grads, prefix, 0.015).data[0])

    @pytest.mark.parametrize("bad", [jnp.nan, jnp.inf])
    def test_nonfinite_active_gradients_fail_under_jit(self, bad: float) -> None:
        grads, prefix = _gradients(1, jnp.array([[bad, 0.0, 0.0]]), [0], jnp.zeros(1))
        evaluate = jax.jit(
            as_result_function(
                lambda g: converged_per_system(g, prefix, 0.1, include_cell=False)
            )
        )
        result = evaluate(grads)
        assert not bool(result.value.data[0])
        with pytest.raises(AssertionError, match="Non-finite relaxation gradients"):
            result.raise_assertion()

    def test_nonfinite_padding_is_ignored(self) -> None:
        grads, prefix = _gradients(
            1, jnp.array([[0.0, 0.0, 0.0], [jnp.nan, 0.0, 0.0]]), [0, 1], jnp.zeros(1)
        )
        evaluate = jax.jit(
            as_result_function(
                lambda g: converged_per_system(g, prefix, 0.1, include_cell=False)
            )
        )
        result = evaluate(grads)
        result.raise_assertion()
        assert bool(result.value.data[0])

    def test_nonfinite_cell_gradient_fails_when_included(self) -> None:
        grads, prefix = _gradients(1, jnp.zeros((1, 3)), [0], jnp.array([jnp.nan]))
        evaluate = jax.jit(
            as_result_function(lambda g: converged_per_system(g, prefix, 0.1))
        )
        result = evaluate(grads)
        with pytest.raises(AssertionError, match="Non-finite relaxation gradients"):
            result.raise_assertion()
