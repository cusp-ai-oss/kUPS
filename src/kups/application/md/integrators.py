# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructor for MD integration steps.

Dispatches on an integrator key to the state-agnostic step factories in
[kups.md.integrators][], binding particle and system views from a concrete
simulation state and wrapping them with a periodic-boundary-aware flow.

Integrator parameters may live on the state (``systems.data.integrator_params``)
or be passed directly via ``parameters=``; in the latter case they are overlaid
onto the systems view with a constant lens and the state need not carry them.
"""

from __future__ import annotations

from typing import Any, Literal, overload, override

from kups.core.cell import Periodic3D, require_periodic_3d_triclinic
from kups.core.data import Table
from kups.core.lens import BaseLens, Lens, bind
from kups.core.propagator import Propagator
from kups.core.typing import HasCell, IsState, SystemId
from kups.core.utils.jax import dataclass, field
from kups.md.integrators import (
    Integrator,
    IsBAOABLangevinParams,
    IsBAOABNPTLangevinParams,
    IsBAOABNPTLangevinSystem,
    IsBAOABNPTLangevinSystemBase,
    IsCSVRNPTParams,
    IsCSVRParams,
    IsMDSystem,
    IsMDSystemNPT,
    IsMDSystemNPTBase,
    IsVerletParams,
    WrapFlow,
    _MDParticleData,
    euclidean_flow,
    make_baoab_langevin_step,
    make_baoab_npt_langevin_step,
    make_csvr_npt_step,
    make_csvr_step,
    make_velocity_verlet_step,
)


@dataclass
class _IntegratorParamsOverlay(BaseLens[Any, Any]):
    """Systems lens that overlays a constant ``integrator_params`` on read.

    ``get`` returns the underlying systems table with its ``integrator_params``
    replaced by the constant; ``set`` delegates to the base lens so dynamic
    system fields (cell, cell momentum, …) are written back to the state.
    """

    base: Lens[Any, Any] = field(static=True)
    parameters: Any

    @override
    def get(self, state: Any) -> Any:
        return bind(self.base.get(state), lambda t: t.data.integrator_params).apply(
            lambda _: self.parameters
        )

    @override
    def set(self, state: Any, value: Any) -> Any:
        return self.base.set(state, value)


def require_baoab_npt_langevin_state(
    systems: Table[SystemId, IsBAOABNPTLangevinSystem],
) -> None:
    """Runtime check that the system table satisfies the BAOAB NPT Langevin shape.

    Call this once on the initial state (outside of jit) before constructing
    the integrator. Verifies that the cell is a 3D-periodic
    :class:`TriclinicFrame` and that the integrator-params is a
    :class:`BAOABNPTLangevinParams`-shaped bundle.
    """
    require_periodic_3d_triclinic(systems.data.cell)


@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsMDSystem[IsVerletParams]]],
    derivative_computation: Propagator[State],
    integrator: Literal["verlet"],
    *,
    parameters: None = None,
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsMDSystem[IsBAOABLangevinParams]]],
    derivative_computation: Propagator[State],
    integrator: Literal["baoab_langevin"],
    *,
    parameters: None = None,
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsMDSystem[IsCSVRParams]]],
    derivative_computation: Propagator[State],
    integrator: Literal["csvr"],
    *,
    parameters: None = None,
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsMDSystemNPT]],
    derivative_computation: Propagator[State],
    integrator: Literal["csvr_npt"],
    *,
    parameters: None = None,
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsBAOABNPTLangevinSystem]],
    derivative_computation: Propagator[State],
    integrator: Literal["baoab_npt_langevin"],
    *,
    parameters: None = None,
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, HasCell[Periodic3D]]],
    derivative_computation: Propagator[State],
    integrator: Literal["verlet"],
    *,
    parameters: IsVerletParams,
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, HasCell[Periodic3D]]],
    derivative_computation: Propagator[State],
    integrator: Literal["baoab_langevin"],
    *,
    parameters: IsBAOABLangevinParams,
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, HasCell[Periodic3D]]],
    derivative_computation: Propagator[State],
    integrator: Literal["csvr"],
    *,
    parameters: IsCSVRParams,
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsMDSystemNPTBase]],
    derivative_computation: Propagator[State],
    integrator: Literal["csvr_npt"],
    *,
    parameters: IsCSVRNPTParams,
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsBAOABNPTLangevinSystemBase]],
    derivative_computation: Propagator[State],
    integrator: Literal["baoab_npt_langevin"],
    *,
    parameters: IsBAOABNPTLangevinParams,
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, Any],
    derivative_computation: Propagator[State],
    integrator: Integrator,
    *,
    parameters: Any = None,
) -> Propagator[State]: ...


def make_md_step_from_state[State](
    state: Lens[State, Any],
    derivative_computation: Propagator[State],
    integrator: Integrator,
    *,
    parameters: Any = None,
) -> Propagator[State]:
    """Build a single MD integration step from a typed state.

    Constructs the appropriate integrator propagator by extracting views for
    particles and systems from ``state`` and wrapping them with a
    [WrapFlow][kups.md.integrators.WrapFlow]
    for periodic-boundary-condition-aware distance computations.

    Supported integrators:

    - ``"verlet"`` — [Velocity Verlet][kups.md.integrators.make_velocity_verlet_step]
      (NVE ensemble, no thermostat). Requires ``integrator_params: IsVerletParams``.
    - ``"baoab_langevin"`` — [BAOAB Langevin][kups.md.integrators.make_baoab_langevin_step]
      (NVT via Langevin friction/noise). Requires
      ``integrator_params: IsBAOABLangevinParams``.
    - ``"csvr"`` — [CSVR][kups.md.integrators.make_csvr_step]
      (NVT via canonical-sampling velocity rescaling, constant volume). Requires
      ``integrator_params: IsCSVRParams``.
    - ``"csvr_npt"`` — [CSVR-NPT][kups.md.integrators.make_csvr_npt_step]
      (NPT via CSVR thermostat with barostat). Requires
      ``integrator_params: IsCSVRNPTParams`` and a ``cell_gradients`` leaf on each system.
    - ``"baoab_npt_langevin"`` — [BAOAB NPT Langevin][kups.md.integrators.make_baoab_npt_langevin_step]
      (Gao–Fang–Wang JCP 2016 fully-flexible-cell NPT). Requires
      ``integrator_params: IsBAOABNPTLangevinParams`` plus ``cell_momentum``
      and ``cell_gradients`` leaves on each system, and a
      :class:`~kups.core.cell.TriclinicFrame` cell.

    The overloads narrow the required state protocol per ``integrator`` literal.

    Args:
        state: Lens into the sub-state with ``particles`` and ``systems``.
        derivative_computation: Propagator that computes forces/gradients and
            updates the state (e.g. a wrapped potential).
        integrator: String key selecting the integration algorithm.
        parameters: Constant integrator parameters. When given they are overlaid
            onto the systems view with a constant lens, so the state need not
            carry ``integrator_params`` on each system.

    Returns:
        [Propagator][kups.core.propagator.Propagator] that advances the
        simulation by one time step.

    Raises:
        ValueError: If ``integrator`` is not one of the supported keys.
    """
    flow = WrapFlow(
        state.focus(
            lambda x: x.systems.map_data(lambda x: x.cell.materialize())[
                x.particles.data.system
            ]
        ),
        euclidean_flow,
    )
    particles_lens = state.focus(lambda x: x.particles)
    systems_lens: Lens[State, Any] = state.focus(lambda x: x.systems)
    if parameters is not None:
        systems_lens = _IntegratorParamsOverlay(systems_lens, parameters)
    match integrator:
        case "verlet":
            return make_velocity_verlet_step(
                particles_lens, systems_lens, derivative_computation, flow
            )
        case "baoab_langevin":
            return make_baoab_langevin_step(
                particles_lens, systems_lens, derivative_computation, flow
            )
        case "csvr":
            return make_csvr_step(
                particles_lens, systems_lens, derivative_computation, flow
            )
        case "csvr_npt":
            return make_csvr_npt_step(
                particles_lens, systems_lens, derivative_computation, flow
            )
        case "baoab_npt_langevin":
            return make_baoab_npt_langevin_step(
                particles_lens, systems_lens, derivative_computation, flow
            )
        case _:
            raise ValueError(f"Unknown integrator: {integrator}")
