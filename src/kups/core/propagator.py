# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State propagators for simulation dynamics and Monte Carlo moves.

This module provides a composable framework for evolving simulation states over
time. Propagators represent any operation that transitions a state from one
configuration to another.

Key components:

- **[Propagator][kups.core.propagator.Propagator]**: Protocol for state evolution functions
- **[propagator_with_assertions][kups.core.propagator.propagator_with_assertions]**: Wrap a propagator to track assertion results
- **[propagate_and_fix][kups.core.propagator.propagate_and_fix]**: Retry propagation until assertions pass
- **[MCMCPropagator][kups.core.propagator.MCMCPropagator]**: Metropolis-Hastings Monte Carlo with acceptance/rejection
- **[SequentialPropagator][kups.core.propagator.SequentialPropagator]**: Chain multiple propagators sequentially
- **[PalindromePropagator][kups.core.propagator.PalindromePropagator]**: Reversible composition maintaining detailed balance
- **[LoopPropagator][kups.core.propagator.LoopPropagator]**: Repeat a propagator multiple times
- **[SwitchPropagator][kups.core.propagator.SwitchPropagator]**: Randomly select from multiple propagators
- **[ResetOnErrorPropagator][kups.core.propagator.ResetOnErrorPropagator]**: Rollback state on assertion failures
- **[ScheduledPropertyPropagator][kups.core.propagator.ScheduledPropertyPropagator]**: Update properties according to schedules

Propagators are composable and JIT-compilable, enabling efficient simulation loops.
"""

import logging
from typing import Callable, Protocol

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.assertion import check_assertions
from kups.core.data.table import Table
from kups.core.lens import Lens, Update, View
from kups.core.patch import Addable, Patch, WithPatch
from kups.core.result import Result, as_result_function
from kups.core.schedule import IncrementSchedule, Schedule, Scheduler
from kups.core.typing import SystemId
from kups.core.utils.jax import (
    dataclass,
    field,
    jit,
    key_chain,
    tree_map,
    tree_structure,
    tree_where_broadcast_last,
)
from kups.core.utils.ops import select_n


class StateProperty[State, Property](Protocol):
    """Protocol for functions that extract properties from states.

    Type Parameters:
        State: Simulation state type
        Property: Type of property to extract
    """

    def __call__(self, key: Array, state: State) -> Property: ...


@dataclass
class StatePropertySum[State, Property: Addable]:
    """Sum multiple state properties together.

    Attributes:
        properties: Tuple of property extractors to sum
    """

    properties: tuple[StateProperty[State, Property], ...] = field(static=True)

    def __call__(self, key: Array, state: State) -> Property:
        chain = key_chain(key)
        result = self.properties[0](next(chain), state)
        for func in self.properties[1:]:
            new_result = func(next(chain), state)
            result += new_result
        return result


class Propagator[State](Protocol):
    """Protocol for state evolution functions.

    A propagator takes a random key and current state, returning an updated state.
    Propagators can represent time evolution (MD integrators), Monte Carlo moves,
    or any other state transformation.

    Type Parameters:
        State: Simulation state type

    Example:
        ```python
        class MyPropagator:
            def __call__(self, key, state):
                # Evolve state
                return updated_state

        # Use in simulation loop
        key = jax.random.PRNGKey(0)
        state = initial_state
        for i in range(1000):
            key, subkey = jax.random.split(key)
            state = propagator(subkey, state)
        ```
    """

    def __call__(self, key: Array, state: State, /) -> State:
        """Propagate the state forward.

        Args:
            key: JAX PRNG key for stochastic operations
            state: Current simulation state

        Returns:
            Updated state after propagation
        """
        ...


def propagator_with_assertions[State](
    propagator: Propagator[State],
) -> Callable[[Array, State], Result[State, State]]:
    """Wrap a propagator to capture assertion results alongside the state.

    Args:
        propagator: Propagator to wrap.

    Returns:
        Function returning a Result that pairs the new state with assertion metadata.
    """
    return as_result_function(propagator)


def propagate_and_fix[State](
    fn: Callable[[Array, State], Result[State, State]],
    key: Array,
    state: State,
    *,
    max_tries: int = 10,
) -> State:
    """Execute a propagator repeatedly until all assertions pass or retries are exhausted.

    On each attempt, failed assertions are repaired via their fix functions.
    Raises if a failed assertion has no fix function or retries run out.

    Args:
        fn: Assertion-aware propagator produced by :func:`propagator_with_assertions`.
        key: JAX PRNG key.
        state: Current simulation state.
        max_tries: Maximum number of repair attempts.

    Returns:
        Propagated state with all assertions satisfied.

    Raises:
        ValueError: If called inside a JAX transform.
        RuntimeError: If assertions still fail after ``max_tries`` attempts.
    """
    is_traced = any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(state))
    if is_traced:
        raise ValueError("propagate_and_fix cannot be jax transformed.")

    for _ in range(max_tries):
        out = fn(key, state)
        state = out.value
        if not out.failed_assertions:
            return state
        state = out.fix_or_raise(state)
    raise RuntimeError("Failed to resolve potential after multiple attempts")


def compose_propagators[S](*propagators: Propagator[S]) -> Propagator[S]:
    """Compose multiple propagators into a single one by sequentially chaining them.

    Args:
        *propagators: Propagators to chain together

    Returns:
        SequentialPropagator that applies each propagator in order

    Example:
        ```python
        combined = compose_propagators(move_particles, update_velocities, check_energy)
        new_state = combined(key, state)
        ```
    """
    return SequentialPropagator(propagators)


@dataclass
class CachePropagator[State, ResultType, **P](Propagator[State]):
    """Propagator that computes a property and caches it in the state.

    Evaluates a state property (e.g., neighbor list, energy) and stores the
    result in the state using a lens-based update.

    Attributes:
        function: Function that computes the property
        update: Update function that stores the result in state
    """

    function: StateProperty[State, ResultType] = field(static=True)
    update: Update[State, ResultType] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        result = self.function(key, state)
        state = self.update(state, result)
        return state


@dataclass
class IdentityPropagator[State](Propagator[State]):
    """No-op propagator that returns the state unchanged.

    Useful as a placeholder or for testing.
    """

    def __call__(self, key: Array, state: State) -> State:
        del key  # Unused
        return state


type LogProbabilityRatio = Table[SystemId, Array]
"""Type alias for log probability ratio arrays."""


class ChangesFn[State, Changes](Protocol):
    """Protocol for functions that propose changes and a log proposal ratio."""

    def __call__(
        self, key: Array, state: State, /
    ) -> tuple[Changes, LogProbabilityRatio]: ...


class PatchFn[State, Changes, Move: Patch](Protocol):
    """Protocol for functions that convert move proposals to state patches."""

    def __call__(self, key: Array, state: State, proposal: Changes) -> Move: ...


def propose_mixed[State, Changes](
    key: Array,
    state: State,
    propose_fns: tuple[ChangesFn[State, Changes], ...],
    weights: tuple[float, ...] | None = None,
) -> tuple[Changes, LogProbabilityRatio, Array]:
    """Compute all proposals eagerly and select one at random.

    All ``propose_fns`` are evaluated, then ``jax.lax.select_n`` picks one.
    Returns (selected_changes, selected_log_ratio, which_index).
    """
    chain = key_chain(key)
    # We purposefully reuse this key for all proposals!
    # This is an optimization, since it allows for common subexpression elimination
    # across proposals.
    key = next(chain)
    all_results = tuple(fn(key, state) for fn in propose_fns)
    all_changes = tuple(r[0] for r in all_results)
    all_log_ratios = tuple(r[1] for r in all_results)
    if weights is None:
        weights = tuple(1.0 for _ in propose_fns)
        which = jax.random.randint(next(chain), (), 0, len(propose_fns))
    else:
        probs = jnp.array(weights) / sum(weights)
        which = jax.random.choice(next(chain), len(propose_fns), p=probs)
    selected = tree_map(lambda *cases: jax.lax.select_n(which, *cases), *all_changes)
    log_ratio = tree_map(
        lambda *cases: jax.lax.select_n(which, *cases), *all_log_ratios
    )
    return selected, log_ratio, which


class LogProbabilityRatioFn[State, Move: Patch](Protocol):
    """Protocol for computing target density ratios.

    Computes log probability ratio of target distribution (e.g., Boltzmann factor).
    """

    def __call__(
        self, state: State, patch: Move
    ) -> WithPatch[LogProbabilityRatio, Patch[State]]: ...


@dataclass
class MCMCPropagator[State, Changes, Move: Patch](Propagator[State]):
    """Metropolis-Hastings Monte Carlo propagator with acceptance/rejection.

    Supports both single-move and mixed-move scenarios. When multiple
    ``propose_fns`` are provided, one is selected at random each step
    (weighted by ``weights``), and only the corresponding scheduler is updated.

    Attributes:
        patch_fn: Converts changes to a state patch.
        propose_fns: Tuple of change proposal functions.
        log_probability_ratio_fn: Computes target density ratio (e.g., Boltzmann).
        parameter_schedulers: One scheduler per propose_fn, updated selectively.
        weights: Selection probabilities per move (unnormalized). None for uniform.
    """

    patch_fn: PatchFn[State, Changes, Move] = field(static=True)
    propose_fns: tuple[ChangesFn[State, Changes], ...] = field(static=True)
    log_probability_ratio_fn: LogProbabilityRatioFn[State, Move] = field(static=True)
    parameter_schedulers: tuple[Scheduler[State, Table[SystemId, Array]], ...] = field(
        static=True
    )
    weights: tuple[float, ...] | None = field(static=True, default=None)

    @jit
    def __call__(self, key: Array, state: State) -> State:
        chain = key_chain(key)

        # We disable JIT here because it allows us to preserve
        # identities of through otherwise jax.jit compiled code.
        # Without disable_jit_
        # x is jax.jit(lambda x: x))(x) # False
        # With disable_jit:
        # with jax.disable_jit():
        #     x is jax.jit(lambda x: x))(x) # True
        # This allows us to skip many select_n calls.
        with jax.disable_jit():
            # Select and propose
            changes, move_log_ratio, which = propose_mixed(
                next(chain), state, self.propose_fns, self.weights
            )
            patch = self.patch_fn(next(chain), state, changes)

            # Acceptance
            density = self.log_probability_ratio_fn(state, patch)
            log_p_ratio = move_log_ratio + density.data
            n_sys = len(log_p_ratio)
            accept = log_p_ratio > jnp.log(jax.random.uniform(next(chain), (n_sys,)))

            # Apply patches
            new_state = patch(state, accept)
            new_state = density.patch(new_state, accept)

            # Selectively update only the chosen scheduler
            candidates = tuple(
                sched(new_state, accept) for sched in self.parameter_schedulers
            )
        return tree_map(lambda *cs: select_n(which, *cs), *candidates)


@dataclass
class SwitchPropagator[State](Propagator[State]):
    """Randomly select and apply one propagator from multiple options.

    Chooses a propagator based on probabilities and applies it to the state.
    Useful for hybrid Monte Carlo schemes with multiple move types.

    Attributes:
        propagators: Tuple of propagators to choose from
        probabilities: Function returning selection probabilities for each propagator

    Warning:
        When vmapped, all propagators are executed and results selected, leading to
        higher compute costs. Use conditionals if vmap efficiency is critical.

    Example:
        ```python
        switch = SwitchPropagator(
            propagators=(translate_move, rotate_move, volume_move),
            probabilities=lambda s: jnp.array([0.7, 0.2, 0.1])
        )

        # Randomly select and apply one move type
        state = switch(key, state)
        ```
    """

    propagators: tuple[Propagator[State], ...] = field(static=True)
    probabilities: View[State, Array] = field(static=True)

    def __post_init__(self):
        if len(self.propagators) == 0:
            raise ValueError("At least one propagator must be provided.")

    def __call__(self, key: Array, state: State) -> State:
        logging.warning(
            "When vmapping SwitchPropagator, all paths will be executed leading to higher compute times. "
            "Ignore this message if you are not vmapping or are aware of the implications. "
        )
        chain = key_chain(key)
        probabilities = self.probabilities(state)
        assert probabilities.ndim == 1, "Probabilities must be a 1D array"
        assert probabilities.size == len(self.propagators), (
            "Number of probabilities must match number of propagators"
        )
        # Sample a propagator based on the probabilities
        idx = jax.random.choice(
            next(chain),
            jnp.arange(len(self.propagators)),
            p=probabilities / probabilities.sum(),
        )
        return jax.lax.switch(idx, self.propagators, next(chain), state)


@dataclass
class SequentialPropagator[State](Propagator[State]):
    """Apply multiple propagators in sequence.

    Chains propagators together, applying each in order with independent random keys.

    Attributes:
        propagators: Tuple of propagators to apply sequentially

    Example:
        ```python
        seq = SequentialPropagator((
            translate_particles,
            rotate_molecules,
            update_neighbor_list
        ))

        # Applies: state → translate → rotate → update_nl
        state = seq(key, state)
        ```
    """

    propagators: tuple[Propagator[State], ...] = field(static=True)

    def __post_init__(self):
        assert len(self.propagators) > 0, "At least one propagator must be provided."

    def __call__(self, key: Array, state: State) -> State:
        chain = key_chain(key)
        for propagator in self.propagators:
            state = propagator(next(chain), state)
        return state


@dataclass
class PalindromePropagator[State](Propagator[State]):
    """Apply propagators forward then backward to preserve detailed balance.

    Applies propagators in sequence: [P₁, P₂, ..., Pₙ, Pₙ, ..., P₂, P₁].
    This "telescope" pattern ensures that if individual propagators satisfy
    detailed balance, the combined propagator also does.

    Critical for maintaining correct equilibrium distributions in MCMC.

    Attributes:
        propagators: Tuple of propagators to apply palindromically

    Mathematical property:
        If each Pᵢ satisfies detailed balance, then the composition
        P₁ ∘ P₂ ∘ ... ∘ Pₙ ∘ Pₙ ∘ ... ∘ P₂ ∘ P₁ also satisfies detailed balance.

    Example:
        ```python
        palindrome = PalindromePropagator((
            translate_x,
            translate_y,
            translate_z
        ))

        # Applies: x → y → z → z → y → x
        # Maintains detailed balance
        state = palindrome(key, state)
        ```
    """

    propagators: tuple[Propagator[State], ...] = field(static=True)

    def __post_init__(self):
        assert len(self.propagators) > 0, "At least one propagator must be provided."

    def __call__(self, key: Array, state: State) -> State:
        chain = key_chain(key)
        for propagator in self.propagators + self.propagators[::-1]:
            state = propagator(next(chain), state)
        return state


@dataclass
class LoopPropagator[State](Propagator[State]):
    """Repeat a propagator multiple times in a loop.

    Applies a single propagator repeatedly for either a fixed number of iterations
    or a dynamic number determined from the state. Uses `jax.lax.while_loop` for
    efficient compilation.

    Attributes:
        propagator: The propagator to repeat
        repetitions: Either a fixed integer or a function extracting repetition count from state

    Example:
        ```python
        # Fixed repetitions
        loop = LoopPropagator(
            propagator=mc_move,
            repetitions=100
        )

        # Dynamic repetitions from state
        adaptive_loop = LoopPropagator(
            propagator=mc_move,
            repetitions=lambda s: s.num_equilibration_steps
        )

        state = loop(key, state)  # Applies mc_move 100 times
        ```
    """

    propagator: Propagator[State] = field(static=True)
    repetitions: View[State, Array] | int = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        chain = key_chain(key)
        if isinstance(self.repetitions, int):
            repetitions = jnp.array(self.repetitions)
        else:
            repetitions = self.repetitions(state)

        def body(carry: tuple[Array, Array, State]):
            i, key, prev_state = carry
            state = prev_state
            key, subkey = jax.random.split(key)
            state = self.propagator(subkey, state)
            return i + 1, key, state

        def cond(carry):
            i, _, _ = carry
            return i < repetitions

        init = (jnp.zeros((), dtype=int), next(chain), state)
        _, _, state = jax.lax.while_loop(cond, body, init)
        return state


@dataclass
class ResetOnErrorPropagator[State](Propagator[State]):
    """Rollback to previous state if runtime assertions fail.

    Wraps a propagator and checks runtime assertions after execution. If any
    assertion fails, reverts to the original state. Useful for robust simulation
    where certain configurations are invalid.

    Attributes:
        propagator: Base propagator to wrap with error handling

    Example:
        ```python
        safe_move = ResetOnErrorPropagator(risky_mc_move)

        # If risky_mc_move produces invalid state (assertion fails),
        # state is reset to original
        state = safe_move(key, state)
        ```

    Note:
        Uses [check_assertions][kups.core.assertion.check_assertions] which must be called within a
        [with_runtime_assertions][kups.core.assertion.with_runtime_assertions] context to function properly.
    """

    propagator: Propagator[State] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        new_state = self.propagator(key, state)
        mask = check_assertions(jax.tree.leaves(new_state)[0])
        result_state = tree_where_broadcast_last(mask, new_state, state)
        return result_state


@dataclass
class ScheduledPropertyPropagator[State, Input, Value](Propagator[State]):
    """Propagator that updates a property according to a schedule.

    Reads the scheduling input (e.g., step number) from the state, applies
    the schedule to compute a new value, and updates the state.

    This is useful for time-dependent parameter changes during simulation,
    such as temperature annealing, pressure ramps, or time step adaptation.

    Type Parameters:
        State: Simulation state type
        Input: Type of scheduling input (typically Array for step/time)
        Value: Type of value being scheduled

    Attributes:
        lens: Lens to access and update the scheduled property
        input_view: View to extract the scheduling input from state
        schedule: Schedule that computes new values

    Example:
        ```python
        from kups.core.schedule import LinearSchedule

        # Temperature annealing from 500K to 300K over 10000 steps
        temp_propagator = ScheduledPropertyPropagator(
            lens=lens(lambda s: s.temperature),
            input_view=lens(lambda s: s.step).get,
            schedule=LinearSchedule(
                start=jnp.array(500.0),
                end=jnp.array(300.0),
                total_steps=jnp.array(10000)
            )
        )

        # In simulation loop:
        state = temp_propagator(key, state)
        ```

    See Also:
        - [Schedule][kups.core.schedule.Schedule]: Protocol for scheduling functions
        - [PropertyScheduler][kups.core.schedule.PropertyScheduler]: Non-propagator scheduler
    """

    lens: Lens[State, Value] = field(static=True)
    input_view: View[State, Input] = field(static=True)
    schedule: Schedule[Input, Value] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        """Apply the schedule to update the state.

        Args:
            key: JAX PRNG key (unused, but required by Propagator protocol)
            state: Current simulation state

        Returns:
            Updated state with scheduled property modified
        """
        del key  # Unused, but required by Propagator protocol
        input_val = self.input_view(state)
        current = self.lens.get(state)
        new = self.schedule(input_val, current)
        return self.lens.set(state, new)


def step_counter_propagator[State](
    step_lens: Lens[State, Array],
) -> ScheduledPropertyPropagator[State, Array, Array]:
    """Build a propagator that increments a step counter by 1 each call.

    Wraps [ScheduledPropertyPropagator][kups.core.propagator.ScheduledPropertyPropagator]
    with an [IncrementSchedule][kups.core.schedule.IncrementSchedule] so that the
    counter stored at ``step_lens`` is advanced by 1 on every propagation step.

    Args:
        step_lens: Lens pointing to the integer step-counter array in the state.
            The array must be broadcastable with an increment of ``[1]``.

    Returns:
        [ScheduledPropertyPropagator][kups.core.propagator.ScheduledPropertyPropagator]
        that increments the counter by 1 each time it is called.
    """
    return ScheduledPropertyPropagator(
        lens=step_lens,
        input_view=step_lens,
        schedule=IncrementSchedule(increment=jnp.array([1])),
    )


@dataclass
class BakeConstantsPropagator[State](Propagator[State]):
    """Wraps a propagator by identifying and caching state leaves that are unchanged.

    Uses ``eval_shape`` to trace the inner propagator and detect which leaves are
    returned via identity (i.e. not modified). Those leaves are snapshot as
    read-only NumPy arrays and injected on every call, avoiding redundant
    device transfers. This may also enable XLA constant folding, as the baked
    values become compile-time constants visible to the compiler.

    Note:
        Baked values are frozen at construction time. Any external mutation of
        those leaves after ``new()`` will **not** be reflected in subsequent
        calls — the cached snapshot is used instead.

    Attributes:
        propagator: The inner propagator to wrap.
        const_indices: Flat indices of constant leaves in the pytree.
        consts: Cached NumPy snapshots of the constant leaves.
    """

    propagator: Propagator[State] = field(static=True)
    const_indices: tuple[int, ...] = field(static=True)
    consts: tuple[np.ndarray, ...] = field(static=True)

    @classmethod
    def new(cls, propagator: Propagator[State], state: State):
        leaf_mask: tuple[bool, ...] = ()

        def f(key, state):
            nonlocal leaf_mask
            with jax.disable_jit():
                out = propagator(key, state)
            in_struc = tree_structure(state)
            out_struc = tree_structure(out)
            assert in_struc == out_struc, (
                "BakePropagator requires the same tree structure"
            )
            in_leaves = in_struc.flatten_up_to(state)
            out_leaves = out_struc.flatten_up_to(out)
            identical_leafes = tuple(
                in_leaf is out_leaf and isinstance(in_leaf, Array)
                for in_leaf, out_leaf in zip(in_leaves, out_leaves)
            )
            leaf_mask = identical_leafes
            return None

        # We are just interested in the side effect of populating leaf_mask, so we ignore the output
        jax.eval_shape(f, jax.random.key(0), state)
        consts_indices = tuple(np.where(leaf_mask)[0])
        leafes = jax.tree.leaves(state)
        consts = tuple(np.asarray(leafes[idx]) for idx in consts_indices)
        logging.info(
            f"Identified {len(consts)} of {len(leafes)} total leaves as constants to bake into the propagator."
        )
        for c in consts:
            c.setflags(write=False)
        return cls(propagator=propagator, const_indices=consts_indices, consts=consts)

    def __call__(self, key: Array, state: State) -> State:
        struc = tree_structure(state)
        leaves = struc.flatten_up_to(state)
        in_leaves = leaves
        leaves = list(leaves)
        # Use constants from the original state for the propagator
        for idx, const in zip(self.const_indices, self.consts):
            leaves[idx] = jnp.asarray(const)
        out = self.propagator(key, struc.unflatten(leaves))
        out_leaves = struc.flatten_up_to(out)
        # For the output we want to preserve the original leaf identities to avoid copies
        for idx in self.const_indices:
            out_leaves[idx] = in_leaves[idx]
        return struc.unflatten(out_leaves)
