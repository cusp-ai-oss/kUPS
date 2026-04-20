# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for propagator functionality."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import SimpleLens, bind, view
from kups.core.parameter_scheduler import (
    AcceptanceHistory,
    Correlation,
    ParameterSchedulerState,
    acceptance_target_schedule,
)
from kups.core.patch import Accept, IdPatch, WithPatch
from kups.core.propagator import (
    BakeConstantsPropagator,
    CachePropagator,
    LoopPropagator,
    MCMCPropagator,
    PalindromePropagator,
    ScheduledPropertyPropagator,
    SequentialPropagator,
    StatePropertySum,
    SwitchPropagator,
    compose_propagators,
    propagate_and_fix,
    propagator_with_assertions,
)
from kups.core.result import Result, as_result_function
from kups.core.schedule import (
    ExponentialSchedule,
    LinearSchedule,
    PropertyScheduler,
)
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, field


# Test fixtures and helpers
@dataclass
class ExampleState:
    """Simple test state for propagator testing."""

    value: float = field(default=1.0)
    array_data: Array = field(default_factory=lambda: jnp.array([1.0, 2.0, 3.0]))
    cached_result: float = field(default=0.0)
    switch_probabilities: Array = field(
        default_factory=lambda: jnp.array([0.5, 0.3, 0.2])
    )
    scheduler_params: Table[SystemId, ParameterSchedulerState] = field(
        default_factory=lambda: Table.arange(
            ParameterSchedulerState(
                value=jnp.array([1.0, 2.0]),
                multiplicity=jnp.array([1.1, 1.1]),
                target=jnp.array([0.5, 0.5]),
                tolerance=jnp.array([0.1, 0.1]),
                correlation=Correlation.NEGATIVE,
                bounds=(None, None),
                history=AcceptanceHistory(
                    values=jnp.zeros((2, 5)),  # 2 systems, 5 history length
                    index=jnp.array([0, 0]),
                ),
            ),
            label=SystemId,
        )
    )
    step: Array = field(default_factory=lambda: jnp.array(0))
    num_systems: int = field(default=2, static=True)


@dataclass
class ArrayPatch:
    """Patch that modifies array data in the state."""

    increment: jax.Array = field(default_factory=lambda: jnp.ones(()))

    def __call__(self, state: ExampleState, accept: Accept) -> ExampleState:
        # Apply increment only where accept is True
        sys_idx = Index.new(list(accept.keys))
        new_value = jnp.where(
            accept[sys_idx][0], state.array_data + self.increment, state.array_data
        )
        new_state = bind(state).focus(lambda s: s.array_data).set(new_value)
        return new_state


@dataclass
class ExamplePatch:
    """Test patch that adds a value to the state."""

    increment: jax.Array = field(default_factory=lambda: jnp.ones(()))

    def __call__(self, state: ExampleState, accept: Accept) -> ExampleState:
        # Apply increment only where accept is True
        sys_idx = Index.new(list(accept.keys))
        new_value = jnp.where(
            accept[sys_idx][0], state.value + self.increment, state.value
        )
        new_state = bind(state).focus(lambda s: s.value).set(new_value)
        return new_state


@dataclass
class AddableProperty:
    """Test property that implements the Addable protocol."""

    magnitude: float = field(default=1.0)

    def __add__(self, other):
        return AddableProperty(magnitude=self.magnitude + other.magnitude)


@dataclass
class ExtendedTestPatch:
    """Extended test patch for propagator testing."""

    increment: float = field(default=1.0)

    def __call__(self, state: ExampleState, accept: Array) -> ExampleState:
        # Apply increment where accept is True
        new_value = jnp.where(accept, state.value + self.increment, state.value)
        new_state = bind(state).focus(lambda s: s.value).set(new_value)
        return new_state


# Helper functions for lens creation
def get_schedule_params(
    state: ExampleState,
) -> Table[SystemId, ParameterSchedulerState]:
    """Get parameter scheduler parameters from test state."""
    return state.scheduler_params


def get_num_systems(state: ExampleState) -> int:
    """Get number of systems from test state."""
    return state.num_systems


# Helper functions for extended test state access
def get_array_data(state: ExampleState) -> Array:
    """Get array data from test state."""
    return state.array_data


def get_cached_result(state: ExampleState) -> float:
    """Get cached result from test state."""
    return state.cached_result


def get_switch_probabilities(state: ExampleState) -> Array:
    """Get switch probabilities from test state."""
    return state.switch_probabilities


def get_step(state: ExampleState) -> Array:
    """Get step number from test state."""
    return state.step


def get_value(state: ExampleState) -> float:
    """Get value from test state."""
    return state.value


def get_extended_scheduler_params(state: ExampleState) -> ParameterSchedulerState:
    return state.scheduler_params


@pytest.fixture
def simple_state():
    """Create a simple test state."""
    return ExampleState()


@pytest.fixture
def rng_key():
    """Create a JAX random key."""
    return jax.random.key(42)


@pytest.fixture
def test_state():
    """Create an extended test state."""
    return ExampleState()


class TestComposePropagators:
    """Tests for compose_propagators function."""

    def test_compose_propagators(self, simple_state, rng_key):
        """Test composing single and multiple propagators."""

        def increment_propagator(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 1.0)

        def multiply_propagator(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value * 2.0)

        # Single propagator
        composed = compose_propagators(increment_propagator)
        result = composed(rng_key, simple_state)
        assert result.value == 2.0

        # Multiple propagators: (1.0 + 1.0) * 2.0 = 4.0
        composed = compose_propagators(increment_propagator, multiply_propagator)
        result = composed(rng_key, simple_state)
        assert result.value == 4.0

        # Also serves as simple propagator protocol test
        result = increment_propagator(rng_key, simple_state)
        assert result.value == 2.0

    def test_empty_propagators_assertion(self, simple_state, rng_key):
        """Test that composing no propagators raises an assertion error."""

        with pytest.raises(
            AssertionError, match="At least one propagator must be provided"
        ):
            compose_propagators()


class TestMCMCPropagator:
    """Tests for MCMCPropagator class."""

    @pytest.fixture
    def schedule_param_lens(self):
        return SimpleLens(get_schedule_params)

    @pytest.fixture
    def parameter_scheduler(self, schedule_param_lens):
        return PropertyScheduler(
            lens=schedule_param_lens,
            schedule=Table.transform(acceptance_target_schedule),
        )

    @staticmethod
    def _changes_fn(key: Array, state: ExampleState, /):
        """ChangesFn: returns (increment, log_ratio)."""
        increment = jax.random.normal(key) * 0.1
        log_ratio = Table.arange(jnp.array([1.0, 1.0]), label=SystemId)
        return increment, log_ratio

    @staticmethod
    def _patch_fn(key: Array, state: ExampleState, proposal: float):
        """PatchFn: converts increment to ExamplePatch."""
        return ExamplePatch(increment=proposal)

    @staticmethod
    def _probability_ratio_fn(state: ExampleState, patch: ExamplePatch):
        log_ratio = -jnp.abs(patch.increment) * 10.0
        ratio = jnp.exp(log_ratio)
        return WithPatch(Table.arange(jnp.array([ratio, ratio]), label=SystemId), patch)

    def test_mcmc_propagator_basic(
        self,
        simple_state,
        rng_key,
        schedule_param_lens,
        parameter_scheduler,
    ):
        """Test MCMC propagator init, call, multiple schedulers."""
        propagator = MCMCPropagator(
            patch_fn=self._patch_fn,
            propose_fns=(self._changes_fn,),
            log_probability_ratio_fn=self._probability_ratio_fn,
            parameter_schedulers=(parameter_scheduler,),
        )
        assert len(propagator.parameter_schedulers) == 1

        result = propagator(rng_key, simple_state)
        assert isinstance(result, ExampleState)
        assert hasattr(result, "value")
        assert hasattr(result, "scheduler_params")

        # Multiple schedulers (same scheduler twice)
        scheduler2 = PropertyScheduler(
            lens=schedule_param_lens,
            schedule=Table.transform(acceptance_target_schedule),
        )
        propagator_multi = MCMCPropagator(
            patch_fn=self._patch_fn,
            propose_fns=(self._changes_fn, self._changes_fn),
            log_probability_ratio_fn=self._probability_ratio_fn,
            parameter_schedulers=(parameter_scheduler, scheduler2),
        )
        result = propagator_multi(rng_key, simple_state)
        assert isinstance(result, ExampleState)

    def test_mcmc_propagator_acceptance_and_determinism(
        self, simple_state, parameter_scheduler
    ):
        """Test acceptance logic and deterministic behavior with same seed."""

        def fixed_changes(key: Array, state: ExampleState, /):
            log_ratio = Table.arange(jnp.array([1.0, 1.0]), label=SystemId)
            return jnp.array(0.5), log_ratio

        def always_accept(state: ExampleState, patch: ExamplePatch):
            return WithPatch(
                Table.arange(jnp.log(jnp.array([1.0, 1.0])), label=SystemId),
                IdPatch(),
            )

        propagator = MCMCPropagator(
            patch_fn=self._patch_fn,
            propose_fns=(fixed_changes,),
            log_probability_ratio_fn=always_accept,
            parameter_schedulers=(parameter_scheduler,),
        )

        key = jax.random.key(0)
        result = propagator(key, simple_state)
        assert isinstance(result, ExampleState)
        assert result.value == simple_state.value + 0.5

        # Determinism
        def det_changes(key: Array, state: ExampleState, /):
            return jnp.array(1.0), Table.arange(jnp.array([1.0, 1.0]), label=SystemId)

        def det_probability(state: ExampleState, patch: ExamplePatch):
            return WithPatch(Table.arange(jnp.array([0.7, 0.7]), label=SystemId), patch)

        propagator2 = MCMCPropagator(
            patch_fn=self._patch_fn,
            propose_fns=(det_changes,),
            log_probability_ratio_fn=det_probability,
            parameter_schedulers=(parameter_scheduler,),
        )

        key = jax.random.key(12345)
        result1 = propagator2(key, simple_state)
        key = jax.random.key(12345)
        result2 = propagator2(key, simple_state)
        npt.assert_array_equal(result1.value, result2.value)


class TestPropagatorIntegration:
    """Integration tests combining different propagator features."""

    def test_composed_mcmc_propagators(self, simple_state, rng_key):
        """Test composing multiple MCMC propagators."""

        schedule_param_lens = SimpleLens(get_schedule_params)

        def changes_fn(key: Array, state: ExampleState, /):
            increment = jax.random.normal(key) * 0.1
            return increment, Table.arange(jnp.array([1.0, 1.0]), label=SystemId)

        def patch_fn(key: Array, state: ExampleState, proposal):
            return ExamplePatch(increment=proposal)

        def probability_ratio_fn(state: ExampleState, patch: ExamplePatch):
            log_ratio = -jnp.abs(patch.increment) * 10.0
            ratio = jnp.exp(log_ratio)
            return WithPatch(
                Table.arange(jnp.array([ratio, ratio]), label=SystemId), patch
            )

        parameter_scheduler = PropertyScheduler(
            lens=schedule_param_lens,
            schedule=Table.transform(acceptance_target_schedule),
        )

        mcmc1 = MCMCPropagator(
            patch_fn=patch_fn,
            propose_fns=(changes_fn,),
            log_probability_ratio_fn=probability_ratio_fn,
            parameter_schedulers=(parameter_scheduler,),
        )

        mcmc2 = MCMCPropagator(
            patch_fn=patch_fn,
            propose_fns=(changes_fn,),
            log_probability_ratio_fn=probability_ratio_fn,
            parameter_schedulers=(parameter_scheduler,),
        )

        composed = compose_propagators(mcmc1, mcmc2)
        result = composed(rng_key, simple_state)

        assert isinstance(result, ExampleState)


class TestStatePropertySum:
    """Tests for StatePropertySum class."""

    def test_single_property(self, test_state, rng_key):
        """Test StatePropertySum with a single property."""

        def property_fn(key: Array, state: ExampleState) -> AddableProperty:
            return AddableProperty(magnitude=state.value * 2.0)

        prop_sum = StatePropertySum(properties=(property_fn,))
        result = prop_sum(rng_key, test_state)

        assert isinstance(result, AddableProperty)
        assert result.magnitude == 2.0  # 1.0 * 2.0

    def test_multiple_properties(self, test_state, rng_key):
        """Test StatePropertySum with multiple properties."""

        def property_fn1(key: Array, state: ExampleState) -> AddableProperty:
            return AddableProperty(magnitude=state.value)

        def property_fn2(key: Array, state: ExampleState) -> AddableProperty:
            return AddableProperty(magnitude=state.value * 3.0)

        def property_fn3(key: Array, state: ExampleState) -> AddableProperty:
            return AddableProperty(magnitude=5.0)

        prop_sum = StatePropertySum(
            properties=(property_fn1, property_fn2, property_fn3)
        )
        result = prop_sum(rng_key, test_state)

        assert isinstance(result, AddableProperty)
        # Should sum: 1.0 + 3.0 + 5.0 = 9.0
        assert result.magnitude == 9.0

    def test_empty_properties_tuple(self, test_state, rng_key):
        """Test StatePropertySum with empty properties tuple raises IndexError."""
        prop_sum = StatePropertySum(properties=())

        with pytest.raises(IndexError):
            prop_sum(rng_key, test_state)


class TestCachePropagator:
    """Tests for CachePropagator class."""

    def test_cache_propagator_basic(self, test_state, rng_key):
        """Test basic CachePropagator functionality."""

        def compute_function(key: Array, state: ExampleState) -> float:
            return state.value * 10.0

        def update_function(state: ExampleState, value: float) -> ExampleState:
            return bind(state).focus(lambda s: s.cached_result).set(value)

        cache_propagator = CachePropagator(
            function=compute_function, update=update_function
        )

        result = cache_propagator(rng_key, test_state)

        assert result.cached_result == 10.0  # 1.0 * 10.0
        assert result.value == 1.0

    def test_cache_propagator_complex_state(self, rng_key):
        """Test CachePropagator with complex state updates."""

        state = ExampleState(value=5.0, array_data=jnp.array([10.0, 20.0, 30.0]))

        def compute_array_sum(key: Array, state: ExampleState) -> float:
            return float(jnp.sum(state.array_data))

        def update_both_fields(state: ExampleState, value: float) -> ExampleState:
            new_state = bind(state).focus(lambda s: s.cached_result).set(value)
            new_state = bind(new_state).focus(lambda s: s.value).set(value / 10.0)
            return new_state

        cache_propagator = CachePropagator(
            function=compute_array_sum, update=update_both_fields
        )

        result = cache_propagator(rng_key, state)

        assert result.cached_result == 60.0  # 10 + 20 + 30
        assert result.value == 6.0  # 60 / 10


class TestSwitchPropagator:
    """Tests for SwitchPropagator class."""

    def test_switch_propagator_basic(self, test_state, rng_key):
        """Test basic SwitchPropagator functionality."""

        def prop1(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 1.0)

        def prop2(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value * 2.0)

        def prop3(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value - 0.5)

        probabilities_view = view(get_switch_probabilities)

        switch_propagator = SwitchPropagator(
            propagators=(prop1, prop2, prop3), probabilities=probabilities_view
        )

        result = switch_propagator(rng_key, test_state)

        assert isinstance(result, ExampleState)
        assert result.value != test_state.value

    def test_switch_propagator_deterministic_selection(self, rng_key):
        """Test SwitchPropagator with deterministic probabilities."""

        state = ExampleState(switch_probabilities=jnp.array([1.0, 0.0]))

        def prop1(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(100.0)

        def prop2(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(200.0)

        probabilities_view = view(get_switch_probabilities)

        switch_propagator = SwitchPropagator(
            propagators=(prop1, prop2), probabilities=probabilities_view
        )

        result = switch_propagator(rng_key, state)

        assert result.value == 100.0

    def test_switch_propagator_empty_propagators(self):
        """Test SwitchPropagator raises error with no propagators."""

        probabilities_view = view(get_switch_probabilities)

        with pytest.raises(
            ValueError, match="At least one propagator must be provided"
        ):
            SwitchPropagator(propagators=(), probabilities=probabilities_view)

    def test_switch_propagator_probability_mismatch(self, test_state, rng_key):
        """Test SwitchPropagator with mismatched probabilities and propagators."""

        def prop1(key: Array, state: ExampleState) -> ExampleState:
            return state

        state = ExampleState(switch_probabilities=jnp.array([0.5, 0.5]))
        probabilities_view = view(get_switch_probabilities)

        switch_propagator = SwitchPropagator(
            propagators=(prop1,),
            probabilities=probabilities_view,
        )

        with pytest.raises(AssertionError, match="Number of probabilities must match"):
            switch_propagator(rng_key, state)

    def test_switch_propagator_multidimensional_probabilities(self, rng_key):
        """Test SwitchPropagator with multidimensional probabilities."""

        state = ExampleState(switch_probabilities=jnp.array([[0.5, 0.5], [0.3, 0.7]]))

        def prop1(key: Array, state: ExampleState) -> ExampleState:
            return state

        def prop2(key: Array, state: ExampleState) -> ExampleState:
            return state

        probabilities_view = view(get_switch_probabilities)

        switch_propagator = SwitchPropagator(
            propagators=(prop1, prop2), probabilities=probabilities_view
        )

        with pytest.raises(AssertionError, match="Probabilities must be a 1D array"):
            switch_propagator(rng_key, state)


class TestSequentialPropagator:
    """Tests for SequentialPropagator class."""

    def test_sequential_propagator(self, test_state, rng_key):
        """Test SequentialPropagator with single and multiple propagators."""

        def prop1(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 10.0)

        def prop2(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value * 2.0)

        def prop3(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value - 5.0)

        # Multiple propagators: ((1 + 10) * 2) - 5 = 17
        sequential_propagator = SequentialPropagator(propagators=(prop1, prop2, prop3))
        result = sequential_propagator(rng_key, test_state)
        assert result.value == 17.0

        # Single propagator: 1 + 10 = 11
        single_propagator = SequentialPropagator(propagators=(prop1,))
        result = single_propagator(rng_key, test_state)
        assert result.value == 11.0

    def test_sequential_propagator_empty(self):
        """Test SequentialPropagator with no propagators."""

        with pytest.raises(AssertionError):
            SequentialPropagator(propagators=())


class TestPalindromePropagator:
    """Tests for PalindromePropagator class."""

    def test_palindrome_propagator_arithmetic(self, test_state, rng_key):
        """Test PalindromePropagator arithmetic with 2 and 3 operations, detailed balance, and vs sequential."""

        def add_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 10.0)

        def multiply_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value * 2.0)

        # Basic 2-op: Forward: 1->11->22, Reverse: 22->44->54
        palindrome = PalindromePropagator(propagators=(add_prop, multiply_prop))
        result = palindrome(rng_key, test_state)
        assert result.value == 54.0

        # Detailed balance property
        def reversible_add(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 1.0)

        def reversible_scale(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value * 1.1)

        palindrome_db = PalindromePropagator(
            propagators=(reversible_add, reversible_scale)
        )
        result = palindrome_db(rng_key, test_state)
        # Forward: 1->2->2.2, Reverse: 2.2->2.42->3.42
        npt.assert_allclose(result.value, 3.42, rtol=1e-10)

        # Three operations: Forward: 1->3->9->8, Reverse: 8->7->21->23
        def op1(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 2.0)

        def op2(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value * 3.0)

        def op3(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value - 1.0)

        palindrome_3 = PalindromePropagator(propagators=(op1, op2, op3))
        result = palindrome_3(rng_key, test_state)
        assert result.value == 23.0

        # Vs sequential comparison
        def increment_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 1.0)

        def double_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value * 2.0)

        sequential = SequentialPropagator(propagators=(increment_prop, double_prop))
        palindrome_cmp = PalindromePropagator(propagators=(increment_prop, double_prop))

        seq_result = sequential(rng_key, test_state)
        pal_result = palindrome_cmp(rng_key, test_state)

        assert seq_result.value == 4.0  # (1+1)*2
        assert pal_result.value == 9.0  # ((1+1)*2)*2+1
        assert seq_result.value != pal_result.value

    def test_palindrome_propagator_single(self, test_state, rng_key):
        """Test PalindromePropagator with single propagator."""

        def increment_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 5.0)

        palindrome_propagator = PalindromePropagator(propagators=(increment_prop,))
        result = palindrome_propagator(rng_key, test_state)
        assert result.value == 11.0  # (1+5)+5

    def test_palindrome_propagator_empty(self):
        """Test PalindromePropagator with no propagators."""

        with pytest.raises(AssertionError):
            PalindromePropagator(propagators=())

    def test_palindrome_propagator_result_propagation(self, test_state, rng_key):
        """Test that PalindromePropagator properly propagates Results."""

        def prop_with_assertion(key: Array, state: ExampleState) -> ExampleState:
            new_state = bind(state).focus(lambda s: s.value).set(state.value + 1.0)
            runtime_assert(
                predicate=jnp.array(True),
                message="Test assertion",
                fmt_args={"value": new_state.value},
                fix_args=new_state.value,
            )
            return new_state

        palindrome_propagator = PalindromePropagator(propagators=(prop_with_assertion,))

        result = as_result_function(palindrome_propagator)(rng_key, test_state)

        assert len(result.assertions) > 0
        assert result.value.value == 3.0  # 1 + 1 + 1


class TestLoopPropagator:
    """Tests for LoopPropagator class."""

    def test_loop_propagator_repetitions(self, test_state, rng_key):
        """Test LoopPropagator with various repetition counts."""

        # Basic: 5 repetitions of +1
        def increment_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 1.0)

        loop = LoopPropagator(propagator=increment_prop, repetitions=5)
        result = loop(rng_key, test_state)
        assert result.value == 6.0  # 1 + 5

        # Single iteration: *3
        def multiply_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value * 3.0)

        loop_single = LoopPropagator(propagator=multiply_prop, repetitions=1)
        result = loop_single(rng_key, test_state)
        assert result.value == 3.0

        # Large iterations: 50 * +0.1
        def small_increment_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 0.1)

        loop_large = LoopPropagator(propagator=small_increment_prop, repetitions=50)
        result = loop_large(rng_key, test_state)
        npt.assert_allclose(result.value, 6.0, rtol=1e-10)

    def test_loop_propagator_with_assertions(self, test_state, rng_key):
        """Test LoopPropagator handles assertions correctly."""

        def prop_with_assertions(key: Array, state: ExampleState) -> ExampleState:
            new_state = bind(state).focus(lambda s: s.value).set(state.value + 1.0)
            runtime_assert(
                new_state.value > 0,
                message="Value should be positive",
                fmt_args={"value": new_state.value},
            )
            return new_state

        loop_propagator = LoopPropagator(propagator=prop_with_assertions, repetitions=3)

        result = as_result_function(loop_propagator)(rng_key, test_state)

        assert result.value.value == 4.0  # 1 + 3
        assert len(result.assertions) > 0


class TestExtendedPropagatorIntegration:
    """Extended integration tests combining different propagator types."""

    def test_nested_propagator_composition(self, test_state, rng_key):
        """Test complex nesting of different propagator types."""

        def base_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 1.0)

        loop_prop = LoopPropagator(propagator=base_prop, repetitions=3)

        def multiply_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value * 2.0)

        sequential_prop = SequentialPropagator(propagators=(loop_prop, multiply_prop))

        def final_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value - 1.0)

        final_composed = compose_propagators(sequential_prop, final_prop)

        result = final_composed(rng_key, test_state)

        # ((1 + 3) * 2) - 1 = 7
        assert result.value == 7.0

    def test_cache_and_switch_integration(self, rng_key):
        """Test integration of CachePropagator with SwitchPropagator."""

        def simple_prop1(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.cached_result).set(25.0)

        def simple_prop2(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value * 2.0)

        state = ExampleState(
            value=5.0,
            switch_probabilities=jnp.array([1.0, 0.0]),
        )

        probabilities_view = view(get_switch_probabilities)
        switch_prop = SwitchPropagator(
            propagators=(simple_prop1, simple_prop2), probabilities=probabilities_view
        )

        result = switch_prop(rng_key, state)

        assert result.cached_result == 25.0
        assert result.value == 5.0

    def test_mcmc_with_sequential_and_loop(self, rng_key):
        """Test MCMC propagator integrated with sequential and loop propagators."""

        state = ExampleState(
            value=1.0,
            array_data=jnp.array([1.0, 2.0]),
        )

        def changes_fn(key: Array, state: ExampleState, /):
            return jnp.array(0.5), Table.arange(jnp.array([1.0, 1.0]), label=SystemId)

        def patch_fn(key: Array, state: ExampleState, proposal):
            return ArrayPatch(increment=proposal)

        def probability_fn(state: ExampleState, patch: ArrayPatch):
            return WithPatch(
                Table.arange(jnp.array([0.9, 0.9]), label=SystemId), IdPatch()
            )

        schedule_param_lens = SimpleLens(get_extended_scheduler_params)

        scheduler = PropertyScheduler(
            lens=schedule_param_lens,
            schedule=Table.transform(acceptance_target_schedule),
        )

        mcmc_prop = MCMCPropagator(
            patch_fn=patch_fn,
            propose_fns=(changes_fn,),
            log_probability_ratio_fn=probability_fn,
            parameter_schedulers=(scheduler,),
        )

        loop_mcmc = LoopPropagator(propagator=mcmc_prop, repetitions=2)

        def cleanup_prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(jnp.round(state.value, 2))

        final_prop = SequentialPropagator(propagators=(loop_mcmc, cleanup_prop))

        result = final_prop(rng_key, state)

        assert isinstance(result, ExampleState)
        assert isinstance(result.value, (float, jnp.ndarray))


class TestScheduledPropertyPropagator:
    """Tests for ScheduledPropertyPropagator class."""

    def test_scheduled_propagator_basic(self, rng_key):
        """Test ScheduledPropertyPropagator with different schedules."""
        state = ExampleState(value=100.0, step=jnp.array(50))

        # Test linear schedule
        propagator = ScheduledPropertyPropagator(
            lens=SimpleLens(get_value),
            input_view=view(get_step),
            schedule=LinearSchedule(
                start=jnp.array(0.0),
                end=jnp.array(100.0),
                total_steps=jnp.array(100),
            ),
        )
        result = propagator(rng_key, state)
        npt.assert_array_almost_equal(result.value, 50.0)

        # Test exponential schedule
        exp_propagator = ScheduledPropertyPropagator(
            lens=SimpleLens(get_value),
            input_view=view(get_step),
            schedule=ExponentialSchedule(rate=jnp.array(0.9), bounds=(None, None)),
        )
        result = exp_propagator(rng_key, state)
        npt.assert_array_almost_equal(result.value, 90.0)

    def test_scheduled_propagator_composition_and_jit(self, rng_key):
        """Test ScheduledPropertyPropagator with composition and JIT."""
        state = ExampleState(value=100.0, step=jnp.array(0))

        decay_propagator = ScheduledPropertyPropagator(
            lens=SimpleLens(get_value),
            input_view=view(get_step),
            schedule=ExponentialSchedule(rate=jnp.array(0.9), bounds=(None, None)),
        )

        loop_propagator = LoopPropagator(propagator=decay_propagator, repetitions=3)

        @jax.jit
        def run(key, state):
            return loop_propagator(key, state)

        result = run(rng_key, state)
        npt.assert_array_almost_equal(result.value, 100.0 * 0.9**3)


class TestPropagatorWithAssertions:
    """Tests for propagator_with_assertions."""

    def test_propagator_with_assertions(self, simple_state, rng_key):
        """Test returns result, captures passing and failing assertions."""

        # Returns result
        def prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 1.0)

        fn = propagator_with_assertions(prop)
        result = fn(rng_key, simple_state)
        assert isinstance(result, Result)
        assert result.value.value == 2.0

        # Captures passing assertion
        def prop_pass(key: Array, state: ExampleState) -> ExampleState:
            new_state = bind(state).focus(lambda s: s.value).set(state.value + 1.0)
            runtime_assert(jnp.array(True), message="always passes")
            return new_state

        fn_pass = propagator_with_assertions(prop_pass)
        result = fn_pass(rng_key, simple_state)
        assert len(result.assertions) == 1
        assert not result.failed_assertions

        # Captures failing assertion
        def prop_fail(key: Array, state: ExampleState) -> ExampleState:
            runtime_assert(jnp.array(False), message="always fails")
            return state

        fn_fail = propagator_with_assertions(prop_fail)
        result = fn_fail(rng_key, simple_state)
        assert result.failed_assertions


class TestPropagateAndFix:
    """Tests for propagate_and_fix."""

    def test_propagate_and_fix(self, simple_state, rng_key):
        """Test succeeds without assertions and applies fix and retries."""

        # Succeeds without assertions
        def prop(key: Array, state: ExampleState) -> ExampleState:
            return bind(state).focus(lambda s: s.value).set(state.value + 1.0)

        fn = propagator_with_assertions(prop)
        result = propagate_and_fix(fn, rng_key, simple_state)
        assert result.value == 2.0

        # Applies fix and retries
        state = ExampleState(value=1.0)

        def prop_fix(key: Array, state: ExampleState) -> ExampleState:
            runtime_assert(
                jnp.array(state.value >= 5.0),
                message="value too small",
                fix_fn=lambda s, v: bind(s).focus(lambda s: s.value).set(v),
                fix_args=jnp.array(5.0),
            )
            return state

        fn_fix = propagator_with_assertions(prop_fix)
        result = propagate_and_fix(fn_fix, rng_key, state)
        assert result.value == 5.0

    def test_raises_after_max_tries(self, rng_key):
        """Propagator always fails; fix does not resolve the condition."""
        state = ExampleState(value=1.0)

        def prop(key: Array, state: ExampleState) -> ExampleState:
            runtime_assert(
                jnp.array(state.value >= 10.0),
                message="never resolves",
                fix_fn=lambda s, _: s,
                fix_args=jnp.array(0.0),
            )
            return state

        fn = propagator_with_assertions(prop)
        with pytest.raises(RuntimeError, match="Failed to resolve"):
            propagate_and_fix(fn, rng_key, state, max_tries=3)

    def test_raises_inside_jax_transform(self, simple_state, rng_key):
        def prop(key: Array, state: ExampleState) -> ExampleState:
            return state

        fn = propagator_with_assertions(prop)
        with pytest.raises(ValueError, match="cannot be jax transformed"):
            jax.jit(lambda k, s: propagate_and_fix(fn, k, s))(rng_key, simple_state)


@dataclass
class _SimpleState:
    position: jax.Array
    velocity: jax.Array
    mass: jax.Array  # constant — not modified by propagator


def _make_propagator():
    """Propagator that only updates position and velocity, leaving mass unchanged."""

    def propagate(_key, state: _SimpleState) -> _SimpleState:
        return _SimpleState(
            position=state.position + state.velocity,
            velocity=state.velocity * 0.99,
            mass=state.mass,
        )

    return propagate


def _make_state():
    return _SimpleState(
        position=jnp.array([1.0, 2.0, 3.0]),
        velocity=jnp.array([0.1, 0.2, 0.3]),
        mass=jnp.array([10.0, 20.0, 30.0]),
    )


class TestBakeConstantsPropagator:
    def test_identifies_constants(self):
        state = _make_state()
        baked = BakeConstantsPropagator.new(_make_propagator(), state)
        assert len(baked.consts) == 1  # only mass

    def test_output_matches_original(self):
        state = _make_state()
        key = jax.random.key(42)
        expected = _make_propagator()(key, state)
        baked = BakeConstantsPropagator.new(_make_propagator(), state)
        result = baked(key, state)
        npt.assert_allclose(result.position, expected.position)
        npt.assert_allclose(result.velocity, expected.velocity)
        npt.assert_allclose(result.mass, expected.mass)

    def test_consts_are_read_only(self):
        state = _make_state()
        baked = BakeConstantsPropagator.new(_make_propagator(), state)
        for c in baked.consts:
            assert not c.flags.writeable

    def test_baked_values_not_affected_by_external_mutation(self):
        """Baked constants are snapshots; changing the input state has no effect."""
        state = _make_state()
        baked = BakeConstantsPropagator.new(_make_propagator(), state)
        # Mutate state with a different mass
        mutated = _SimpleState(
            position=state.position,
            velocity=state.velocity,
            mass=jnp.array([999.0, 999.0, 999.0]),
        )
        result = baked(jax.random.key(0), mutated)
        # Mass in output should reflect the *input* leaf identity, not the baked value
        npt.assert_allclose(result.mass, mutated.mass)

    def test_all_leaves_modified(self):
        """When all leaves change, no constants are baked."""

        def propagate(_key, state: _SimpleState) -> _SimpleState:
            return _SimpleState(
                position=state.position + 1,
                velocity=state.velocity + 1,
                mass=state.mass + 1,
            )

        state = _make_state()
        baked = BakeConstantsPropagator.new(propagate, state)
        assert len(baked.consts) == 0

    def test_all_leaves_constant(self):
        """When no leaves change, all are baked."""

        def propagate(_key, state: _SimpleState) -> _SimpleState:
            return state

        state = _make_state()
        baked = BakeConstantsPropagator.new(propagate, state)
        assert len(baked.consts) == 3

    def test_repeated_calls(self):
        """Baked propagator produces correct results across multiple calls."""
        state = _make_state()
        baked = BakeConstantsPropagator.new(_make_propagator(), state)
        propagate = _make_propagator()
        key = jax.random.key(0)
        for _ in range(3):
            state = baked(key, state)
        expected = _make_state()
        for _ in range(3):
            expected = propagate(key, expected)
        npt.assert_allclose(state.position, expected.position)
        npt.assert_allclose(state.velocity, expected.velocity)
