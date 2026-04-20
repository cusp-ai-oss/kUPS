# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for capacity management with scalar and pytree sizes."""

import jax.numpy as jnp

from kups.core.assertion import with_runtime_assertions
from kups.core.capacity import (
    FixedCapacity,
    LensCapacity,
    LensCapacityFix,
    MultipliedCapacity,
)
from kups.core.lens import lens
from kups.core.utils.jax import dataclass


@dataclass
class MockState:
    capacity: int


@dataclass
class MockStateTuple:
    capacity: tuple[int, int]


@dataclass
class MockStateDict:
    capacity: dict[str, int]


def make_base_capacity(size: int = 100) -> LensCapacity[MockState, int]:
    return LensCapacity(size=size, size_lens=lens(lambda s: s.capacity))


def make_tuple_capacity(
    size: tuple[int, int] = (100, 200),
) -> LensCapacity[MockStateTuple, tuple[int, int]]:
    return LensCapacity(size=size, size_lens=lens(lambda s: s.capacity))


def make_dict_capacity(
    size: dict[str, int] | None = None,
) -> LensCapacity[MockStateDict, dict[str, int]]:
    if size is None:
        size = {"atoms": 100, "bonds": 50}
    return LensCapacity(size=size, size_lens=lens(lambda s: s.capacity))


class TestMultipliedCapacity:
    def test_multiplied_capacity(self):
        """Merged: size, nested, assertion_scales, assertion_resize."""
        base = make_base_capacity(100)
        multiplied = MultipliedCapacity(base, factor=3)
        assert multiplied.size == 300

        # nested
        doubled = make_base_capacity(10).multiply(2)
        assert doubled.multiply(2).size == 40

        # assertion scales
        base = make_base_capacity(100)
        multiplied = MultipliedCapacity(base, factor=4)
        updated = multiplied.generate_assertion(jnp.array(200))
        assert updated.size == 400

        # assertion triggers resize
        base = make_base_capacity(10)
        multiplied = MultipliedCapacity(base, factor=2)
        updated = multiplied.generate_assertion(jnp.array(40))
        assert updated.size == 64


class TestPytreeCapacity:
    def test_size_and_structure(self):
        """Merged: tuple_size, dict_size."""
        assert make_tuple_capacity((100, 200)).size == (100, 200)
        assert make_dict_capacity({"atoms": 100, "bonds": 50}).size == {
            "atoms": 100,
            "bonds": 50,
        }

    def test_tuple_assertions(self):
        """Merged: within_bounds, first_exceeds, second_exceeds, both_exceed, batch."""
        cap = make_tuple_capacity((100, 200))
        assert cap.generate_assertion(jnp.array([50, 150])).size == (100, 200)
        assert cap.generate_assertion(jnp.array([150, 150])).size == (256, 200)
        assert cap.generate_assertion(jnp.array([50, 250])).size == (100, 256)
        assert cap.generate_assertion(jnp.array([150, 300])).size == (256, 512)

        # batch
        required = jnp.array([[50, 150], [80, 180], [150, 100]])
        assert cap.generate_assertion(required).size == (256, 200)

    def test_dict_assertions(self):
        """Merged: within_bounds, one_exceeds."""
        cap = make_dict_capacity({"atoms": 100, "bonds": 50})
        assert cap.generate_assertion(jnp.array([80, 40])).size == {
            "atoms": 100,
            "bonds": 50,
        }
        assert cap.generate_assertion(jnp.array([150, 40])).size == {
            "atoms": 256,
            "bonds": 50,
        }


class TestMultipliedPytreeCapacity:
    def test_multiplied_pytree(self):
        """Merged: tuple_mult, dict_mult, nested_mult, assertion_resize."""
        # tuple
        base = make_tuple_capacity((100, 200))
        assert MultipliedCapacity(base, factor=3).size == (300, 600)

        # dict
        base = make_dict_capacity({"atoms": 100, "bonds": 50})
        assert MultipliedCapacity(base, factor=2).size == {"atoms": 200, "bonds": 100}

        # nested
        assert make_tuple_capacity((10, 20)).multiply(2).multiply(2).size == (40, 80)

        # assertion resize
        base = make_tuple_capacity((10, 20))
        multiplied = MultipliedCapacity(base, factor=2)
        updated = multiplied.generate_assertion(jnp.array([50, 30]))
        assert updated.size == (64, 40)


class TestCapacityFix:
    def test_capacity_fix(self):
        """Merged: fix_tuple, fix_dict, max_of_current_target, batch_max."""
        size_lens = lens(lambda s: s.capacity, cls=MockStateTuple)
        fix = LensCapacityFix(size_lens)

        # tuple
        state = MockStateTuple(capacity=(100, 200))
        assert fix(state, jnp.array([150, 250])).capacity == (150, 250)

        # dict
        size_lens_d = lens(lambda s: s.capacity, cls=MockStateDict)
        fix_d = LensCapacityFix(size_lens_d)
        state_d = MockStateDict(capacity={"atoms": 100, "bonds": 50})
        assert fix_d(state_d, jnp.array([150, 80])).capacity == {
            "atoms": 150,
            "bonds": 80,
        }

        # max of current and target
        state = MockStateTuple(capacity=(100, 200))
        assert fix(state, jnp.array([50, 250])).capacity == (100, 250)

        # batch uses max
        state = MockStateTuple(capacity=(100, 200))
        targets = jnp.array([[150, 250], [100, 300], [50, 200]])
        assert fix(state, targets).capacity == (150, 300)


class TestFixedCapacity:
    def test_fixed_capacity(self):
        """Merged: within_bounds, exceeds, multiply."""
        cap = FixedCapacity(size=(100, 200))
        assert cap.generate_assertion(jnp.array([50, 150])).size == (100, 200)

        # exceeds -> assertion fails
        cap = FixedCapacity(size=100)

        @with_runtime_assertions
        def check():
            return cap.generate_assertion(jnp.array(150))

        _, assertions = check()
        assert assertions[0].failed()

        # multiply
        cap = FixedCapacity(size=(10, 20), error_msg="msg")
        multiplied = cap.multiply(3)
        assert multiplied.size == (30, 60)
        assert multiplied.error_msg == "msg"
