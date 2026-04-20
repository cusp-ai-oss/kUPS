# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import pytest

from kups.core.utils.functools import (
    compose,
    constant,
    curry,
    flip,
    pack_args,
    pipe,
    select_nth,
    typed,
    uncurry,
    unpack_args,
)


class TestCompose:
    def test_compose(self):
        # Basic composition: compose(f, g)(x) = f(g(x))
        assert compose(lambda x: x + 1, lambda x: x * 2)(5) == 11
        # Order matters
        assert compose(lambda x: x * 2, lambda x: x + 1)(5) == 12
        # Different types
        assert compose(len, str)(12345) == 5


class TestCurryUncurry:
    def test_curry_uncurry(self):
        def add(a: int, b: int) -> int:
            return a + b

        def multiply(a: int, b: int) -> int:
            return a * b

        # Basic currying
        assert curry(add)(5)(3) == 8
        # Reuse curried functions
        double = curry(multiply)(2)
        triple = curry(multiply)(3)
        assert double(5) == 10
        assert triple(7) == 21
        # Curry with different types
        assert curry(lambda s, n: s * n)("hello")(3) == "hellohellohello"
        # curry-uncurry roundtrip
        assert uncurry(curry(add))(5, 3) == 8
        # uncurry-curry roundtrip
        assert curry(uncurry(lambda a: lambda b: a * b))(4)(5) == 20


class TestPackUnpackArgs:
    def test_pack_unpack_args(self):
        def add(a: int, b: int) -> int:
            return a + b

        def sum_three(a: int, b: int, c: int) -> int:
            return a + b + c

        # Pack args
        assert pack_args(add)((3, 5)) == 8
        assert pack_args(sum_three)((1, 2, 3)) == 6
        assert pack_args(lambda x: x * 2)((5,)) == 10
        # Unpack args
        assert unpack_args(lambda args: args[0] + args[1])(3, 5) == 8
        assert unpack_args(lambda args: sum(args))(1, 2, 3) == 6
        # pack-unpack roundtrip
        assert unpack_args(pack_args(add))(3, 5) == add(3, 5)


class TestConstant:
    def test_constant(self):
        assert constant(42)("anything") == 42
        assert constant(42)(None) == 42
        assert constant("hello")(123) == "hello"
        assert constant(None)("test") is None
        lst = [1, 2, 3]
        assert constant(lst)("ignored") is lst


class TestFlip:
    def test_flip(self):
        # Basic flip
        assert flip(lambda a, b: a - b)(10, 3) == -7
        # With strings
        assert flip(lambda a, b: a + b)("hello", "world") == "worldhello"
        # With different types
        assert flip(lambda n, d: n / d)(10.0, 2.0) == pytest.approx(0.2)


class TestTyped:
    def test_happy_paths(self):
        assert typed(lambda x: x * 2, int)(5) == 10
        assert typed(lambda a, b: a + b, int, int)(3, 5) == 8
        assert typed(lambda a, b, c: a + b + c, int, int, int)(1, 2, 3) == 6
        assert typed(lambda n, a: f"{n} is {a}", str, int)("Alice", 30) == "Alice is 30"

        # Subclass accepted
        class Animal:
            pass

        class Dog(Animal):
            pass

        assert typed(lambda a: "ok", Animal)(Dog()) == "ok"
        # Return value preserved
        assert typed(lambda x: (x * 2, str(x)), int)(5) == (10, "5")

    def test_errors(self):
        typed_double = typed(lambda x: x * 2, int)
        with pytest.raises(TypeError, match="Expected type"):
            typed_double("not an int")
        typed_add = typed(lambda a, b: a + b, int, int)
        with pytest.raises(TypeError, match="Expected type"):
            typed_add("x", 5)
        with pytest.raises(TypeError, match="Expected type"):
            typed_add(3, "y")
        with pytest.raises(TypeError, match="Argument count"):
            typed_add(5)
        with pytest.raises(TypeError, match="Argument count"):
            typed_add(5, 3, 7)


class TestPipe:
    def test_pipe(self):
        # pipe(f, g)(x) = g(f(x)), same as compose(g, f)(x)
        assert (
            pipe(lambda x: x + 1, lambda x: x * 3)(10)
            == compose(lambda x: x * 3, lambda x: x + 1)(10)
            == 33
        )


class TestSelectNth:
    def test_select_nth(self):
        assert select_nth(0)("a", "b", "c") == "a"
        assert select_nth(1)(10, 20, 30) == 20
        assert select_nth(2)(1, 2, 3) == 3


class TestFunctoolsIntegration:
    def test_integration(self):
        # compose + curry
        assert compose(lambda x: x * 2, curry(lambda a, b: a + b)(5))(3) == 16
        # flip + curry
        assert curry(flip(lambda a, b: a - b))(10)(3) == -7
        # compose + constant
        assert compose(lambda x: x * 2, constant(5))("anything") == 10
        # typed + compose
        composed = compose(lambda x: x + 1, lambda x: x * 2)
        assert typed(composed, int)(5) == 11
        with pytest.raises(TypeError, match="Expected type"):
            typed(composed, int)("not an int")
        # complex pipeline
        full = compose(compose(len, str), curry(lambda a, b: a + b)(100))
        assert full(23) == 3


if __name__ == "__main__":
    pytest.main([__file__])
