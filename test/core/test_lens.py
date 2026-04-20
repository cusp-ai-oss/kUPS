# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for lens functionality."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.data import Table
from kups.core.lens import (
    HasLensFields,
    IndexLens,
    LambdaLens,
    Lens,
    LensField,
    MergedLens,
    Modifier,
    SimpleBoundLens,
    SimpleLens,
    TreePathView,
    all_isinstance_lens,
    all_where_lens,
    bind,
    lens,
    lens_property,
    view,
)
from kups.core.utils.jax import dataclass, field


# Test fixtures and helpers
@dataclass
class Person:
    """Test data structure for lens operations."""

    name: str
    age: int
    address: "Address"


@dataclass
class Address:
    """Nested test data structure."""

    street: str
    city: str
    zipcode: str


@dataclass
class Company:
    """Test data structure with arrays."""

    name: str
    employees: jax.Array  # Array of employee IDs
    scores: jax.Array  # Array of performance scores


@pytest.fixture
def person():
    """Create a test person instance."""
    return Person(
        name="John Doe",
        age=30,
        address=Address(street="123 Main St", city="Anytown", zipcode="12345"),
    )


@pytest.fixture
def company():
    """Create a test company instance."""
    return Company(
        name="Tech Corp",
        employees=jnp.array([101, 102, 103, 104]),
        scores=jnp.array([85.5, 92.0, 78.5, 88.0]),
    )


@pytest.fixture(params=[SimpleLens, LambdaLens])
def direct_lens(request):
    return request.param


# Lens factory fixtures for parameterized testing
@pytest.fixture
def name_lens(direct_lens):
    """Factory for creating name lenses of different types."""

    def get_name(p: Person) -> str:
        return p.name

    if direct_lens is SimpleLens:
        return lens(get_name)
    elif direct_lens is LambdaLens:

        def set_name(state: Person, value: str) -> Person:
            return Person(name=value, age=state.age, address=state.address)

        return LambdaLens(view(get_name), set_name)


@pytest.fixture
def age_lens(direct_lens):
    """Factory for creating age lenses of different types."""

    def get_age(p: Person) -> int:
        return p.age

    if direct_lens is SimpleLens:
        return lens(get_age)
    elif direct_lens is LambdaLens:

        def set_age(state: Person, value: int) -> Person:
            return Person(name=state.name, age=value, address=state.address)

        return LambdaLens(view(get_age), set_age)


@pytest.fixture(params=[("age_lens", 30), ("name_lens", "John Doe")])
def person_lens_and_target(request, direct_lens):
    """Factory for creating person lenses of different types."""
    lens = request.getfixturevalue(request.param[0])
    target = request.param[1]
    return lens, target


@pytest.fixture(params=[SimpleLens, LambdaLens])
def scores_lens(request):
    """Factory for creating scores lenses of different types."""

    def get_scores(c: Company) -> jax.Array:
        return c.scores

    if request.param is SimpleLens:
        return lens(get_scores)
    elif request.param is LambdaLens:

        def set_scores(state: Company, value: jax.Array) -> Company:
            return Company(name=state.name, employees=state.employees, scores=value)

        return LambdaLens(view(get_scores), set_scores)


class TestView:
    """Tests for View protocol and view function."""

    def test_view_creation(self):
        """Test creating a view from a callable."""

        def get_name(p: Person) -> str:
            return p.name

        name_view = view(get_name)

        # Should be callable
        assert callable(name_view)

        # Should have proper representation
        repr_str = repr(name_view)
        assert "View(where=" in repr_str
        assert "get_name" in repr_str

    def test_view_extraction(self, person):
        """Test extracting values using a view."""

        def get_name(p: Person) -> str:
            return p.name

        name_view = view(get_name)
        assert name_view(person) == "John Doe"

    def test_nested_view_extraction(self, person):
        """Test extracting nested values using a view."""

        def get_street(p: Person) -> str:
            return p.address.street

        street_view = view(get_street)
        assert street_view(person) == "123 Main St"


class TestLensOperations:
    """Unified tests for lens operations across different lens types."""

    def test_lens_get(self, person, person_lens_and_target):
        """Test getting values through lenses."""
        lens, target = person_lens_and_target
        assert lens.get(person) == target

    def test_lens_set(self, person, person_lens_and_target):
        """Test setting values through lenses."""
        lens, prev_value = person_lens_and_target

        # Set name
        new_person = lens.set(person, "new_value")
        assert new_person is not person  # Ensure a new instance is returned
        assert lens.get(new_person) == "new_value"

    def test_lens_apply_modifier(self, person, person_lens_and_target):
        """Test applying a modifier function through lenses."""
        lens, val = person_lens_and_target

        # Apply modifier to increment age
        modified_person = lens.apply(person, lambda age: age * 2)
        assert lens.get(modified_person) == val * 2

    def test_lens_focus(self, person):
        """Test focusing lenses on nested data."""

        def get_address(p: Person) -> Address:
            return p.address

        def get_street(addr: Address) -> str:
            return addr.street

        address_lens = lens(get_address)
        street_lens = address_lens.focus(get_street)

        assert street_lens.get(person) == "123 Main St"

        # Test setting through focused lens
        new_person = street_lens.set(person, "456 Oak Ave")
        assert new_person.address.street == "456 Oak Ave"
        assert new_person.address.city == "Anytown"  # Other fields unchanged

    def test_lens_bind(self, person, person_lens_and_target):
        """Test binding lenses to specific objects."""
        lens, target = person_lens_and_target
        bound_lens = lens.bind(person)

        assert isinstance(bound_lens, SimpleBoundLens)
        assert bound_lens.get() == target

        # Test setting through bound lens
        new_person = bound_lens.set("Jane Doe")
        assert lens.get(new_person) == "Jane Doe"

    def test_lens_slice_arrays(self, company, scores_lens):
        """Test slicing arrays through lenses."""
        # Create a slice lens for first two scores
        slice_lens = scores_lens.at(jnp.array([0, 1]))

        # Test getting sliced values
        sliced_scores = slice_lens.get(company)
        npt.assert_array_equal(sliced_scores, jnp.array([85.5, 92.0]))

        # Test setting sliced values
        new_company = slice_lens.set(company, jnp.array([90.0, 95.0]))
        expected_scores = jnp.array([90.0, 95.0, 78.5, 88.0])
        npt.assert_array_equal(new_company.scores, expected_scores)


class TestNestedLens:
    """Tests specific to nested lens functionality."""

    def test_nested_lens_creation(self):
        """Test creating nested lenses."""

        def get_address(p: Person) -> Address:
            return p.address

        def get_street(addr: Address) -> str:
            return addr.street

        address_lens = lens(get_address)
        street_lens = address_lens.focus(get_street)

        assert isinstance(street_lens, Lens)

    def test_deeply_nested_operations(self, person):
        """Test deeply nested lens operations."""

        def get_address(p: Person) -> Address:
            return p.address

        def get_street(addr: Address) -> str:
            return addr.street

        def get_first_char(s: str) -> str:
            return s[0]

        address_lens = lens(get_address)
        street_lens = address_lens.focus(get_street)
        first_char_lens = street_lens.focus(get_first_char)

        assert first_char_lens.get(person) == "1"


class TestBoundLens:
    """Tests for bound lens functionality."""

    def test_bound_lens_operations(self, person):
        """Test all bound lens operations."""

        def get_name(p: Person) -> str:
            return p.name

        name_lens = lens(get_name)
        bound_lens = name_lens.bind(person)

        # Test get
        assert bound_lens.get() == "John Doe"

        # Test set
        new_person = bound_lens.set("Jane Doe")
        assert new_person.name == "Jane Doe"
        assert person.name == "John Doe"  # Original unchanged

        # Test apply
        new_person = bound_lens.apply(lambda name: name.upper())
        assert new_person.name == "JOHN DOE"

        # Test focus
        def get_address(p: Person) -> Address:
            return p.address

        def get_street(addr: Address) -> str:
            return addr.street

        address_lens = lens(get_address)
        address_bound = address_lens.bind(person)
        street_bound = address_bound.focus(get_street)

        assert street_bound.get() == "123 Main St"

    def test_bound_lens_slice(self, company):
        """Test slicing through bound lenses."""

        def get_scores(c: Company) -> jax.Array:
            return c.scores

        scores_lens = lens(get_scores)
        bound_lens = scores_lens.bind(company)

        slice_bound = bound_lens.at(jnp.array([0, 1]))
        sliced_scores = slice_bound.get()
        npt.assert_array_equal(sliced_scores, jnp.array([85.5, 92.0]))


class TestIndexLens:
    """Tests specific to index lens functionality."""

    def test_index_lens_creation(self, company):
        """Test creating index lenses."""

        def get_scores(c: Company) -> jax.Array:
            return c.scores

        scores_lens = lens(get_scores)
        index_lens = scores_lens.at(jnp.array([0, 2]))

        assert isinstance(index_lens, IndexLens)

    def test_index_lens_limitations(self, company):
        """Test index lens limitations."""

        def get_scores(c: Company) -> jax.Array:
            return c.scores

        scores_lens = lens(get_scores)
        index_lens = scores_lens.at(jnp.array([0, 1]))

        with pytest.raises(RuntimeError, match="IndexLens cannot be focused further"):
            index_lens.focus(lambda x: x)

        with pytest.raises(RuntimeError, match="IndexLens cannot be sliced further"):
            index_lens.at(jnp.array([0]))


class TestLambdaLensSpecific:
    """Tests specific to LambdaLens functionality that differs from SimpleLens."""

    def test_lambda_lens_custom_transformation(self, person):
        """Test lambda lens with custom transformation on get."""

        def get_upper_name(person: Person) -> str:
            return person.name.upper()

        def set_name(state: Person, value: str) -> Person:
            return Person(name=value, age=state.age, address=state.address)

        upper_name_lens = LambdaLens(view(get_upper_name), set_name)

        # Test get with transformation
        assert upper_name_lens.get(person) == "JOHN DOE"

        # Test set
        new_person = upper_name_lens.set(person, "Jane Doe")
        assert new_person.name == "Jane Doe"
        assert new_person.age == 30  # Other fields preserved


class TestBindFunction:
    """Tests for the bind function."""

    def test_bind_without_getter(self, person):
        """Test bind function without a getter (identity lens)."""
        bound_lens = bind(person)
        assert isinstance(bound_lens, SimpleBoundLens)
        assert bound_lens.get() == person


class TestLensComposition:
    """Tests for lens composition and complex operations."""

    def test_lens_immutability(self, person):
        """Test that lens operations don't mutate original data."""

        def get_name(p: Person) -> str:
            return p.name

        name_lens = lens(get_name)
        original_name = person.name

        # Modify through lens
        new_person = name_lens.set(person, "Jane Doe")

        # Original should be unchanged
        assert person.name == original_name
        assert new_person.name == "Jane Doe"

    def test_complex_nested_modifications(self, person):
        """Test complex nested modifications."""

        def get_address(p: Person) -> Address:
            return p.address

        def get_street(addr: Address) -> str:
            return addr.street

        def get_city(addr: Address) -> str:
            return addr.city

        # Create multiple nested lenses
        address_lens = lens(get_address)
        street_lens = address_lens.focus(get_street)
        city_lens = address_lens.focus(get_city)

        # Apply multiple modifications
        new_person = street_lens.set(person, "456 Oak Ave")
        new_person = city_lens.set(new_person, "Newtown")

        # Check all modifications applied correctly
        assert new_person.address.street == "456 Oak Ave"
        assert new_person.address.city == "Newtown"
        assert new_person.address.zipcode == "12345"  # Unchanged
        assert new_person.name == "John Doe"  # Unchanged
        assert new_person.age == 30  # Unchanged

    def test_lens_with_jax_transformations(self, company):
        """Test lenses work with JAX transformations."""

        def get_scores(c: Company) -> jax.Array:
            return c.scores

        scores_lens = lens(get_scores)

        # Test with modifications
        def increment_scores(company):
            return scores_lens.set(company, scores_lens.get(company) + 1.0)

        new_company = increment_scores(company)
        expected = company.scores + 1.0
        npt.assert_array_equal(new_company.scores, expected)


class TestErrorHandling:
    """Tests for error handling in lens operations."""

    def test_lens_with_invalid_path(self):
        """Test lens behavior with invalid access paths."""

        def get_invalid_field(p: Person) -> str:
            return p.nonexistent_field  # type: ignore

        invalid_lens = lens(get_invalid_field)

        # The lens creation should succeed
        assert isinstance(invalid_lens, SimpleLens)

        # But accessing should fail
        person = Person("John", 30, Address("123 Main", "Town", "12345"))
        with pytest.raises(AttributeError):
            invalid_lens.get(person)


class TestTypeCompatibility:
    """Tests for type compatibility and edge cases."""

    def test_lens_with_arrays(self, company):
        """Test lenses work properly with JAX arrays."""

        def get_employees(c: Company) -> jax.Array:
            return c.employees

        employees_lens = lens(get_employees)

        # Test getting arrays
        employees = employees_lens.get(company)
        assert isinstance(employees, jax.Array)

        # Test setting arrays
        new_employees = jnp.array([201, 202, 203])
        new_company = employees_lens.set(company, new_employees)
        npt.assert_array_equal(new_company.employees, new_employees)

    def test_modifier_type_alias(self):
        """Test that Modifier type alias works correctly."""

        def apply_modifier(value: int, mod: Modifier[int]) -> int:
            return mod(value)

        result = apply_modifier(5, lambda x: x * 2)
        assert result == 10

    def test_lens_with_complex_data_structures(self):
        """Test lenses with more complex nested structures."""

        @dataclass
        class NestedData:
            values: jax.Array
            metadata: dict[str, Any]

        @dataclass
        class ComplexStruct:
            data: NestedData
            name: str

        # Create test data
        nested_data = NestedData(
            values=jnp.array([1.0, 2.0, 3.0]),
            metadata={"version": "1.0", "author": "test"},
        )
        complex_struct = ComplexStruct(data=nested_data, name="test_struct")

        # Create lens for nested array
        def get_data(cs: ComplexStruct) -> NestedData:
            return cs.data

        def get_values(nd: NestedData) -> jax.Array:
            return nd.values

        data_lens = lens(get_data)
        values_lens = data_lens.focus(get_values)

        # Test getting nested array
        values = values_lens.get(complex_struct)
        npt.assert_array_equal(values, jnp.array([1.0, 2.0, 3.0]))

        # Test setting nested array
        new_values = jnp.array([4.0, 5.0, 6.0])
        new_struct = values_lens.set(complex_struct, new_values)
        npt.assert_array_equal(new_struct.data.values, new_values)
        assert new_struct.name == "test_struct"  # Other fields preserved


class TestMergedLens:
    """Tests for MergedLens functionality."""

    @pytest.fixture
    def person_lenses(self):
        """Create name and age lenses for testing."""
        name_lens = lens(lambda p: p.name, cls=Person)
        age_lens = lens(lambda p: p.age, cls=Person)
        return name_lens, age_lens

    @pytest.fixture
    def merged_lens(self, person_lenses):
        """Create a merged lens from name and age lenses."""
        name_lens, age_lens = person_lenses
        return name_lens.merge(age_lens)

    def test_merge_creation(self, person_lenses):
        """Test creating a merged lens using the merge method."""
        name_lens, age_lens = person_lenses
        merged = name_lens.merge(age_lens)

        assert isinstance(merged, MergedLens)
        # Access the actual MergedLens attributes
        assert hasattr(merged, "left")
        assert hasattr(merged, "right")

    def test_merged_lens_get(self, person, merged_lens):
        """Test getting values through a merged lens."""
        result = merged_lens.get(person)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "John Doe"  # name
        assert result[1] == 30  # age

    def test_merged_lens_set(self, person, merged_lens):
        """Test setting values through a merged lens."""
        new_values = ("Jane Smith", 25)
        new_person = merged_lens.set(person, new_values)

        # Check that a new instance is created
        assert new_person is not person

        # Check that both values were set correctly
        assert new_person.name == "Jane Smith"
        assert new_person.age == 25

        # Check that other attributes are preserved
        assert new_person.address == person.address

    def test_merged_lens_set_partial_values(self, person, merged_lens):
        """Test setting only one value in a merged lens tuple."""
        # Set only the name, keep the age the same
        new_values = ("Jane Smith", 30)
        new_person = merged_lens.set(person, new_values)

        assert new_person.name == "Jane Smith"
        assert new_person.age == 30

    def test_merged_lens_focus(self, person, merged_lens):
        """Test focusing on part of the merged lens result."""
        # Focus on the first element (name)
        name_focused = merged_lens.focus(lambda tuple_val: tuple_val[0])

        assert name_focused.get(person) == "John Doe"

        new_person = name_focused.set(person, "Jane Smith")
        assert new_person.name == "Jane Smith"
        assert new_person.age == 30  # age should be unchanged

    def test_merged_lens_focus_second_element(self, person, merged_lens):
        """Test focusing on the second element of merged lens result."""
        # Focus on the second element (age)
        age_focused = merged_lens.focus(lambda tuple_val: tuple_val[1])

        assert age_focused.get(person) == 30

        new_person = age_focused.set(person, 35)
        assert new_person.name == "John Doe"  # name should be unchanged
        assert new_person.age == 35

    def test_merged_lens_apply(self, person, merged_lens):
        """Test applying a modifier to merged lens values."""
        # Apply a modifier that uppercases the name and doubles the age
        result = merged_lens.apply(person, lambda vals: (vals[0].upper(), vals[1] * 2))
        assert result.name == "JOHN DOE"
        assert result.age == 60  # 30 * 2

    def test_merged_lens_bind(self, person, merged_lens):
        """Test binding a merged lens to a specific data structure."""
        bound_lens = merged_lens.bind(person)

        assert isinstance(bound_lens, SimpleBoundLens)

        # Test bound operations
        result = bound_lens.get()
        assert result == ("John Doe", 30)

        new_person = bound_lens.set(("Jane Smith", 25))
        assert new_person.name == "Jane Smith"
        assert new_person.age == 25

    def test_merged_lens_at(self, merged_lens):
        """Test creating an indexed lens from a merged lens."""
        # Create test data with arrays
        test_data = Company(
            name="Test Corp",
            employees=jnp.array([1, 2, 3, 4]),
            scores=jnp.array([85.0, 90.0, 75.0, 88.0]),
        )

        # Create lenses for employees and scores
        employees_lens = lens(lambda c: c.employees, cls=Company)
        scores_lens = lens(lambda c: c.scores, cls=Company)
        merged = employees_lens.merge(scores_lens)

        # Create indexed lens for first two elements
        indexed_lens = merged.at(slice(0, 2))

        result = indexed_lens.get(test_data)
        expected_employees = jnp.array([1, 2])
        expected_scores = jnp.array([85.0, 90.0])

        npt.assert_array_equal(result[0], expected_employees)
        npt.assert_array_equal(result[1], expected_scores)

    def test_nested_merged_lens(self, person):
        """Test merging a merged lens with another lens."""
        name_lens = lens(lambda p: p.name, cls=Person)
        age_lens = lens(lambda p: p.age, cls=Person)
        address_lens = lens(lambda p: p.address.city, cls=Person)

        # First merge name and age
        name_age_merged = name_lens.merge(age_lens)

        # Then merge that with address
        triple_merged = name_age_merged.merge(address_lens)

        result = triple_merged.get(person)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], tuple)  # First element is the merged (name, age)
        assert len(result[0]) == 2
        assert result[0][0] == "John Doe"  # name
        assert result[0][1] == 30  # age
        assert result[1] == "Anytown"  # city

    def test_merged_lens_with_nested_lens(self, person):
        """Test merging lenses where one is a nested lens."""
        name_lens = lens(lambda p: p.name, cls=Person)
        street_lens = lens(lambda p: p.address, cls=Person).focus(
            lambda addr: addr.street
        )

        merged = name_lens.merge(street_lens)

        result = merged.get(person)
        assert result == ("John Doe", "123 Main St")

        new_person = merged.set(person, ("Jane Smith", "456 Oak Ave"))
        assert new_person.name == "Jane Smith"
        assert new_person.address.street == "456 Oak Ave"
        assert new_person.address.city == "Anytown"  # unchanged
        assert new_person.age == 30  # unchanged

    def test_merged_lens_with_lambda_lens(self, person):
        """Test merging with a LambdaLens."""
        name_lens = lens(lambda p: p.name, cls=Person)

        # Create a lambda lens for age
        def set_age(state: Person, value: int) -> Person:
            return Person(name=state.name, age=value, address=state.address)

        age_lambda_lens = LambdaLens(view(lambda p: p.age), set_age)

        merged = name_lens.merge(age_lambda_lens)

        result = merged.get(person)
        assert result == ("John Doe", 30)

        new_person = merged.set(person, ("Jane Smith", 25))
        assert new_person.name == "Jane Smith"
        assert new_person.age == 25

    def test_merged_lens_immutability(self, person, merged_lens):
        """Test that merged lens operations don't mutate the original data."""
        original_name = person.name
        original_age = person.age

        # Perform various operations
        merged_lens.get(person)
        merged_lens.apply(person, lambda x: (x[0].upper(), x[1] + 10))

        # Original should be unchanged
        assert person.name == original_name
        assert person.age == original_age

    def test_merged_lens_bound_focus(self, person, merged_lens):
        """Test focusing on a bound merged lens."""
        bound_lens = merged_lens.bind(person)

        # Focus on the name (first element)
        name_focused = bound_lens.focus(lambda vals: vals[0])

        assert name_focused.get() == "John Doe"
        new_person = name_focused.set("Jane Smith")
        assert new_person.name == "Jane Smith"
        assert new_person.age == 30

    def test_merged_lens_bound_merge(self, person):
        """Test merging a bound lens with another lens."""
        name_lens = lens(lambda p: p.name, cls=Person)
        age_lens = lens(lambda p: p.age, cls=Person)

        # Bind the name lens and merge with age lens
        bound_name = name_lens.bind(person)
        merged_with_age = bound_name.merge(age_lens)

        result = merged_with_age.get()
        assert result == ("John Doe", 30)


class TestTreePathView:
    """Tests for TreePathView functionality."""

    def test_tree_path_view_creation(self):
        """Test creating a TreePathView."""
        from jax.tree_util import GetAttrKey

        path = (GetAttrKey("address"), GetAttrKey("street"))
        view = TreePathView(path)

        assert view.path == path

    def test_tree_path_view_attribute_access(self, person):
        """Test TreePathView with attribute access."""
        from jax.tree_util import GetAttrKey

        # Create path to person.name
        path = (GetAttrKey("name"),)
        view = TreePathView(path)

        result = view(person)
        assert result == "John Doe"

    def test_tree_path_view_nested_attribute_access(self, person):
        """Test TreePathView with nested attribute access."""
        from jax.tree_util import GetAttrKey

        # Create path to person.address.street
        path = (GetAttrKey("address"), GetAttrKey("street"))
        view = TreePathView(path)

        result = view(person)
        assert result == "123 Main St"

    def test_tree_path_view_dict_access(self):
        """Test TreePathView with dictionary access."""
        from jax.tree_util import DictKey

        data = {"user": {"name": "John", "age": 30}}
        path = (DictKey("user"), DictKey("name"))
        view = TreePathView(path)

        result = view(data)
        assert result == "John"

    def test_tree_path_view_sequence_access(self):
        """Test TreePathView with sequence access."""
        from jax.tree_util import SequenceKey

        data = [["a", "b"], ["c", "d"]]
        path = (SequenceKey(0), SequenceKey(1))
        view = TreePathView(path)

        result = view(data)
        assert result == "b"

    def test_tree_path_view_mixed_access(self):
        """Test TreePathView with mixed access types."""
        from jax.tree_util import DictKey, GetAttrKey, SequenceKey

        @dataclass
        class Container:
            data: dict[str, Any]

        container = Container(data={"items": [{"name": "first"}, {"name": "second"}]})

        path = (
            GetAttrKey("data"),
            DictKey("items"),
            SequenceKey(1),
            DictKey("name"),
        )
        view = TreePathView(path)

        result = view(container)
        assert result == "second"

    def test_tree_path_view_empty_path(self):
        """Test TreePathView with empty path returns original object."""
        view = TreePathView(())

        data = {"test": "value"}
        result = view(data)
        assert result is data

    def test_tree_path_view_invalid_key_type(self):
        """Test TreePathView with invalid key type raises error."""

        # Create a mock key that isn't one of the expected types
        class InvalidKey:
            pass

        path = (InvalidKey(),)
        view = TreePathView(path)

        with pytest.raises(TypeError, match="Unknown path type"):
            view({"test": "data"})


class TestAllWhereLens:
    """Tests for all_where_lens functionality."""

    def test_all_where_lens_find_numbers(self):
        """Test all_where_lens finding all numbers in a pytree."""

        # Create mixed data structure
        data = {
            "numbers": [1, 2, 3],
            "strings": ["a", "b"],
            "nested": {"value": 42, "text": "hello"},
        }

        # Create lens that finds all integers
        number_lens = all_where_lens(data, lambda x: isinstance(x, int))

        result = number_lens.get(data)
        # Should find all integers: 1, 2, 3, 42
        assert len(result) == 4
        assert set(result) == {1, 2, 3, 42}

    def test_all_where_lens_find_strings(self):
        """Test all_where_lens finding all strings in a pytree."""

        data = {
            "numbers": [1, 2, 3],
            "strings": ["hello", "world"],
            "nested": {"text": "nested_string", "value": 42},
        }

        # Create lens that finds all strings
        string_lens = all_where_lens(data, lambda x: isinstance(x, str))

        result = string_lens.get(data)
        assert len(result) == 3
        assert set(result) == {"hello", "world", "nested_string"}

    def test_all_where_lens_custom_condition(self):
        """Test all_where_lens with custom condition."""

        data = {"values": [1, 5, 10, 15, 20], "other": {"num": 8, "text": "test"}}

        # Find all numbers greater than 10
        large_number_lens = all_where_lens(
            data, lambda x: isinstance(x, int) and x > 10
        )

        result = large_number_lens.get(data)
        assert len(result) == 2
        assert set(result) == {15, 20}

    def test_all_where_lens_with_dataclasses(self, person):
        """Test all_where_lens finding dataclass instances."""

        data = {
            "person": person,
            "address": person.address,
            "numbers": [1, 2, 3],
            "nested": {"another_address": Address("Test St", "Test City", "00000")},
        }

        # Find all Address instances
        address_lens = all_where_lens(data, lambda x: isinstance(x, Address))

        result = address_lens.get(data)
        # Note: all_where_lens uses is_leaf=conditional, so it treats matching objects as leaves
        # This means person.address appears twice: once directly as "address" and once as person.address
        assert len(result) == 3
        # Check that we found the right addresses
        streets = {addr.street for addr in result}
        assert streets == {"123 Main St", "Test St"}

    def test_all_where_lens_empty_result(self):
        """Test all_where_lens when no elements match."""

        data = {"strings": ["a", "b", "c"], "lists": [[], [1, 2]]}

        # Look for floats (should find none)
        float_lens = all_where_lens(data, lambda x: isinstance(x, float))

        result = float_lens.get(data)
        assert result == ()

    def test_all_where_lens_complex_structure(self):
        """Test all_where_lens with complex nested structure."""

        @dataclass
        class Node:
            value: int
            children: tuple["Node", ...]

        # Create tree structure
        leaf1 = Node(1, ())
        leaf2 = Node(2, ())
        leaf3 = Node(3, ())
        branch = Node(10, (leaf1, leaf2))
        root = Node(100, (branch, leaf3))

        # Find all Node instances
        node_lens = all_where_lens(root, lambda x: isinstance(x, Node))

        result = node_lens.get(root)
        # Note: all_where_lens treats matching objects as leaves, so it only finds the root
        # because once it finds a Node, it doesn't traverse its children
        assert len(result) == 1  # Only the root node

        # Check we got the right value
        assert result[0].value == 100

    def test_all_where_lens_with_arrays(self):
        """Test all_where_lens with JAX arrays."""

        data = {
            "arrays": [jnp.array([1, 2]), jnp.array([3, 4])],
            "scalars": [1, 2, 3],
            "nested": {"arr": jnp.array([5, 6])},
        }

        # Find all JAX arrays
        array_lens = all_where_lens(data, lambda x: isinstance(x, jax.Array))

        result = array_lens.get(data)
        assert len(result) == 3

        # Check array contents
        all_values = []
        for arr in result:
            all_values.extend(arr.tolist())
        assert set(all_values) == {1, 2, 3, 4, 5, 6}


class TestAllIsinstanceLens:
    """Tests for all_isinstance_lens functionality."""

    def test_all_isinstance_lens_find_integers(self):
        """Test all_isinstance_lens finding integers."""

        data = {
            "mixed": [1, "hello", 2, 3.14, 3],
            "nested": {"value": 42, "text": "world"},
        }

        int_lens = all_isinstance_lens(data, int)
        result = int_lens.get(data)

        assert len(result) == 4
        assert set(result) == {1, 2, 3, 42}

    def test_all_isinstance_lens_find_strings(self):
        """Test all_isinstance_lens finding strings."""

        data = {
            "mixed": [1, "hello", 2, "world"],
            "nested": {"text": "nested", "num": 42},
        }

        string_lens = all_isinstance_lens(data, str)
        result = string_lens.get(data)

        assert len(result) == 3
        assert set(result) == {"hello", "world", "nested"}

    def test_all_isinstance_lens_find_addresses(self, person):
        """Test all_isinstance_lens finding Address instances."""

        data = {
            "person": person,
            "addresses": [
                Address("First St", "City 1", "11111"),
                Address("Second St", "City 2", "22222"),
            ],
            "other": {"count": 5, "name": "test"},
        }

        address_lens = all_isinstance_lens(data, Address)
        result = address_lens.get(data)

        assert len(result) == 3  # person.address + 2 in addresses list
        streets = {addr.street for addr in result}
        assert streets == {"123 Main St", "First St", "Second St"}

    def test_all_isinstance_lens_find_persons(self, person):
        """Test all_isinstance_lens finding Person instances."""

        other_person = Person("Jane", 25, Address("Other St", "Other City", "99999"))

        data = {
            "people": [person, other_person],
            "other": {"address": person.address, "count": 2},
        }

        person_lens = all_isinstance_lens(data, Person)
        result = person_lens.get(data)

        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"John Doe", "Jane"}

    def test_all_isinstance_lens_find_arrays(self):
        """Test all_isinstance_lens finding JAX arrays."""

        data = {
            "arrays": [jnp.array([1, 2]), jnp.array([3, 4])],
            "lists": [[5, 6], [7, 8]],
            "nested": {"tensor": jnp.array([9, 10])},
        }

        array_lens = all_isinstance_lens(data, jax.Array)
        result = array_lens.get(data)

        assert len(result) == 3
        # Check all arrays have expected values
        all_values = []
        for arr in result:
            all_values.extend(arr.tolist())
        assert set(all_values) == {1, 2, 3, 4, 9, 10}

    def test_all_isinstance_lens_empty_result(self):
        """Test all_isinstance_lens when no instances found."""

        data = {"strings": ["a", "b", "c"], "numbers": [1, 2, 3], "lists": [[], [1]]}

        # Look for Address instances (should find none)
        address_lens = all_isinstance_lens(data, Address)
        result = address_lens.get(data)

        assert result == ()

    def test_all_isinstance_lens_inheritance(self):
        """Test all_isinstance_lens with class inheritance."""

        @dataclass
        class Animal:
            name: str

        @dataclass
        class Dog(Animal):
            breed: str

        @dataclass
        class Cat(Animal):
            indoor: bool

        dog = Dog("Buddy", "Labrador")
        cat = Cat("Whiskers", True)

        data = {"pets": [dog, cat], "other": {"count": 2}}

        # Find all Animal instances (should include both Dog and Cat)
        animal_lens = all_isinstance_lens(data, Animal)
        result = animal_lens.get(data)

        assert len(result) == 2
        names = {animal.name for animal in result}
        assert names == {"Buddy", "Whiskers"}

        # Find only Dog instances
        dog_lens = all_isinstance_lens(data, Dog)
        dog_result = dog_lens.get(data)

        assert len(dog_result) == 1
        assert dog_result[0].name == "Buddy"

    def test_all_isinstance_lens_nested_complex_structure(self):
        """Test all_isinstance_lens with deeply nested structure."""

        @dataclass
        class Container:
            items: list[Any]

        inner_addresses = [
            Address("Inner 1", "Inner City 1", "11111"),
            Address("Inner 2", "Inner City 2", "22222"),
        ]

        container = Container(inner_addresses)

        data = {
            "top_address": Address("Top", "Top City", "00000"),
            "container": container,
            "nested": {"deep": {"address": Address("Deep", "Deep City", "33333")}},
        }

        address_lens = all_isinstance_lens(data, Address)
        result = address_lens.get(data)

        assert len(result) == 4  # top + 2 inner + 1 deep
        streets = {addr.street for addr in result}
        assert streets == {"Top", "Inner 1", "Inner 2", "Deep"}

    def test_all_isinstance_lens_with_static_field_comprehension(self):
        """Test all_isinstance_lens with tuple comprehension focusing on static field."""

        @dataclass
        class Item:
            size: int = field(static=True)

        item = Item(1)

        # Create lens that finds all Item instances and focuses on their size fields
        size_lens = all_isinstance_lens(item, Item).focus(
            lambda x: tuple(y.size for y in x)
        )

        # Get should return tuple of sizes
        result = size_lens.get(item)
        assert result == (1,)

        # Set should update the size field
        new_item = size_lens.set(item, (2,))
        assert new_item.size == 2
        assert item.size == 1  # Original unchanged


class TestLensField:
    """Tests for the LensField descriptor."""

    def test_basic_lensfield_operations(self):
        """Test basic LensField operations: class access, instance access, get, set."""

        @dataclass
        class Point(HasLensFields):
            x: LensField[float] = field(default=0.0)
            y: LensField[float] = field(default=0.0)

        # Class access returns a lens
        x_lens = Point.x
        y_lens = Point.y
        assert isinstance(x_lens, SimpleLens)
        assert isinstance(y_lens, SimpleLens)

        # Instance access returns value
        point = Point(x=3.0, y=4.0)
        assert point.x == 3.0
        assert point.y == 4.0

        # Lens get works
        assert x_lens.get(point) == 3.0
        assert y_lens.get(point) == 4.0

        # Lens set works and preserves other fields
        new_point = x_lens.set(point, 5.0)
        assert new_point.x == 5.0
        assert new_point.y == 4.0  # y unchanged
        assert point.x == 3.0  # Original unchanged

    def test_lens_focus_composition(self):
        """Test that LensField lenses can be composed with focus."""

        @dataclass
        class Point(HasLensFields):
            x: LensField[float] = field(default=0.0)
            y: LensField[float] = field(default=0.0)

        @dataclass
        class Line(HasLensFields):
            start: LensField[Point]
            end: LensField[Point]

        line = Line(start=Point(x=0.0, y=0.0), end=Point(x=10.0, y=10.0))

        # Compose lenses: Line.start then Point.x
        start_x_lens = Line.start.focus(lambda p: p.x)

        assert start_x_lens.get(line) == 0.0

        new_line = start_x_lens.set(line, 5.0)
        assert new_line.start.x == 5.0
        assert new_line.start.y == 0.0
        assert new_line.end.x == 10.0

    def test_with_jax_arrays(self):
        """Test LensField works with JAX arrays."""

        @dataclass
        class ArrayContainer(HasLensFields):
            data: LensField[jax.Array]

        container = ArrayContainer(data=jnp.array([1, 2, 3]))

        # Instance access
        npt.assert_array_equal(container.data, jnp.array([1, 2, 3]))

        # Lens access
        data_lens = ArrayContainer.data
        npt.assert_array_equal(data_lens.get(container), jnp.array([1, 2, 3]))

        # Lens set
        new_container = data_lens.set(container, jnp.array([4, 5, 6]))
        npt.assert_array_equal(new_container.data, jnp.array([4, 5, 6]))

    def test_with_nested_dataclasses(self):
        """Test LensField with nested dataclass structures."""

        @dataclass
        class Inner(HasLensFields):
            value: LensField[int] = field(default=0)

        @dataclass
        class Outer(HasLensFields):
            inner: LensField[Inner]

        obj = Outer(inner=Inner(value=42))

        # Access through nested structure
        assert obj.inner.value == 42

        # Use lens to access inner
        inner_lens = Outer.inner
        inner = inner_lens.get(obj)
        assert inner.value == 42

        # Compose lenses
        value_lens = Outer.inner.focus(lambda i: i.value)
        assert value_lens.get(obj) == 42

        new_obj = value_lens.set(obj, 100)
        assert new_obj.inner.value == 100
        assert obj.inner.value == 42  # Original unchanged

    def test_lens_apply_modifier(self):
        """Test that LensField lenses work with apply modifier."""

        @dataclass
        class Counter(HasLensFields):
            count: LensField[int] = field(default=0)

        counter = Counter(count=5)

        count_lens = Counter.count
        new_counter = count_lens.apply(counter, lambda x: x + 10)

        assert new_counter.count == 15
        assert counter.count == 5  # Original unchanged

    def test_lens_bind(self):
        """Test that LensField lenses can be bound to instances."""

        @dataclass
        class Point(HasLensFields):
            x: LensField[float] = field(default=0.0)
            y: LensField[float] = field(default=0.0)

        point = Point(x=3.0, y=4.0)

        # Bind the lens to the instance
        bound_x = Point.x.bind(point)

        assert bound_x.get() == 3.0

        new_point = bound_x.set(7.0)
        assert new_point.x == 7.0

    def test_works_in_jax_jit(self):
        """Test that LensField works within JAX JIT compilation."""

        @dataclass
        class Point(HasLensFields):
            x: LensField[float] = field(default=0.0)
            y: LensField[float] = field(default=0.0)

        @jax.jit
        def double_x(p: Point) -> Point:
            return Point.x.set(p, p.x * 2)

        point = Point(x=5.0, y=3.0)
        result = double_x(point)

        assert result.x == 10.0
        assert result.y == 3.0

    def test_with_none_values(self):
        """Test that LensField works with None values."""

        @dataclass
        class Container(HasLensFields):
            value: LensField[int | None]

        # Create with None
        container = Container(value=None)
        assert container.value is None

        # Get with lens
        value_lens = Container.value
        assert value_lens.get(container) is None

        # Set to non-None
        new_container = value_lens.set(container, 42)
        assert new_container.value == 42
        assert container.value is None

        # Set back to None
        none_container = value_lens.set(new_container, None)
        assert none_container.value is None

    def test_inheritance_with_lensfield(self):
        """Test that LensField works correctly with inheritance."""

        @dataclass
        class Base(HasLensFields):
            x: LensField[float] = field(default=0.0)

        @dataclass
        class Derived(Base):
            y: LensField[float] = field(default=0.0)

        obj = Derived(x=1.0, y=2.0)

        # Both fields should work
        assert obj.x == 1.0
        assert obj.y == 2.0

        # Both lenses should work
        x_lens = Derived.x
        y_lens = Derived.y

        assert x_lens.get(obj) == 1.0
        assert y_lens.get(obj) == 2.0

        new_obj = x_lens.set(obj, 10.0)
        assert new_obj.x == 10.0
        assert new_obj.y == 2.0

    def test_complex_type_annotations(self):
        """Test LensField with complex type annotations."""

        @dataclass
        class ComplexTypes(HasLensFields):
            list_field: LensField[list[int]]
            dict_field: LensField[dict[str, float]]
            tuple_field: LensField[tuple[int, str]]

        obj = ComplexTypes(
            list_field=[1, 2, 3],
            dict_field={"a": 1.0, "b": 2.0},
            tuple_field=(42, "hello"),
        )

        # Instance access
        assert obj.list_field == [1, 2, 3]
        assert obj.dict_field == {"a": 1.0, "b": 2.0}
        assert obj.tuple_field == (42, "hello")

        # Lens access
        assert ComplexTypes.list_field.get(obj) == [1, 2, 3]
        assert ComplexTypes.dict_field.get(obj) == {"a": 1.0, "b": 2.0}
        assert ComplexTypes.tuple_field.get(obj) == (42, "hello")

        # Lens set
        new_obj = ComplexTypes.list_field.set(obj, [4, 5, 6])
        assert new_obj.list_field == [4, 5, 6]
        assert obj.list_field == [1, 2, 3]  # Original unchanged

    def test_lensfield_with_empty_containers(self):
        """Test LensField with empty container types."""

        @dataclass
        class Containers(HasLensFields):
            empty_list: LensField[list] = field(default_factory=list)
            empty_dict: LensField[dict] = field(default_factory=dict)

        obj = Containers(empty_list=[], empty_dict={})

        assert obj.empty_list == []
        assert obj.empty_dict == {}

        # Lens operations should work
        new_obj = Containers.empty_list.set(obj, [1, 2, 3])
        assert new_obj.empty_list == [1, 2, 3]
        assert obj.empty_list == []  # Original unchanged

    def test_forward_reference_in_lensfield(self):
        """Test that LensField works with forward references.

        This test ensures that get_type_hints() is used instead of accessing
        dataclass_fields[name].type directly, as forward references need to be
        resolved properly.
        """

        @dataclass
        class Node(HasLensFields):
            value: LensField[int]
            # Forward reference using string annotation
            next: LensField["Node | None"]

        node2 = Node(value=2, next=None)
        node1 = Node(value=1, next=node2)

        # Class access should return lenses
        value_lens = Node.value
        next_lens = Node.next
        assert isinstance(value_lens, Lens)
        assert isinstance(next_lens, Lens)

        # Lens operations should work
        assert value_lens.get(node1) == 1
        assert next_lens.get(node1) == node2

        # Update operations
        new_node = value_lens.set(node1, 10)
        assert new_node.value == 10
        assert node1.value == 1  # Original unchanged

    def test_string_annotation_in_lensfield(self):
        """Test that LensField works with string annotations.

        String annotations (PEP 563) require get_type_hints() to resolve properly.
        """

        @dataclass
        class Container(HasLensFields):
            # String annotations that need to be resolved
            data: LensField["Array"]
            count: LensField["int"]

        arr = jnp.array([1.0, 2.0, 3.0])
        container = Container(data=arr, count=3)

        # Class access should return lenses
        data_lens = Container.data
        count_lens = Container.count
        assert isinstance(data_lens, Lens)
        assert isinstance(count_lens, Lens)

        # Lens operations should work
        assert jnp.array_equal(data_lens.get(container), arr)
        assert count_lens.get(container) == 3

        # Update operations
        new_arr = jnp.array([4.0, 5.0, 6.0])
        new_container = data_lens.set(container, new_arr)
        assert jnp.array_equal(new_container.data, new_arr)
        assert jnp.array_equal(container.data, arr)  # Original unchanged

    def test_nested_forward_reference(self):
        """Test LensField with nested structures containing forward references."""

        @dataclass
        class Tree(HasLensFields):
            value: LensField[int]
            # Forward references to Tree itself
            left: LensField["Tree | None"]
            right: LensField["Tree | None"]

        # Build a small tree
        leaf1 = Tree(value=3, left=None, right=None)
        leaf2 = Tree(value=7, left=None, right=None)
        root = Tree(value=5, left=leaf1, right=leaf2)

        # Access lenses
        value_lens = Tree.value
        left_lens = Tree.left
        right_lens = Tree.right

        # Navigate and update
        assert value_lens.get(root) == 5
        assert left_lens.get(root).value == 3
        assert right_lens.get(root).value == 7

        # Update nested value
        new_leaf = value_lens.set(leaf1, 30)
        new_root = left_lens.set(root, new_leaf)
        assert new_root.left.value == 30
        assert root.left.value == 3  # Original unchanged


class TestMethodInvocationInLenses:
    """Tests for method invocation support in traversal lenses (issue #153).

    The key insight is that method calls work in lens focus when the method
    returns something that maps back to fields in the original object.
    """

    def test_method_call_returning_field_reference_get(self):
        """Test get() with method that returns reference to original fields."""

        @dataclass
        class A:
            a: jax.Array

        @dataclass
        class B:
            a: jax.Array
            b: jax.Array

            def as_a(self) -> A:
                return A(self.a)

        b = B(a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6]))

        # Method call in focus - get should work
        result = bind(b).focus(lambda x: x.as_a()).get()

        assert isinstance(result, A)
        assert jnp.array_equal(result.a, b.a)

    def test_method_call_returning_field_reference_set(self):
        """Test set() with method that returns reference to original fields.

        This is the key use case from issue #153 comment by n-gao:
        bind(b).focus(lambda x: x.as_a()).set(a) should work because
        as_a() returns A(self.a) which references self.a.
        """

        @dataclass
        class A:
            a: jax.Array

        @dataclass
        class B:
            a: jax.Array
            b: jax.Array

            def as_a(self) -> A:
                return A(self.a)

        b = B(a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6]))
        new_a = A(a=jnp.array([10, 20, 30]))

        # This should work! The method returns a reference to self.a
        new_b = bind(b).focus(lambda x: x.as_a()).set(new_a)

        assert jnp.array_equal(new_b.a, jnp.array([10, 20, 30]))
        assert jnp.array_equal(new_b.b, b.b)  # b unchanged
        assert jnp.array_equal(b.a, jnp.array([1, 2, 3]))  # Original unchanged

    def test_method_call_equivalent_to_class_method_access(self):
        """Test that lambda x: x.method() works like ClassName.method."""

        @dataclass
        class A:
            a: jax.Array

        @dataclass
        class B:
            a: jax.Array
            b: jax.Array

            def as_a(self) -> A:
                return A(self.a)

        @dataclass
        class C:
            b: B

        b = B(a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6]))
        c = C(b=b)
        new_a = A(a=jnp.array([10, 20, 30]))

        # These should be equivalent:
        result1 = bind(c).focus(lambda c: c.b).focus(B.as_a).set(new_a)
        result2 = bind(c).focus(lambda c: c.b.as_a()).set(new_a)

        assert jnp.array_equal(result1.b.a, result2.b.a)
        assert jnp.array_equal(result1.b.a, jnp.array([10, 20, 30]))

    def test_method_call_with_nested_focus(self):
        """Test method calls in nested focus chains."""

        @dataclass
        class A:
            a: jax.Array

        @dataclass
        class B:
            a: jax.Array
            b: jax.Array

            def as_a(self) -> A:
                return A(self.a)

        @dataclass
        class C:
            b: B
            c: jax.Array

        b = B(a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6]))
        c = C(b=b, c=jnp.array([7, 8, 9]))
        new_a = A(a=jnp.array([100, 200, 300]))

        # Focus through nested structure with method call
        result = bind(c).focus(lambda c: c.b.as_a()).set(new_a)

        assert jnp.array_equal(result.b.a, jnp.array([100, 200, 300]))
        assert jnp.array_equal(result.b.b, b.b)
        assert jnp.array_equal(result.c, c.c)

    def test_indexed_map_data_get(self):
        """Test that Table.map_data works with get() (issue #153)."""
        x = Table(tuple(range(5)), jnp.arange(5))

        # Getting through map_data should work
        result = bind(x).focus(lambda s: s.map_data(lambda x: x * 2)).get()

        assert isinstance(result, Table)
        assert jnp.array_equal(result.data, jnp.arange(5) * 2)

    def test_apply_for_transformations(self):
        """Test that apply() works for transformations (as suggested in issue #153)."""
        x = Table(tuple(range(5)), jnp.arange(5))

        # Use apply for transformations as suggested by jonkhler
        result = bind(x).apply(lambda s: s.map_data(lambda x: x * 2))

        assert isinstance(result, Table)
        assert jnp.array_equal(result.data, jnp.arange(5) * 2)


class TestTraversalLensIteration:
    """Tests for iteration support in traversal lenses.

    Note: Tests use mutable containers (dict/list) which use shallow copy.
    These warnings are expected and intentional - tested explicitly in TestItemLens.
    """

    def test_dict_iteration_and_setting(self):
        """Test dict iteration patterns for both GET and SET."""

        @dataclass
        class Container:
            data: dict = field(static=True)

        c = Container(data={"a": 1, "b": 2, "c": 3})

        # Dict iteration yields keys
        result = bind(c).focus(lambda c: [k for k in c.data]).get()
        assert result == ["a", "b", "c"]

        # .keys(), .values(), .items() work for GET
        assert bind(c).focus(lambda c: list(c.data.keys())).get() == ["a", "b", "c"]
        assert bind(c).focus(lambda c: list(c.data.values())).get() == [1, 2, 3]
        assert bind(c).focus(lambda c: list(c.data.items())).get() == [
            ("a", 1),
            ("b", 2),
            ("c", 3),
        ]

        # Dict comprehension for reading with transformation
        result = bind(c).focus(lambda c: {k: v * 2 for k, v in c.data.items()}).get()
        assert result == {"a": 2, "b": 4, "c": 6}

        # SET via dict.values()
        result = bind(c).focus(lambda c: list(c.data.values())).set([100, 200, 300])
        assert result.data == {"a": 100, "b": 200, "c": 300}

        # SET via dict comprehension with explicit indexing
        result = (
            bind(c)
            .focus(lambda c: {k: c.data[k] for k in c.data})
            .set({"a": "x", "b": "y", "c": "z"})
        )
        assert result.data == {"a": "x", "b": "y", "c": "z"}

    def test_list_iteration_and_setting(self):
        """Test list iteration patterns for both GET and SET."""

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[10, 20, 30])

        # List iteration yields elements
        result = bind(c).focus(lambda c: [item for item in c.items]).get()
        assert result == [10, 20, 30]

        # SET via list comprehension
        result = (
            bind(c).focus(lambda c: [item for item in c.items]).set([100, 200, 300])
        )
        assert result.items == [100, 200, 300]

    def test_tuple_comprehension_patterns(self):
        """Test tuple comprehension patterns work correctly."""

        @dataclass
        class Container:
            data: tuple = field(static=True)

        c = Container(data=(1, 2, 3))

        # Tuple from comprehension
        result = bind(c).focus(lambda c: tuple(x for x in c.data)).get()
        assert result == (1, 2, 3)

    def test_nested_comprehension_patterns(self):
        """Test nested comprehension patterns."""

        @dataclass
        class Container:
            matrix: list = field(static=True)

        c = Container(matrix=[[1, 2], [3, 4], [5, 6]])

        # Nested list access
        result = bind(c).focus(lambda c: [[x for x in row] for row in c.matrix]).get()
        assert result == [[1, 2], [3, 4], [5, 6]]

        # Setting nested structure
        result = (
            bind(c)
            .focus(lambda c: [[x for x in row] for row in c.matrix])
            .set([[10, 20], [30, 40], [50, 60]])
        )
        assert result.matrix == [[10, 20], [30, 40], [50, 60]]

    def test_mixed_list_dict_comprehension(self):
        """Test comprehensions with mixed list and dict structures."""

        @dataclass
        class Container:
            data: dict = field(static=True)

        c = Container(data={"items": [1, 2, 3], "other": [4, 5, 6]})

        # Access nested list inside dict
        result = bind(c).focus(lambda c: [x for x in c.data["items"]]).get()
        assert result == [1, 2, 3]

        # Set nested list inside dict
        result = (
            bind(c).focus(lambda c: [x for x in c.data["items"]]).set([100, 200, 300])
        )
        assert result.data["items"] == [100, 200, 300]
        assert result.data["other"] == [4, 5, 6]  # Unchanged

    def test_len_support_in_traversal(self):
        """Test that len() works on traversed sequences."""

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[1, 2, 3, 4, 5])

        # Use len in focus
        result = bind(c).focus(lambda c: c.items[: len(c.items) // 2]).get()

        assert result == [1, 2]

    def test_contains_in_dict_traversal(self):
        """Test that 'in' operator works on dict traversals."""

        @dataclass
        class Container:
            data: dict = field(static=True)

        c = Container(data={"a": 1, "b": 2, "c": 3})

        # Direct membership check should work
        result = bind(c).focus(lambda c: "a" in c.data).get()
        assert result is True

        result = bind(c).focus(lambda c: "x" in c.data).get()
        assert result is False

    def test_contains_in_list_traversal(self):
        """Test that 'in' operator works on list traversals."""

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[1, 2, 3, 4, 5])

        # Direct membership check should work
        result = bind(c).focus(lambda c: 3 in c.items).get()
        assert result is True

        result = bind(c).focus(lambda c: 99 in c.items).get()
        assert result is False

    def test_contains_consistency_list_vs_dict(self):
        """Test that 'in' behaves consistently between list and dict traversals."""

        @dataclass
        class Container:
            items: list = field(static=True)
            data: dict = field(static=True)

        c = Container(items=[1, 2, 3], data={"a": 1, "b": 2})

        # Both should work with 'in' operator
        list_check = bind(c).focus(lambda c: 1 in c.items).get()
        dict_check = bind(c).focus(lambda c: "a" in c.data).get()

        assert list_check is True
        assert dict_check is True

        # Both should correctly report non-membership
        list_check = bind(c).focus(lambda c: 99 in c.items).get()
        dict_check = bind(c).focus(lambda c: "x" in c.data).get()

        assert list_check is False
        assert dict_check is False

    def test_contains_in_conditional_focus(self):
        """Test that 'in' operator works in conditional expressions."""

        @dataclass
        class Container:
            data: dict = field(static=True)
            items: list = field(static=True)

        c = Container(data={"key": 100}, items=[1, 2, 3])

        # Conditional based on dict membership
        result = bind(c).focus(lambda c: c.data["key"] if "key" in c.data else 0).get()
        assert result == 100

        # Conditional based on list membership
        result = bind(c).focus(lambda c: "found" if 2 in c.items else "not found").get()
        assert result == "found"


class TestBuiltinFunctionsInTraversal:
    """Tests for Python built-in functions used within lens traversals.

    Note: Tests use mutable containers which use shallow copy.
    """

    def test_max_min_sum_in_traversal(self):
        """Test max(), min(), sum() work in traversal.

        max() and min() are settable because they return the actual traversal
        path to the max/min element. sum() fails because it returns a computed
        scalar with no path.
        """

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[3, 1, 4, 1, 5, 9, 2, 6])

        # GET works for all
        assert bind(c).focus(lambda c: max(c.items)).get() == 9
        assert bind(c).focus(lambda c: min(c.items)).get() == 1
        assert bind(c).focus(lambda c: sum(c.items)).get() == 31

        # max() SET works - returns traversal to max element (index 5, value 9)
        result = bind(c).focus(lambda c: max(c.items)).set(100)
        assert result.items == [3, 1, 4, 1, 5, 100, 2, 6]

        # min() SET works - returns traversal to first min element (index 1, value 1)
        result = bind(c).focus(lambda c: min(c.items)).set(0)
        assert result.items == [3, 0, 4, 1, 5, 9, 2, 6]

        # sum() SET fails - returns computed scalar with no valid path
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: sum(c.items)).set(999)

    def test_sorted_in_traversal(self):
        """Test sorted() works in traversal including set().

        sorted() uses __lt__ for comparisons, which delegates to the current
        value. The traversals themselves are sorted and maintain their paths,
        so setting writes back to the original positions in sorted order.
        """

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[3, 1, 4, 1, 5])

        # Get sorted
        result = bind(c).focus(lambda c: sorted(c.items)).get()
        assert result == [1, 1, 3, 4, 5]

        result = bind(c).focus(lambda c: sorted(c.items, reverse=True)).get()
        assert result == [5, 4, 3, 1, 1]

        # Set sorted - values written back in sorted order positions
        # Original: [3, 1, 4, 1, 5] -> sorted indices: [1, 3, 0, 2, 4]
        # Set [10, 20, 30, 40, 50] to those positions
        result = bind(c).focus(lambda c: sorted(c.items)).set([10, 20, 30, 40, 50])
        assert result.items == [30, 10, 40, 20, 50]

    def test_sorted_with_key_in_traversal(self):
        """Test sorted() with key function works in traversal including set()."""

        @dataclass
        class Item:
            value: int

        @dataclass
        class Container:
            items: tuple

        c = Container(items=(Item(3), Item(1), Item(4), Item(1), Item(5)))

        # Get sorted by value
        result = (
            bind(c).focus(lambda c: tuple(sorted(c.items, key=lambda x: x.value))).get()
        )
        assert [i.value for i in result] == [1, 1, 3, 4, 5]

        # Set sorted - updates items at their sorted positions
        result = (
            bind(c)
            .focus(lambda c: tuple(sorted(c.items, key=lambda x: x.value)))
            .set((Item(10), Item(20), Item(30), Item(40), Item(50)))
        )
        # Original positions: [3,1,4,1,5] -> sorted gives indices [1,3,0,2,4]
        assert [i.value for i in result.items] == [30, 10, 40, 20, 50]

    def test_reversed_in_traversal(self):
        """Test reversed() works in traversal including set().

        reversed() uses __getitem__ in reverse order, so it yields traversals
        that maintain paths to the original indices - setting works!
        """

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[1, 2, 3, 4, 5])

        # Get works
        result = bind(c).focus(lambda c: [x for x in reversed(c.items)]).get()
        assert result == [5, 4, 3, 2, 1]

        # Set works - values are written back in reverse order
        result = (
            bind(c)
            .focus(lambda c: [x for x in reversed(c.items)])
            .set([50, 40, 30, 20, 10])
        )
        assert result.items == [10, 20, 30, 40, 50]

    def test_zip_enumerate_in_traversal(self):
        """Test zip() and enumerate() work in traversal including set().

        Both zip() and enumerate() yield traversals when iterating, so the
        values can be set back to the original structure.
        """

        @dataclass
        class Container:
            keys: list = field(static=True)
            values: list = field(static=True)

        c = Container(keys=["a", "b", "c"], values=[1, 2, 3])

        # Get with zip
        result = bind(c).focus(lambda c: list(zip(c.keys, c.values))).get()
        assert result == [("a", 1), ("b", 2), ("c", 3)]

        # Set via zip - setting the values from the second iterable
        result = (
            bind(c)
            .focus(lambda c: [v for k, v in zip(c.keys, c.values)])
            .set([100, 200, 300])
        )
        assert result.values == [100, 200, 300]
        assert result.keys == ["a", "b", "c"]  # Keys unchanged

        # Get with enumerate
        result = bind(c).focus(lambda c: list(enumerate(c.keys))).get()
        assert result == [(0, "a"), (1, "b"), (2, "c")]

        # Set via enumerate - setting the values
        result = (
            bind(c)
            .focus(lambda c: [v for i, v in enumerate(c.values)])
            .set([10, 20, 30])
        )
        assert result.values == [10, 20, 30]

    def test_any_all_in_traversal(self):
        """Test any() and all() work in traversal for GET but fail on SET."""

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[1, 2, 3, 4, 5])

        # GET works
        assert bind(c).focus(lambda c: any(x > 4 for x in c.items)).get() is True
        assert bind(c).focus(lambda c: any(x > 10 for x in c.items)).get() is False
        assert bind(c).focus(lambda c: all(x > 0 for x in c.items)).get() is True
        assert bind(c).focus(lambda c: all(x > 2 for x in c.items)).get() is False

        # SET fails - boolean predicate evaluation has no inverse
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: any(x > 4 for x in c.items)).set(False)
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: all(x > 0 for x in c.items)).set(False)

    def test_map_filter_in_traversal(self):
        """Test map() and filter() work in traversal.

        map() with transformation is GET-only (SET fails).
        filter() with comparison is settable (paths preserved).
        """

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[1, 2, 3, 4, 5])

        # Map with transformation - GET works
        result = bind(c).focus(lambda c: list(map(lambda x: x * 2, c.items))).get()
        assert result == [2, 4, 6, 8, 10]

        # Map with transformation - SET fails (computed values)
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: list(map(lambda x: x * 2, c.items))).set(
                [10, 20, 30, 40, 50]
            )

        # Filter with condition - NOW settable! Comparison operators delegate
        # to current value, so filtering works and paths are preserved
        result = bind(c).focus(lambda c: list(filter(lambda x: x > 2, c.items))).get()
        assert result == [3, 4, 5]

        # Set filtered items - only items matching filter are updated
        result = (
            bind(c)
            .focus(lambda c: list(filter(lambda x: x > 2, c.items)))
            .set([30, 40, 50])
        )
        assert result.items == [1, 2, 30, 40, 50]  # 1,2 unchanged; 3,4,5 -> 30,40,50

    def test_filter_with_comparison_is_settable(self):
        """Test that filter with comparison condition is settable.

        The comparison operators on _PathTraversal delegate to the current
        value for the comparison but preserve the traversal path. This enables
        filtering to select which items to update.
        """

        @dataclass
        class Item:
            name: str
            value: int

        @dataclass
        class Container:
            items: tuple

        c = Container(
            items=(
                Item("a", 1),
                Item("b", 10),
                Item("c", 2),
                Item("d", 20),
                Item("e", 3),
            )
        )

        # Filter items where value > 5
        result = (
            bind(c)
            .focus(lambda c: tuple(item for item in c.items if item.value > 5))
            .get()
        )
        assert [i.name for i in result] == ["b", "d"]

        # Set only items where value > 5 - others unchanged
        result = (
            bind(c)
            .focus(lambda c: tuple(item for item in c.items if item.value > 5))
            .set((Item("B", 100), Item("D", 200)))
        )
        assert [i.name for i in result.items] == ["a", "B", "c", "D", "e"]
        assert [i.value for i in result.items] == [1, 100, 2, 200, 3]

    def test_filter_with_multiple_conditions(self):
        """Test filter with multiple comparison conditions."""

        @dataclass
        class Container:
            items: tuple

        c = Container(items=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

        # Filter: 3 <= x <= 7
        result = (
            bind(c)
            .focus(lambda c: tuple(x for x in c.items if x >= 3 and x <= 7))
            .get()
        )
        assert result == (3, 4, 5, 6, 7)

        # Set those filtered items
        result = (
            bind(c)
            .focus(lambda c: tuple(x for x in c.items if x >= 3 and x <= 7))
            .set((30, 40, 50, 60, 70))
        )
        assert result.items == (1, 2, 30, 40, 50, 60, 70, 8, 9, 10)

    def test_filter_with_arithmetic_condition(self):
        """Test filter with arithmetic in condition - GET and SET."""

        @dataclass
        class Item:
            a: int
            b: int

        @dataclass
        class Container:
            items: tuple

        c = Container(
            items=(
                Item(1, 2),  # 1*2=2
                Item(3, 4),  # 3*4=12 > 10 ✓
                Item(2, 3),  # 2*3=6
                Item(5, 5),  # 5*5=25 > 10 ✓
                Item(1, 1),  # 1*1=1
            )
        )

        # GET: filter where a * b > 10
        result = (
            bind(c)
            .focus(lambda c: tuple(item for item in c.items if item.a * item.b > 10))
            .get()
        )
        assert [(i.a, i.b) for i in result] == [(3, 4), (5, 5)]

        # SET: only items matching condition are updated
        result = (
            bind(c)
            .focus(lambda c: tuple(item for item in c.items if item.a * item.b > 10))
            .set((Item(30, 40), Item(50, 50)))
        )
        assert [(i.a, i.b) for i in result.items] == [
            (1, 2),
            (30, 40),  # was (3,4)
            (2, 3),
            (50, 50),  # was (5,5)
            (1, 1),
        ]

    def test_filter_with_set_membership(self):
        """Test filter with set membership (x in {...}) - GET and SET."""

        @dataclass
        class Container:
            items: tuple

        c = Container(items=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

        # GET: filter where x in {2, 4, 6, 8}
        result = (
            bind(c)
            .focus(lambda c: tuple(x for x in c.items if x in {2, 4, 6, 8}))
            .get()
        )
        assert result == (2, 4, 6, 8)

        # SET: only items in set are updated
        result = (
            bind(c)
            .focus(lambda c: tuple(x for x in c.items if x in {2, 4, 6, 8}))
            .set((20, 40, 60, 80))
        )
        assert result.items == (1, 20, 3, 40, 5, 60, 7, 80, 9, 10)

    def test_identity_map_is_settable(self):
        """Test that identity map (map(lambda x: x, ...)) is settable.

        Unlike transformation maps, identity maps preserve traversals.
        """

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[1, 2, 3, 4, 5])

        # Identity map preserves traversals - settable!
        result = (
            bind(c)
            .focus(lambda c: list(map(lambda x: x, c.items)))
            .set([10, 20, 30, 40, 50])
        )
        assert result.items == [10, 20, 30, 40, 50]

    def test_always_true_filter_is_settable(self):
        """Test that filter with always-true condition is settable.

        When the condition doesn't evaluate traversals (e.g., always True),
        the traversal paths are preserved for setting.
        """

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[1, 2, 3, 4, 5])

        # Filter with always-true condition preserves traversals - settable!
        result = (
            bind(c)
            .focus(lambda c: list(filter(lambda x: True, c.items)))
            .set([10, 20, 30, 40, 50])
        )
        assert result.items == [10, 20, 30, 40, 50]


class TestIndexingAndSlicingInTraversal:
    """Tests for indexing and slicing within lens traversals.

    Note: Tests use mutable containers which use shallow copy.
    """

    def test_negative_indexing(self):
        """Test negative indexing works in traversal."""

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[10, 20, 30, 40, 50])

        # Get with negative index
        assert bind(c).focus(lambda c: c.items[-1]).get() == 50
        assert bind(c).focus(lambda c: c.items[-2]).get() == 40

        # Set with negative index
        result = bind(c).focus(lambda c: c.items[-1]).set(999)
        assert result.items == [10, 20, 30, 40, 999]

    def test_slice_get_and_set(self):
        """Test slicing works for both get and set in traversal."""

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[1, 2, 3, 4, 5])

        # Get with slice
        assert bind(c).focus(lambda c: c.items[1:4]).get() == [2, 3, 4]
        assert bind(c).focus(lambda c: c.items[:2]).get() == [1, 2]
        assert bind(c).focus(lambda c: c.items[3:]).get() == [4, 5]

        # Set with slice
        result = bind(c).focus(lambda c: c.items[1:4]).set([20, 30, 40])
        assert result.items == [1, 20, 30, 40, 5]

    def test_step_slice(self):
        """Test step slicing works in traversal including set()."""

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Get with step slices
        assert bind(c).focus(lambda c: c.items[::2]).get() == [0, 2, 4, 6, 8]
        assert bind(c).focus(lambda c: c.items[1::2]).get() == [1, 3, 5, 7, 9]
        assert bind(c).focus(lambda c: c.items[::-1]).get() == [
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
        ]

        # Set with step slice - sets every other element
        result = bind(c).focus(lambda c: c.items[::2]).set([100, 200, 300, 400, 500])
        assert result.items == [100, 1, 200, 3, 300, 5, 400, 7, 500, 9]

        # Set odd indices
        result = bind(c).focus(lambda c: c.items[1::2]).set([10, 30, 50, 70, 90])
        assert result.items == [0, 10, 2, 30, 4, 50, 6, 70, 8, 90]

    def test_tuple_slicing_with_apply(self):
        """Test slicing a tuple of arrays and applying a transformation."""
        x = (jnp.arange(3), jnp.arange(4), jnp.arange(5))
        slice_lens = lens(lambda x: x[:2])

        # Get returns the first two elements
        result = slice_lens.get(x)
        assert len(result) == 2
        npt.assert_array_equal(result[0], jnp.arange(3))
        npt.assert_array_equal(result[1], jnp.arange(4))

        # Apply transformation to the sliced portion
        new_x = slice_lens.apply(
            x, lambda sliced: jax.tree.map(lambda y: y * 10, sliced)
        )
        assert len(new_x) == 3
        npt.assert_array_equal(new_x[0], jnp.arange(3) * 10)
        npt.assert_array_equal(new_x[1], jnp.arange(4) * 10)
        npt.assert_array_equal(new_x[2], jnp.arange(5))  # Unchanged

    def test_deep_nested_dict_access(self):
        """Test deeply nested dictionary access and modification."""

        @dataclass
        class Container:
            nested: dict = field(static=True)

        c = Container(nested={"level1": {"level2": {"level3": {"value": 42}}}})

        # Get deeply nested value
        result = (
            bind(c)
            .focus(lambda c: c.nested["level1"]["level2"]["level3"]["value"])
            .get()
        )
        assert result == 42

        # Set deeply nested value
        result = (
            bind(c)
            .focus(lambda c: c.nested["level1"]["level2"]["level3"]["value"])
            .set(999)
        )
        assert result.nested["level1"]["level2"]["level3"]["value"] == 999


class TestStringOperationsInTraversal:
    """Tests for string operations within lens traversals.

    String methods and formatting produce computed values - GET works, SET fails.
    """

    def test_string_operations_get_only(self):
        """Test string operations work in traversal for GET but fail on SET."""

        @dataclass
        class Container:
            text: str = field(static=True)
            name: str = field(static=True)
            value: int = field(static=True)

        c = Container(text="hello world", name="test", value=42)

        # String methods - GET works
        assert bind(c).focus(lambda c: c.text.upper()).get() == "HELLO WORLD"
        assert bind(c).focus(lambda c: c.text.split()).get() == ["hello", "world"]
        assert bind(c).focus(lambda c: len(c.text)).get() == 11

        # F-string formatting - GET works
        result = bind(c).focus(lambda c: f"Name: {c.name}, Value: {c.value}").get()
        assert result == "Name: test, Value: 42"

        # SET fails for all computed string operations
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: c.text.upper()).set("NEW TEXT")
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: f"Name: {c.name}").set("ignored")


class TestComparisonAndBooleanOperators:
    """Tests for comparison and boolean operators in traversals.

    Comparison and boolean operators return computed booleans - GET works, SET fails.
    """

    def test_comparison_and_boolean_get_only(self):
        """Test comparison/boolean/conditional ops work for GET but fail on SET."""

        @dataclass
        class Container:
            value: int = field(static=True)
            a: int = field(static=True)
            b: int = field(static=True)
            maybe: object = field(static=True)

        c = Container(value=42, a=10, b=20, maybe=None)

        # Comparison operators - GET works
        assert bind(c).focus(lambda c: c.value == 42).get() is True
        assert bind(c).focus(lambda c: c.value < 50).get() is True

        # Boolean operators - GET works
        assert bind(c).focus(lambda c: c.a > 5 and c.b > 15).get() is True
        assert bind(c).focus(lambda c: not (c.a > 15)).get() is True

        # Identity comparisons - GET works
        assert bind(c).focus(lambda c: c.maybe is None).get() is True

        # Conditional expressions - GET works
        result = bind(c).focus(lambda c: "high" if c.value > 30 else "low").get()
        assert result == "high"

        # SET fails for all computed boolean operations
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: c.value == 42).set(False)
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: c.a > 5 and c.b > 15).set(False)
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: "high" if c.value > 30 else "low").set("medium")


class TestArithmeticOperatorsInTraversal:
    """Tests for arithmetic operators in traversals.

    Arithmetic operators delegate to the current value, enabling sort keys
    and computed expressions. Attempting to SET a computed result will fail
    with a clear error (no valid path).
    """

    def test_basic_arithmetic(self):
        """Test basic arithmetic operators work in traversal."""

        @dataclass
        class Container:
            value: int = field(static=True)

        c = Container(value=10)

        # All arithmetic ops should work for GET
        assert bind(c).focus(lambda c: c.value + 5).get() == 15
        assert bind(c).focus(lambda c: c.value - 3).get() == 7
        assert bind(c).focus(lambda c: c.value * 2).get() == 20
        assert bind(c).focus(lambda c: c.value / 4).get() == 2.5
        assert bind(c).focus(lambda c: c.value // 3).get() == 3
        assert bind(c).focus(lambda c: c.value % 3).get() == 1
        assert bind(c).focus(lambda c: c.value**2).get() == 100

    def test_reflected_arithmetic(self):
        """Test reflected arithmetic operators for GET but fail on SET."""

        @dataclass
        class Container:
            value: int = field(static=True)

        c = Container(value=5)

        # GET works - reflected ops (traversal on right side of operator)
        assert bind(c).focus(lambda c: 20 + c.value).get() == 25
        assert bind(c).focus(lambda c: 20 - c.value).get() == 15
        assert bind(c).focus(lambda c: 4 * c.value).get() == 20
        assert bind(c).focus(lambda c: 20 / c.value).get() == 4.0
        assert bind(c).focus(lambda c: 2**c.value).get() == 32

        # SET fails - reflected arithmetic produces computed values
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: 20 + c.value).set(100)
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: 2**c.value).set(64)

    def test_unary_operators(self):
        """Test unary operators for GET but fail on SET."""

        @dataclass
        class Container:
            value: int = field(static=True)

        c = Container(value=5)

        # GET works
        assert bind(c).focus(lambda c: -c.value).get() == -5
        assert bind(c).focus(lambda c: +c.value).get() == 5
        assert bind(c).focus(lambda c: abs(-c.value)).get() == 5

        # SET fails - unary operators produce computed values
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: -c.value).set(10)
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: abs(-c.value)).set(10)

    def test_sort_key_with_arithmetic(self):
        """Test that arithmetic in sort keys works for both GET and SET."""

        @dataclass
        class Item:
            val: int

        @dataclass
        class Container:
            items: tuple

        c = Container(items=(Item(3), Item(1), Item(4), Item(1), Item(5)))

        # Sort by negative value (descending order)
        result = (
            bind(c).focus(lambda c: tuple(sorted(c.items, key=lambda x: -x.val))).get()
        )
        assert [i.val for i in result] == [5, 4, 3, 1, 1]

        # SET works - arithmetic is just for sort key, not the target
        result = (
            bind(c)
            .focus(lambda c: tuple(sorted(c.items, key=lambda x: -x.val)))
            .set((Item(50), Item(40), Item(30), Item(20), Item(10)))
        )
        # Values set in descending-sorted positions
        assert [i.val for i in result.items] == [30, 20, 40, 10, 50]

    def test_sort_key_with_complex_expression(self):
        """Test sort with more complex key expressions."""

        @dataclass
        class Item:
            x: int
            y: int

        @dataclass
        class Container:
            items: tuple

        c = Container(items=(Item(1, 10), Item(2, 5), Item(3, 15)))

        # Sort by x * y (computed key)
        result = (
            bind(c)
            .focus(lambda c: tuple(sorted(c.items, key=lambda i: i.x * i.y)))
            .get()
        )
        # 1*10=10, 2*5=10, 3*15=45 -> sorted: [Item(1,10), Item(2,5), Item(3,15)]
        # Note: stable sort keeps original order for equal keys
        assert [(i.x, i.y) for i in result] == [(1, 10), (2, 5), (3, 15)]

    def test_bitwise_operators(self):
        """Test bitwise operators work in traversal for GET but fail on SET."""

        @dataclass
        class Container:
            value: int = field(static=True)

        c = Container(value=0b1010)  # 10

        # GET works
        assert bind(c).focus(lambda c: c.value & 0b1100).get() == 0b1000  # 8
        assert bind(c).focus(lambda c: c.value | 0b0101).get() == 0b1111  # 15
        assert bind(c).focus(lambda c: c.value ^ 0b1100).get() == 0b0110  # 6
        assert bind(c).focus(lambda c: c.value << 1).get() == 0b10100  # 20
        assert bind(c).focus(lambda c: c.value >> 1).get() == 0b0101  # 5

        # SET fails - bitwise operators produce computed values
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: c.value & 0b1100).set(0)
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: c.value << 1).set(40)


class TestTypeConversionsInTraversal:
    """Tests for type conversions within lens traversals.

    Type conversions produce new values - GET works, SET fails.

    Note: Tests use mutable containers which use shallow copy.
    """

    def test_type_conversions(self):
        """Test type conversions work in traversal.

        tuple() preserves paths and is settable. int()/str()/set() produce
        computed values and fail on SET.
        """

        @dataclass
        class Container:
            num_str: str = field(static=True)
            num: int = field(static=True)
            items: list = field(static=True)

        c = Container(num_str="123", num=42, items=[1, 2, 2, 3])

        # GET works for all
        assert bind(c).focus(lambda c: int(c.num_str)).get() == 123
        assert bind(c).focus(lambda c: str(c.num)).get() == "42"
        assert bind(c).focus(lambda c: tuple(c.items)).get() == (1, 2, 2, 3)
        assert bind(c).focus(lambda c: set(c.items)).get() == {1, 2, 3}

        # tuple() SET works - preserves paths
        result = bind(c).focus(lambda c: tuple(c.items)).set((9, 8, 7, 6))
        assert result.items == [9, 8, 7, 6]

        # int()/str() SET fails - scalar conversion loses path
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: int(c.num_str)).set(456)
        with pytest.raises(ValueError, match="computed value"):
            bind(c).focus(lambda c: str(c.num)).set("99")

    def test_bool_conversion(self):
        """Test bool() and truthiness work in traversal for GET but fail on SET."""

        @dataclass
        class Container:
            items: list = field(static=True)

        c_full = Container(items=[1, 2, 3])
        c_empty = Container(items=[])

        # GET works
        assert bind(c_full).focus(lambda c: bool(c.items)).get() is True
        assert bind(c_empty).focus(lambda c: bool(c.items)).get() is False

        # Truthiness in conditional
        assert bind(c_full).focus(lambda c: "yes" if c.items else "no").get() == "yes"
        assert bind(c_empty).focus(lambda c: "yes" if c.items else "no").get() == "no"

        # SET fails - bool conversion returns computed boolean
        with pytest.raises(ValueError, match="computed value"):
            bind(c_full).focus(lambda c: bool(c.items)).set(False)


class TestUnpackingInTraversal:
    """Tests for unpacking operators in lens traversals.

    Unpacking operators preserve traversal paths, making them settable.

    Note: Tests use mutable containers which use shallow copy.
    """

    def test_star_unpacking(self):
        """Test * unpacking works in traversal including set()."""

        @dataclass
        class Container:
            items: list = field(static=True)

        c = Container(items=[1, 2, 3])

        # Get works
        result = bind(c).focus(lambda c: [*c.items]).get()
        assert result == [1, 2, 3]

        result = bind(c).focus(lambda c: [0, *c.items, 4]).get()
        assert result == [0, 1, 2, 3, 4]

        # Set works - unpacking preserves traversals
        result = bind(c).focus(lambda c: [*c.items]).set([10, 20, 30])
        assert result.items == [10, 20, 30]

    def test_double_star_unpacking(self):
        """Test ** unpacking works in traversal including set()."""

        @dataclass
        class Container:
            data: dict = field(static=True)

        c = Container(data={"a": 1, "b": 2})

        # Get works
        result = bind(c).focus(lambda c: {**c.data}).get()
        assert result == {"a": 1, "b": 2}

        result = bind(c).focus(lambda c: {**c.data, "c": 3}).get()
        assert result == {"a": 1, "b": 2, "c": 3}

        # Set works - unpacking preserves traversals
        result = bind(c).focus(lambda c: {**c.data}).set({"a": 100, "b": 200})
        assert result.data == {"a": 100, "b": 200}


class TestGetAttrHasAttrInTraversal:
    """Tests for getattr() and hasattr() within lens traversals."""

    def test_getattr_with_default(self):
        """Test getattr() with default value works in traversal including set().

        getattr() returns a traversal to the attribute, making it settable.
        """

        @dataclass
        class Container:
            value: int = field(static=True)

        c = Container(value=42)

        # Get works
        result = bind(c).focus(lambda c: getattr(c, "value", 0)).get()
        assert result == 42

        result = bind(c).focus(lambda c: getattr(c, "nonexistent", 999)).get()
        assert result == 999

        # Set works for existing attributes
        result = bind(c).focus(lambda c: getattr(c, "value", 0)).set(100)
        assert result.value == 100

    def test_hasattr_check(self):
        """Test hasattr() works in traversal.

        Note: set() not tested - hasattr() returns a boolean indicating
        attribute existence, not a reference to the attribute.
        """

        @dataclass
        class Container:
            value: int = field(static=True)

        c = Container(value=42)

        assert bind(c).focus(lambda c: hasattr(c, "value")).get() is True
        assert bind(c).focus(lambda c: hasattr(c, "nonexistent")).get() is False

    def test_safe_navigation_pattern(self):
        """Test safe navigation pattern (checking before accessing).

        Note: set() not tested - conditional expressions make the path
        dependent on runtime evaluation, which is non-deterministic for setting.
        """

        @dataclass
        class Container:
            maybe: object = field(static=True)

        c_with_value = Container(maybe={"key": 42})
        c_with_none = Container(maybe=None)

        result = (
            bind(c_with_value)
            .focus(lambda c: c.maybe["key"] if c.maybe is not None else "default")
            .get()
        )
        assert result == 42

        result = (
            bind(c_with_none)
            .focus(lambda c: c.maybe["key"] if c.maybe is not None else "default")
            .get()
        )
        assert result == "default"


class TestSimpleLensFallback:
    """Tests for SimpleLens fallback to traversal lens mechanism."""

    def test_simplelens_fallback_to_traversal(self):
        """Test that SimpleLens uses traversal lens for setting values."""

        @dataclass
        class CustomClass:
            value: int

        obj = CustomClass(value=42)

        # Create a lens that might trigger fallback
        value_lens = lens(lambda o: o.value)

        # Get should work
        assert value_lens.get(obj) == 42

        # Set should work (via traversal lens fallback)
        new_obj = value_lens.set(obj, 100)
        assert new_obj.value == 100
        assert obj.value == 42  # Original unchanged

    def test_error_message_for_invalid_lens(self):
        """Test that ValueError is raised with clear message for invalid lenses."""

        @dataclass
        class TestObj:
            x: int

        def bad_where(obj: TestObj) -> int:
            # This modifies the object - invalid for a lens
            object.__setattr__(obj, "x", 999)
            return obj.x

        bad_lens = lens(bad_where)
        obj = TestObj(x=10)

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError) as exc_info:
            bad_lens.set(obj, 20)

        error_msg = str(exc_info.value)
        assert "Cannot set value" in error_msg

    def test_successful_fallback_scenario(self):
        """Test a scenario where fallback to traversal lens succeeds."""

        @dataclass
        class Point:
            x: float
            y: float

        point = Point(x=1.0, y=2.0)

        # Simple field access should work via either path
        x_lens = lens(lambda p: p.x)
        y_lens = lens(lambda p: p.y)

        # Both should work
        assert x_lens.get(point) == 1.0
        assert y_lens.get(point) == 2.0

        new_point = x_lens.set(point, 10.0)
        assert new_point.x == 10.0
        assert new_point.y == 2.0
        assert point.x == 1.0  # Original unchanged

    def test_incorrect_assignment(self):
        # Empty list - returns original dict unchanged
        original = {"key": "value"}
        with pytest.raises(ValueError, match="Expected list, got 'ignored'."):
            bind(original).focus(lambda d: []).set("ignored")
        with pytest.raises(ValueError, match="Expected tuple, got 'ignored'."):
            bind(original).focus(lambda d: ()).set("ignored")
        with pytest.raises(ValueError, match="Expected dict, got 'ignored'."):
            bind(original).focus(lambda d: {}).set("ignored")


class TestItemLens:
    """Tests for _ItemLens behavior with various collection types."""

    def test_item_lens_mutable_collections(self):
        """Test _ItemLens with mutable dict and list (shallow copy behavior)."""

        @dataclass
        class Container:
            data: dict = field(static=True)
            items: list = field(static=True)

        c = Container(data={"a": 1, "b": 2}, items=[10, 20, 30])

        # Dict access and set - uses shallow copy
        result = bind(c).focus(lambda c: c.data["b"]).set(200)
        assert result.data == {"a": 1, "b": 200}
        assert c.data == {"a": 1, "b": 2}  # Original unchanged

        # List access and set - uses shallow copy
        result = bind(c).focus(lambda c: c.items[1]).set(200)
        assert result.items == [10, 200, 30]
        assert c.items == [10, 20, 30]  # Original unchanged

    def test_item_lens_with_immutable_sequence_tuple(self):
        """Test _ItemLens works correctly with tuples (immutable sequences)."""

        @dataclass
        class Container:
            items: tuple = field(static=True)

        c = Container(items=(1, 2, 3, 4, 5))

        # Set through tuple - should NOT warn, and should work correctly
        result = bind(c).focus(lambda c: c.items[2]).set(30)

        assert result.items == (1, 2, 30, 4, 5)
        assert isinstance(result.items, tuple)
        # Original should be unchanged
        assert c.items == (1, 2, 3, 4, 5)

    def test_item_lens_with_immutable_mapping(self):
        """Test _ItemLens works correctly with immutable mappings."""
        from types import MappingProxyType

        @dataclass
        class Container:
            data: MappingProxyType = field(static=True)

        original_dict = {"x": 100, "y": 200, "z": 300}
        c = Container(data=MappingProxyType(original_dict))

        # Set through MappingProxyType - immutable mapping
        result = bind(c).focus(lambda c: c.data["y"]).set(999)

        assert result.data["y"] == 999
        assert result.data["x"] == 100
        assert result.data["z"] == 300
        # Original should be unchanged
        assert c.data["y"] == 200

    def test_item_lens_preserves_sequence_type(self):
        """Test that _ItemLens preserves the original sequence type."""

        @dataclass
        class Container:
            items: tuple = field(static=True)

        # Test with tuple
        c_tuple = Container(items=(1, 2, 3))
        result_tuple = bind(c_tuple).focus(lambda c: c.items[0]).set(10)
        assert isinstance(result_tuple.items, tuple)
        assert result_tuple.items == (10, 2, 3)

    def test_item_lens_multiple_updates(self):
        """Test multiple updates through _ItemLens."""

        @dataclass
        class Container:
            data: tuple = field(static=True)

        c = Container(data=(1, 2, 3, 4, 5))

        # Chain multiple updates
        c = bind(c).focus(lambda c: c.data[0]).set(10)
        c = bind(c).focus(lambda c: c.data[2]).set(30)
        c = bind(c).focus(lambda c: c.data[4]).set(50)

        assert c.data == (10, 2, 30, 4, 50)


class TestLensProperty:
    """Tests for the lens_property decorator."""

    def test_class_access_returns_lens(self):
        """Test that accessing lens_property on the class returns a lens."""

        @dataclass
        class A(HasLensFields):
            _a: int

            @lens_property
            def a(self) -> int:
                return self._a

        # Accessing on the class should return a lens
        a_lens = A.a
        assert isinstance(a_lens, SimpleLens)

    def test_instance_access_returns_value(self):
        """Test that accessing lens_property on an instance returns the value."""

        @dataclass
        class A(HasLensFields):
            _a: int

            @lens_property
            def a(self) -> int:
                return self._a

        obj = A(_a=5)
        assert obj.a == 5

    def test_lens_get_and_set(self):
        """Test that the lens from class access can get and set values."""

        @dataclass
        class A(HasLensFields):
            _a: int

            @lens_property
            def a(self) -> int:
                return self._a

        obj = A(_a=10)

        # Get via lens
        a_lens = A.a
        assert a_lens.get(obj) == 10

        # Set via lens
        new_obj = a_lens.set(obj, 20)
        assert new_obj._a == 20
        assert obj._a == 10  # Original unchanged

    def test_lens_property_preserves_metadata(self):
        """Test that lens_property preserves function metadata."""

        @dataclass
        class A(HasLensFields):
            _value: int

            @lens_property
            def value(self) -> int:
                """This is a docstring."""
                return self._value

        prop = A.__dict__["value"]
        assert prop.__doc__ == "This is a docstring."
        assert prop.__name__ == "value"

    def test_lens_property_descriptor_protocol(self):
        """Test that lens_property implements descriptor protocol correctly."""

        @dataclass
        class A(HasLensFields):
            _value: int

            @lens_property
            def value(self) -> int:
                return self._value

        # Check that the class attribute is a lens_property
        assert isinstance(A.__dict__["value"], lens_property)

        # Check that __set_name__ was called
        prop = A.__dict__["value"]
        assert prop._name == "value"


class TestCallableHandlingInTraversal:
    """Comprehensive tests for callable handling in _PathTraversal.

    These tests cover various callable types that can appear in lens focus functions:
    - Instance methods (bound methods)
    - Class methods (@classmethod)
    - Static methods (@staticmethod)
    - Regular functions
    - Callable objects (instances with __call__ method)
    - Lambda functions
    """

    def test_instance_method_returns_field_reference(self):
        """Test instance method that returns a reference to self fields."""

        @dataclass
        class Inner:
            value: jax.Array

        @dataclass
        class Outer:
            x: jax.Array
            y: jax.Array

            def as_inner(self) -> Inner:
                return Inner(self.x)

        obj = Outer(x=jnp.array([1, 2, 3]), y=jnp.array([4, 5, 6]))
        new_inner = Inner(value=jnp.array([10, 20, 30]))

        # Get through method call
        result_get = bind(obj).focus(lambda o: o.as_inner()).get()
        assert jnp.array_equal(result_get.value, obj.x)

        # Set through method call
        result_set = bind(obj).focus(lambda o: o.as_inner()).set(new_inner)
        assert jnp.array_equal(result_set.x, jnp.array([10, 20, 30]))
        assert jnp.array_equal(result_set.y, obj.y)  # Unchanged

    def test_classmethod_in_focus(self):
        """Test @classmethod in lens focus - should be called directly without lens traversal."""

        @dataclass
        class Container:
            value: int = 42

            @classmethod
            def create_default(cls) -> "Container":
                return cls(value=100)

        obj = Container(value=1)

        # Accessing classmethod - should just call it and return new object
        # Note: This doesn't do lens traversal since classmethod doesn't reference self
        result = bind(obj).focus(lambda c: c.create_default()).get()
        assert result.value == 100

    def test_staticmethod_in_focus(self):
        """Test @staticmethod in lens focus - should be called directly."""

        @dataclass
        class Container:
            value: int

            @staticmethod
            def double(x: int) -> int:
                return x * 2

        obj = Container(value=5)

        # Static method should be called directly
        result = bind(obj).focus(lambda c: c.double(c.value)).get()
        assert result == 10

    def test_regular_function_in_focus(self):
        """Test regular function (not method) in lens focus."""

        def transform(x: jax.Array) -> jax.Array:
            return x * 2

        @dataclass
        class Container:
            data: jax.Array

        obj = Container(data=jnp.array([1, 2, 3]))

        # Function called in focus
        result = bind(obj).focus(lambda c: transform(c.data)).get()
        assert jnp.array_equal(result, jnp.array([2, 4, 6]))

    def test_callable_object_with_call_method(self):
        """Test callable object (instance with __call__ method) in lens focus.

        This is the key edge case from PR review - hasattr(obj, '__call__')
        doesn't mean obj has __func__ directly; we need to check __call__.__func__.
        """

        class Transformer:
            """A callable object that transforms data."""

            def __init__(self, multiplier: int):
                self.multiplier = multiplier

            def __call__(self, x: jax.Array) -> jax.Array:
                return x * self.multiplier

        @dataclass
        class Container:
            data: jax.Array

        obj = Container(data=jnp.array([1, 2, 3]))
        transformer = Transformer(multiplier=3)

        # Callable object should work in focus
        result = bind(obj).focus(lambda c: transformer(c.data)).get()
        assert jnp.array_equal(result, jnp.array([3, 6, 9]))

    def test_callable_object_without_lens_traversal(self):
        """Test callable object that doesn't need lens traversal support."""

        class Adder:
            def __init__(self, amount: int):
                self.amount = amount

            def __call__(self, x: int) -> int:
                return x + self.amount

        @dataclass
        class Container:
            value: int = field(static=True)

        obj = Container(value=10)
        adder = Adder(5)

        # Callable should work
        result = bind(obj).focus(lambda c: adder(c.value)).get()
        assert result == 15

    def test_lambda_function_in_focus(self):
        """Test lambda function in lens focus."""

        @dataclass
        class Container:
            data: jax.Array

        obj = Container(data=jnp.array([1, 2, 3]))

        # Lambda in focus
        result = bind(obj).focus(lambda c: (lambda x: x + 1)(c.data)).get()
        assert jnp.array_equal(result, jnp.array([2, 3, 4]))

    def test_method_with_arguments(self):
        """Test instance method with additional arguments."""

        @dataclass
        class Container:
            data: jax.Array

            def scale_and_shift(self, scale: float, shift: float) -> jax.Array:
                return self.data * scale + shift

        obj = Container(data=jnp.array([1.0, 2.0, 3.0]))

        # Method call with arguments
        result = bind(obj).focus(lambda c: c.scale_and_shift(2.0, 1.0)).get()
        assert jnp.array_equal(result, jnp.array([3.0, 5.0, 7.0]))

    def test_chained_method_calls(self):
        """Test chained method calls in lens focus."""

        @dataclass
        class Inner:
            value: jax.Array

            def doubled(self) -> "Inner":
                return Inner(self.value * 2)

        @dataclass
        class Outer:
            inner: Inner

            def get_inner(self) -> Inner:
                return self.inner

        inner = Inner(value=jnp.array([1, 2, 3]))
        outer = Outer(inner=inner)

        # Chained method calls
        result = bind(outer).focus(lambda o: o.get_inner().doubled()).get()
        assert jnp.array_equal(result.value, jnp.array([2, 4, 6]))

    def test_callable_class_instance_with_state(self):
        """Test callable object that maintains internal state."""

        class Counter:
            def __init__(self):
                self.count = 0

            def __call__(self, x: jax.Array) -> jax.Array:
                # Note: In JAX context, we shouldn't mutate state in JIT
                # This is just testing that the callable works
                return x + 1

        @dataclass
        class Container:
            data: jax.Array

        obj = Container(data=jnp.array([10, 20, 30]))
        counter = Counter()

        result = bind(obj).focus(lambda c: counter(c.data)).get()
        assert jnp.array_equal(result, jnp.array([11, 21, 31]))

    def test_nested_callable_in_dataclass(self):
        """Test callable stored as a field in dataclass."""

        class Processor:
            def __call__(self, x: jax.Array) -> jax.Array:
                return x**2

        @dataclass
        class Container:
            data: jax.Array
            process: Processor = field(static=True)

        processor = Processor()
        obj = Container(data=jnp.array([2, 3, 4]), process=processor)

        # Access callable field and call it
        result = bind(obj).focus(lambda c: c.process(c.data)).get()
        assert jnp.array_equal(result, jnp.array([4, 9, 16]))

    def test_builtin_callable_in_focus(self):
        """Test builtin callable (like len) in lens focus."""

        @dataclass
        class Container:
            items: list = field(static=True)

        obj = Container(items=[1, 2, 3, 4, 5])

        # len() is a builtin callable
        result = bind(obj).focus(lambda c: len(c.items)).get()
        assert result == 5

    def test_partial_function_in_focus(self):
        """Test functools.partial in lens focus."""
        from functools import partial

        def multiply(x: jax.Array, factor: float) -> jax.Array:
            return x * factor

        @dataclass
        class Container:
            data: jax.Array

        obj = Container(data=jnp.array([1.0, 2.0, 3.0]))
        triple = partial(multiply, factor=3.0)

        result = bind(obj).focus(lambda c: triple(c.data)).get()
        assert jnp.array_equal(result, jnp.array([3.0, 6.0, 9.0]))


class TestNamedTupleSupport:
    """Tests for NamedTuple compatibility with the lens system."""

    def test_namedtuple_basic_get_set(self):
        """Test basic get/set on NamedTuple fields."""
        from typing import NamedTuple

        class Point(NamedTuple):
            x: float
            y: float

        p = Point(x=1.0, y=2.0)

        # Get
        result = bind(p).focus(lambda p: p.x).get()
        assert result == 1.0

        # Set
        new_p = bind(p).focus(lambda p: p.x).set(10.0)
        assert new_p.x == 10.0
        assert new_p.y == 2.0
        assert p.x == 1.0  # Original unchanged

    def test_namedtuple_nested_in_dataclass(self):
        """Test NamedTuple nested inside a dataclass."""
        from typing import NamedTuple

        class Point(NamedTuple):
            x: float
            y: float

        @dataclass
        class Container:
            point: Point
            label: str

        c = Container(point=Point(x=1.0, y=2.0), label="origin")

        # Get nested
        result = bind(c).focus(lambda c: c.point.x).get()
        assert result == 1.0

        # Set nested
        new_c = bind(c).focus(lambda c: c.point.x).set(10.0)
        assert new_c.point.x == 10.0
        assert new_c.point.y == 2.0
        assert new_c.label == "origin"

    def test_dataclass_nested_in_namedtuple(self):
        """Test dataclass nested inside a NamedTuple."""
        from typing import NamedTuple

        @dataclass
        class Metadata:
            name: str
            version: int

        class Record(NamedTuple):
            data: jax.Array
            meta: Metadata

        r = Record(data=jnp.array([1, 2, 3]), meta=Metadata(name="test", version=1))

        # Get nested dataclass field
        result = bind(r).focus(lambda r: r.meta.name).get()
        assert result == "test"

        # Set nested dataclass field
        new_r = bind(r).focus(lambda r: r.meta.name).set("updated")
        assert new_r.meta.name == "updated"
        assert new_r.meta.version == 1
        assert jnp.array_equal(new_r.data, jnp.array([1, 2, 3]))

    def test_namedtuple_with_array(self):
        """Test NamedTuple containing JAX arrays."""
        from typing import NamedTuple

        class State(NamedTuple):
            positions: jax.Array
            velocities: jax.Array

        s = State(
            positions=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            velocities=jnp.array([[0.1, 0.2], [0.3, 0.4]]),
        )

        # Set array field
        new_pos = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        new_s = bind(s).focus(lambda s: s.positions).set(new_pos)

        assert jnp.array_equal(new_s.positions, new_pos)
        assert jnp.array_equal(new_s.velocities, s.velocities)

    def test_namedtuple_tuple_of_namedtuples(self):
        """Test tuple containing multiple NamedTuples."""
        from typing import NamedTuple

        class Point(NamedTuple):
            x: float
            y: float

        @dataclass
        class Shape:
            points: tuple

        s = Shape(points=(Point(0, 0), Point(1, 1), Point(2, 2)))

        # Get through comprehension
        xs = bind(s).focus(lambda s: tuple(p.x for p in s.points)).get()
        assert xs == (0, 1, 2)

        # Set through comprehension
        new_s = bind(s).focus(lambda s: tuple(p.x for p in s.points)).set((10, 20, 30))
        assert new_s.points[0].x == 10
        assert new_s.points[1].x == 20
        assert new_s.points[2].x == 30
        # y values unchanged
        assert new_s.points[0].y == 0
        assert new_s.points[1].y == 1


class TestIdentityLens:
    """Tests for identity lens (focusing on root object)."""

    def test_identity_lens_set_dataclass(self):
        """Test that identity lens can replace entire dataclass."""

        @dataclass
        class Simple:
            value: int

        original = Simple(value=1)
        replacement = Simple(value=42)

        # Focus on root and replace
        result = bind(original).focus(lambda x: x).set(replacement)

        assert result.value == 42
        assert original.value == 1  # Original unchanged

    def test_identity_lens_set_nested(self):
        """Test identity lens on nested field replaces that field entirely."""

        @dataclass
        class Inner:
            x: int
            y: int

        @dataclass
        class Outer:
            inner: Inner
            label: str

        original = Outer(inner=Inner(x=1, y=2), label="test")
        new_inner = Inner(x=10, y=20)

        # Replace inner entirely
        result = bind(original).focus(lambda o: o.inner).set(new_inner)

        assert result.inner.x == 10
        assert result.inner.y == 20
        assert result.label == "test"


class TestDuplicateReferences:
    """Tests for handling duplicate object references in tree."""

    def test_shared_array_reference(self):
        """Test behavior when same array appears twice in tree."""
        shared = jnp.array([1, 2, 3])

        @dataclass
        class Container:
            a: jax.Array
            b: jax.Array

        # Both fields reference same array
        c = Container(a=shared, b=shared)

        # Setting through 'a' should only update 'a'
        new_c = bind(c).focus(lambda c: c.a).set(jnp.array([10, 20, 30]))

        assert jnp.array_equal(new_c.a, jnp.array([10, 20, 30]))
        # b should remain unchanged (original shared value)
        assert jnp.array_equal(new_c.b, jnp.array([1, 2, 3]))

    def test_shared_object_in_tuple(self):
        """Test shared object appearing multiple times in tuple."""

        @dataclass
        class Item:
            value: int

        shared = Item(value=1)

        @dataclass
        class Container:
            items: tuple

        # Same item appears twice
        c = Container(items=(shared, shared, Item(value=2)))

        # Focus on first item's value
        result = bind(c).focus(lambda c: c.items[0].value).set(100)

        # First item updated
        assert result.items[0].value == 100
        # Second item (same original object) should NOT be updated
        # because we're setting by path, not by object identity
        assert result.items[1].value == 1
        assert result.items[2].value == 2


class TestDeepNesting:
    """Stress tests for deeply nested structures."""

    def test_deep_nesting_10_levels(self):
        """Test lens operations on 10 levels of nesting."""

        @dataclass
        class Node:
            value: int
            child: Any = None

        # Build 10 levels deep
        node = Node(value=10)
        for i in range(9, 0, -1):
            node = Node(value=i, child=node)

        # Access deepest value
        result = (
            bind(node)
            .focus(
                lambda n: n.child.child.child.child.child.child.child.child.child.value
            )
            .get()
        )
        assert result == 10

        # Set deepest value
        new_node = (
            bind(node)
            .focus(
                lambda n: n.child.child.child.child.child.child.child.child.child.value
            )
            .set(999)
        )
        assert (
            new_node.child.child.child.child.child.child.child.child.child.value == 999
        )
        assert node.child.child.child.child.child.child.child.child.child.value == 10

    def test_deep_mixed_structure(self):
        """Test deep nesting with mixed types (dataclass, dict, tuple)."""

        @dataclass
        class Level1:
            data: dict = field(static=True)

        @dataclass
        class Level2:
            items: tuple

        @dataclass
        class Level3:
            value: int

        structure = Level1(
            data={
                "nested": Level2(
                    items=(Level3(value=1), Level3(value=2), Level3(value=3))
                )
            }
        )

        # Access through mixed types
        result = bind(structure).focus(lambda s: s.data["nested"].items[1].value).get()
        assert result == 2

        # Set through mixed types
        new_structure = (
            bind(structure).focus(lambda s: s.data["nested"].items[1].value).set(200)
        )
        assert new_structure.data["nested"].items[1].value == 200
        assert structure.data["nested"].items[1].value == 2


class TestPropertyAlias:
    """Tests for setting through property aliases."""

    def test_property_alias_set(self):
        """Setting via a property that aliases another field updates the underlying field."""

        @dataclass
        class A:
            a: Array

            @property
            def b(self) -> Array:
                return self.a

        state = A(a=jnp.arange(2))
        result = lens(lambda x: x.b).set(state, jnp.arange(5))

        npt.assert_array_equal(result.a, jnp.arange(5))
        npt.assert_array_equal(result.b, jnp.arange(5))

    def test_property_alias_nested(self):
        """Property alias that dereferences a nested field."""

        @dataclass
        class Inner:
            x: Array

        @dataclass
        class Outer:
            inner: Inner

            @property
            def x(self) -> Array:
                return self.inner.x

        state = Outer(inner=Inner(x=jnp.array([1, 2, 3])))
        result = lens(lambda o: o.x).set(state, jnp.array([9, 8, 7]))

        npt.assert_array_equal(result.inner.x, jnp.array([9, 8, 7]))
        npt.assert_array_equal(result.x, jnp.array([9, 8, 7]))
