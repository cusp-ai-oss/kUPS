# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Field handler system for customizing dataclass field behavior.

This module provides a metaclass-based system for customizing how dataclass fields
are accessed at the class level. It's used by LensField to provide lens access when
accessing fields as class attributes.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, get_origin, get_type_hints


class FieldHandler(Protocol):
    """Protocol for field handler functions in the lens field system.

    A field handler is called when accessing a class attribute that corresponds
    to a dataclass field with a registered field type (e.g., LensField).

    When called with (cls, name), returns the computed value for the field access
    (e.g., a Lens object).
    """

    def __call__(self, cls, name) -> Any: ...


_FIELD_HANDLERS: dict[Any, FieldHandler] = {}


def register_field_handler(field_type: type) -> Callable[[FieldHandler], FieldHandler]:
    """Register a handler function for a specific field type.

    This decorator registers a FieldHandler that will be invoked when accessing
    class attributes annotated with the specified field type on classes that use
    the FieldMetaAccess metaclass.

    Args:
        field_type: The type (e.g., LensField) to register a handler for

    Returns:
        A decorator that registers the handler function.

    Example:
        >>> from kups.core.lens import LensField, lens
        >>> @register_field_handler(LensField)
        ... def handle_lens_field(cls, name):
        ...     return lens(lambda obj: getattr(obj, name))
    """

    def decorator(func: FieldHandler) -> FieldHandler:
        _FIELD_HANDLERS[field_type] = func
        return func

    return decorator


class FieldMetaAccess(type):
    """Metaclass that intercepts class attribute access for special field types.

    This metaclass enables custom behavior when accessing class attributes that
    correspond to dataclass fields with registered field types (e.g., LensField).

    When accessing a class attribute:
    1. Checks if the class has __dataclass_fields__ (is a dataclass)
    2. If the attribute name matches a field with a registered type, invokes
       the corresponding field handler
    3. Otherwise, uses normal attribute access

    Example:
        >>> from kups.core.utils.jax import dataclass
        >>> from kups.core.lens import LensField, HasLensFields
        >>> @dataclass
        ... class Point(HasLensFields):
        ...     x: LensField[float]
        ...
        >>> Point.x  # Returns a Lens[Point, float] via the LensField handler
        >>> point = Point(x=5.0)
        >>> point.x  # Returns 5.0 (normal instance access)
    """

    def __getattribute__(cls, name):
        try:
            dataclass_fields = object.__getattribute__(cls, "__dataclass_fields__")
        except AttributeError:
            return super().__getattribute__(name)
        if name in dataclass_fields:
            # Use get_type_hints with proper namespace resolution for forward references.
            # Get the module's global namespace and add the class itself to localns
            # to handle self-referential forward references
            localns = {cls.__name__: cls}
            try:
                type_hints = get_type_hints(cls, localns=localns)
                field_type = get_origin(type_hints[name])
            except NameError:
                # Unresolvable type parameters from generic base classes
                # (e.g., Python 3.12 PEP 695 generics). Try to get origin from
                # the raw annotation without full resolution.
                ann = dataclass_fields[name].type
                if isinstance(ann, str):
                    # String annotation - check by name prefix matching
                    for registered_type in _FIELD_HANDLERS:
                        if ann.startswith(registered_type.__name__ + "["):
                            handler = _FIELD_HANDLERS[registered_type]
                            return handler(cls, name)
                    return super().__getattribute__(name)
                field_type = get_origin(ann)
            if field_type in _FIELD_HANDLERS:
                handler = _FIELD_HANDLERS[field_type]
                return handler(cls, name)
        return super().__getattribute__(name)
