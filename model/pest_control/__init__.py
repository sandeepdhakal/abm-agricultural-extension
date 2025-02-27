"""Pest control methods."""

from .control_methods import (
    BioControl,
    NoControl,
    PestControl,
)

NO_CONTROL = NoControl()

__all__ = [
    BioControl,
    NoControl,
    NO_CONTROL,
    PestControl,
]
