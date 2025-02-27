# -*- coding: utf-8 -*-
"""Module with different pest control methods.

Every pest control model needs to provide a `current_efficacy` method with the following
signature:
```current_efficacy(num_days: int, pests: str | list[str]) -> float```

Classes have been created for different types of control methods: no control, seed
treatment and insecticide.
"""

from __future__ import annotations

from typing import Self


class PestControl:
    """A base model for pest control methods."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
    ) -> None:
        """Init a PestControl instance.

        Args:
            name: the name of the control method.
        """
        self.name: str = name
        """The name of the control method."""


class NoControl(PestControl):
    """A control method that does nothing."""

    def __init__(self: Self) -> Self:
        """Init NoControl."""
        super().__init__("No control")


class BioControl(PestControl):
    """A model of a biocontrol method."""

    def __init__(
        self,
        name: str,
    ) -> None:
        """Init a biocontrol model."""
        super().__init__(name)
