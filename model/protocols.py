# -*- coding: utf-8 -*-
"""Collection of protocols in the project."""

from typing import Any, Protocol, Self

import pandas as pd


class Agent(Protocol):
    """Protocol for an agent used in this model simulation."""

    def step(self: Self, **env_data: dict[str, Any]) -> None:
        """Take action for the current timestep."""

    def update(self: Self) -> None:
        """Update itself to prepare for the next timestep."""


class SpatialObject(Protocol):
    """Protocol for any spatial object in this model."""

    spatial_id: int


class Scenario(Protocol):
    """Protocol for scenarios that provide contextual information for models."""

    def __set__item__(self: Self, key: str, df: pd.DataFrame) -> None:
        """Set some data for this scenario."""

    def __get_item__(self: Self, key: str) -> pd.DataFrame:
        """Fetch data in this scenario, if it exists."""

    def __next__(self: Self) -> dict[str, pd.DataFrame]:
        """Provide data for the next timestep in this scenario."""

    def keys(self: Self) -> list[str]:
        """List of all data keys in this scenario."""

    def update(self: Self, timestep: int) -> None:
        """Update the scenario for the specified timestep."""


class Submodel(Protocol):
    """Protocol for submodels that will be added to the model."""

    def step(self: Self) -> None:
        """Process the next step of this submodel."""
