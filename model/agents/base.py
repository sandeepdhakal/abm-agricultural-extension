# -*- coding: utf-8 -*-
"""Module with base agent class."""

from typing import Any, Self

import shortuuid


class NotASpatialObjectError(Exception):
    """Raised when a spatial agent is expected, but a non-spatial agent is used."""

    def __init__(self: Self, message: str = "The agent is not spatial.") -> None:
        """Init NotASpatialObjectError with `message`."""
        super().__init__(message)


class NotAssignedToAnyDigitalTwinError(Exception):
    """Raised when an agent action requires a ditital twin but it its missing."""

    def __init__(
        self: Self,
        message: str = "The agent does not belong to any digital twin yet.",
    ) -> None:
        """Init NotAssignedToAnyDigitalTwinError with `message`."""
        super().__init__(message)


class Agent:
    """Base agent for the DT."""

    def __init__(self: Self, is_spatial: bool = False) -> None:
        """Init an abstract Agent object.

        Args:
            is_spatial: whether the agent is a spatial agent.

        """
        self.unique_id: str = shortuuid.uuid()
        """A unique id for identifying this agent, created using `shortuuid`."""

        self.is_spatial = is_spatial
        """Whether this agent is spatially located."""

    def step(self: Self, **env_data: dict[str, Any]) -> None:
        """Execute a single step of the agent."""

    def update(self: Self) -> None:
        """Update itself for the next timestep."""
