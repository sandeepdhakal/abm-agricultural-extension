# -*- coding: utf-8 -*-

"""Module to represent a grower agent."""

from typing import Any, Self


from ..pest_control import PestControl
from .base import Agent
from .spatial import Field


class Grower(Agent):
    """Grower who can own fields."""

    def __init__(
        self: Self,
        **attrs: float,
    ) -> None:
        """Init Grower."""
        Agent.__init__(self)

        self.owned_fields: list[Field] = []  # doesn't own any fields yet
        """The fields (ids) owned by the agent."""

        self.control_method_preference: dict[PestControl, float] = {}
        """Preference [0-1] for pest control methods."""

        self.propensity_to_change: float = 0.5
        """How open this grower is to adopting new practices."""

        self.risk_aversion: float = 1.0
        """How risk averse the grower is to changing their current practices."""

        self.connections: list[Agent] = []
        """Other agents that this grower is connected to."""

    def own_field(self: Self, field: Field) -> None:
        """Add `field` to the list of fields owned by this grower."""
        if field.unique_id not in self.owned_fields:
            self.owned_fields.append(field.unique_id)
        field.owner = self

    def step(self: Self, **env_data: dict[str, Any]) -> None:
        """Take some action(s) for this timestep.

        When deciding these actions, the grower will consider their own context as well
        as any additional info about its environment (*env_data*).

        At this stage the following decisions are made:
            1. Update pest control method if required.

        Args:
            env_data: Any additional information about their environment that the grower
                needs to consider. Some information this can contain includes:
                `risk_df`: pest risk to fields
        """
        # for now, just select the control method the grower has the best experience
        # with different control methods can be chosen for different fields, based on
        # the risk to them

    def update(self: Self) -> None:
        """Update itself for the next timestep."""
