# -*- coding: utf-8 -*-
"""Module with spatial objects for the digital twin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Self

import pandas as pd

from ..pest_control import NO_CONTROL, PestControl
from .base import NotAssignedToAnyDigitalTwinError

if TYPE_CHECKING:
    from .grower import Grower


class SpatialObject:
    """A base class to represent any spatial object."""

    def __init__(self: Self, spatial_id: int) -> None:
        """Init SpatialObject with `spatial_id` id."""
        self.spatial_id: int = spatial_id
        """int: The code to represent this field in the shapefile."""

    def step(self: Self) -> None:
        """Execute a single step of the agent."""

    def update(self: Self) -> None:
        """Update for moving to the next timestep."""


class Field(SpatialObject):
    """A class to represent a field in the DT.

    Only one crop can be grown in a field at any time. However, depending on the
    seasons, market prices, water availability or other considerations, different crops
    might be grown at different times of the year. The grower might also decide to
    change the crop(s) grow or not to grow any crop at all.
    """

    def __init__(
        self: Self,
        spatial_id: int,
        /,
    ) -> None:
        """Init Field with `spatial_id` id."""
        SpatialObject.__init__(self, spatial_id)

        self.control_method: Optional[PestControl] = NO_CONTROL
        """The control method being used, if any."""

        self.ts_since_control_method_change: int = 0
        """The number of timesteps since the last control method change."""

        self.owner: Grower = None
        """:obj:`Grower`: The current owner (grower)."""

    def assign_owner(self: Self, owner: Grower) -> None:
        """Assign the `owner` for this field."""
        owner.own_field(self)

    def set_control_method(
        self: Self,
        control_method: Optional[PestControl] = NO_CONTROL,
    ) -> None:
        """Set the control method for this field.

        Args:
            control_method: The control method being used. This also resets the
                `ts_since_control_method_change` value to 0.
        """
        # only if the control method is different
        if self.control_method == control_method:
            return

        self.control_method = control_method
        self.ts_since_control_method_change = 0
        self.log()

    def update(self: Self) -> None:
        """Prepare for the next timestep."""
        self.ts_since_control_method_change += 1

    def log(self: Self) -> None:
        """Log this field's information with the digital twin's dataframe.

        The following information is logged: the crop grown, and the control method used
        (else None).
        """
        if self.dt is None:
            raise NotAssignedToAnyDigitalTwinError()

        cols = ["crop", "control_method"]
        values = [
            self.crop.crop_name,
            None if self.control_method is None else self.control_method.name,
        ]

        self.dt.data["field"].loc[(self.dt.timestep, self.spatial_id), cols] = values

    def history(
        self: Self,
        start_timestep: int = 0,
        end_timestep: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the history of this field as a dataframe.

        The history contains the following for each timestep: the crop grown, the
        control method used (else None), estimated adult pest population, and risk from
        pests.

        Args:
            start_timestep: The first timestep for which to show the history.
            end_timestep : The final timestep for which to show the history. If None,
                the current final timestep is used. Must be greater than
                `start_timestep`, else the order will be inverted.

        Returns:
            A pandas dataframe with the 'timestep' as the index.
        """
        if self.dt is None:
            raise NotAssignedToAnyDigitalTwinError()

        if end_timestep is not None and end_timestep < start_timestep:
            start_timestep, end_timestep = end_timestep, start_timestep

        timestep_slice = slice(start_timestep, end_timestep)
        field_df = (
            self.dt.data["field"]
            .loc[(timestep_slice, self.spatial_id),]
            .droplevel("spatial_id")
        )

        risk_df = (
            self.dt.data["pest_risk"]
            .loc[(timestep_slice, self.spatial_id),]
            .droplevel("spatial_id")
        )
        pest_df = (
            self.dt.data["pest_pop"]
            .loc[(timestep_slice, self.spatial_id), (slice(None), "adult")]
            .droplevel("spatial_id")
        )
        pest_df.columns = ["_".join(col) for col in pest_df.columns]

        return pd.concat([field_df, risk_df, pest_df], axis=1)
