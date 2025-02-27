# -*- coding: utf-8 -*-
"""A digital twin module.

This module provides the ability to create a digital twin model, as well as some helper
functions.
"""

from datetime import date, timedelta
from typing import Any, Iterator, Optional, Self, Type

import geopandas as gpd
import pandas as pd

from .protocols import DTAgent, DTScenario, DTSpatialObject, DTSubmodel


class Model:
    """Digital Twin class for running IPM simulation."""

    def __init__(
        self: Self,
        gdf: gpd.GeoDataFrame,
        scenario: DTScenario,
        start_date: Optional[date] = None,
    ) -> None:
        """Init DigitalTwin.

        Args:
            gdf: The geodataframe for the DT's geography.
            scenario: A scenario object holdinng the contextual information for the DT.
            start_date: The start date for the DT.
        """
        self.running: bool = False
        """Whether the model simulation is currently running."""

        self._used_ids = ()
        self._agents = {}

        self.timestep: int = 0
        """The current timestep in the DT. A timestep corresponds to 1 day."""

        self.start_date: date = start_date
        """The start date for the DT."""

        self.date: date = start_date or date.today()
        """The current date in the DT."""

        self.scenario: DTScenario = scenario
        """A scenario object holdinng all the contextual information for the DT"""

        self.submodels: dict[str, DTSubmodel] = {}
        """A dict of different submodels that are part of this DT."""

        self._spatial_objects = {}

        self.cropping_mask: pd.Series = None
        """Mask for filtering crops from the base `gdf`."""

        self._set_geodataframe(gdf)
        self._set_data_structures()

    def _set_data_structures(self: Self) -> None:
        # set data structures
        self.data: dict[str, Any] = {}  # TODO: move data outside of the DT class
        """Data releated to the DT."""

        self.data["field"] = pd.DataFrame(columns=["crop", "control_method", "awm"])
        self.data["field"].index = pd.MultiIndex(
            levels=[[], []],
            names=["timestep", "spatial_id"],
            codes=[[], []],
        )

    def _set_geodataframe(self: Self, gdf: gpd.GeoDataFrame) -> None:
        """Set the DT's spatial data."""
        self.gdf: gpd.GeoDataFrame = gdf.sort_index()
        """The geodataframe for the DT's geography."""

    def step(self: Self) -> None:
        """Move to the next timestep.

        All registered agents, spatial objects, the contextual scenario and submodels
        are asked to move to the next timestep. It is up to them to decide if and what
        they do.
        """
        for spo in self.spatial_objects():
            spo.step()

        self.timestep += 1
        self.date += timedelta(days=1)

    def update(self: Self) -> None:
        """Prepare for the next timestep.

        All registered agents, spatial objects, the contextual scenario and submodels
        are asked to prepare for the next timestep. It is up to them to decide whether
        or not and how they update themselves.
        """
        # all submodels, first to allow agents to act after contextual updates
        for submodel in self.submodels.values():
            submodel.step()

        # for first iteration, check the data
        if not self.data:
            self.data["pest_risk"] = pd.DataFrame(
                index=self.gdf[self.cropping_mask].index,
            )

        for spo in self.spatial_objects():
            spo.update()
            # spo.log()

        for agent in self.agents():
            agent.update()

    def check_unique_id(self: Self, new_id: int) -> bool:
        """Check whether `new_id` is a unique id for the DT."""
        return f"a{new_id}" not in self._agents

    def add_spatial_object(self: Self, spatial_obj: DTSpatialObject) -> None:
        """Add a `spatial_obj` spatial object to the DT.

        While adding the spatial object, the `dt` of the object is set to this
        DT, and its `unique_id` is set to `spatial_id` if not already set.
        """
        spatial_obj.dt = self

        if not hasattr(spatial_obj, "spatial_id") or spatial_obj.spatial_id is None:
            raise ValueError("spatial object must have spatial_id")

        spatial_obj.unique_id = spatial_obj.spatial_id

        if spatial_obj.spatial_id in self._spatial_objects:
            raise ValueError("id is not unique")
        if spatial_obj.spatial_id not in self.gdf.index:
            raise ValueError("id doesn't exist in gdf")

        self._spatial_objects[spatial_obj.spatial_id] = spatial_obj

    def spatial_objects(
        self: Self,
        /,
        obj_type: Optional[str] = None,
        ids: Optional[Iterator[int]] = None,
    ) -> Iterator[DTSpatialObject]:
        """Return spatial objects in the DT that meet the specifications.

        Args:
            ids: ids for which to get spatial objects. If ids is None, all spatial
                objects in the DT are returned.
            obj_type: the type of the objects to be returned.
        """
        filtered_objs = (
            list(
                filter(
                    lambda x: isinstance(x, obj_type),
                    self._spatial_objects.values(),
                ),
            )
            if obj_type is not None
            else list(self._spatial_objects.values())
        )

        # TODO: return a generator
        return (
            filtered_objs
            if ids is None
            else [o for o in filtered_objs if o.spatial_id in ids]
        )

    def add_agent(self: Self, agent: DTAgent) -> None:
        """Add the agent to the DT, if it doesn't already exist.

        Only the `unique_id`` of the agent is used to check if it already exists.
        """
        if not self.check_unique_id(agent.unique_id):
            raise ValueError("id is not unique.")
        self._agents[agent.unique_id] = agent
        agent.dt = self

    def agents(
        self: Self,
        /,
        agent_type: Optional[Type] = None,
        ids: Iterator[str] = None,
    ) -> Iterator[DTAgent]:
        """Return a generator to get agents corresponding to the specified ids.

        Args:
            agent_type: The type of the agents to return for iteration.
            ids: Ids for which to get agents. If ids is None, all agents are returned.
        """
        if agent_type is not None and not isinstance(agent_type, type):
            raise TypeError("Invalid agent_type")

        for i in ids or list(self._agents.keys()):
            if (a := self._agents.get(i)) and (
                agent_type is None or isinstance(a, agent_type)
            ):
                yield a

    def add_submodel(self: Self, name: str, submodel: DTSubmodel) -> None:
        """Add a submodel to this digital twin model.

        All submodels must be added using this method. The submodel must implement the
        `step` method and a `NotImplementedError` will be thrown otherwise. This method
        will link the digital twin to the submodel by assigning the digital twin model
        to the submodel's `dt` attribute. This allows the submodel to reference to the
        parent digital twin model.

        Args:
            name: The name of the submodel. Must be unique, otherwise an error will be
                thrown.
            submodel : any A submodel that will be part of this DT.

        Raises:
            NotImplementedError: If the submodel doesn't implement the *step* method.
        """
        if hasattr(submodel, "step") and callable(submodel.step):
            self.submodels[name] = submodel
            submodel.dt = self
        else:
            raise NotImplementedError(
                "step method must be implemented by all submodels",
            )
