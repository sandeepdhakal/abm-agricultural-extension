# -*- coding: utf-8 -*-
"""Scenario management.

Using scenarios is a convenient way of passing large volumes of contextual information
to digital twin models.
"""

from calendar import monthrange
from typing import Optional, Self

import geopandas as gpd
import numpy as np
import pandas as pd


EPSG = 3577
EPSG_GEO = 4283


class Scenario:
    """Class to collect parametersvalues and contextual information for a digital twin.

    Must include a shapefile for spatial information. The contextual information is
    mplemented using several pandas dataframe, where the columns are the values and
    each row represents the scenario for a timestep in the model. There can be multiple
    dataframes for different sub-scenarios, such as crops, demographics, etc.

    For the shapefile, the data must already in the correct projection.

    The dataframes should have the following columns.

    *Population Dataframe*: (spatial_object_id, population, type)
    population is a multi-level column with sub-columns for each timestep.
    """

    def __init__(
        self: Self,
        timeseries_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> None:
        """Init Scenario.

        Args:
            timeseries_data: data related to different aspects of the scenario for each
                timestep.
        """
        self.timestep: int = 0
        """The current timestep correspondind to the associated DT."""

        self.data = {}  # empty dict of scenario data
        if isinstance(timeseries_data, dict):
            for k, v in timeseries_data.items():
                self[k] = v

    def __setitem__(self: Self, key: str, df: pd.DataFrame) -> None:
        """Assign new data as `df` for this scenario with `key`."""
        if not isinstance(key, str):
            raise KeyError("only string keys are accepted")
        if not isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
            raise (
                KeyError(
                    "This accepts either a pandas dataframe or geopandas geodataframe,"
                    f", but received {type(df)} instead",
                )
            )

        self.data[key] = df

    def __getitem__(self: Self, key: str) -> pd.DataFrame:
        """Retrive data from the scenario corresponding to `key`."""
        if not isinstance(key, str):
            raise KeyError("only string keys are accepted")

        try:
            return self.data[key]
        except KeyError as err:
            raise KeyError(f"'{key}' doesn't exist in the scenario data") from err

    def __iter__(self: Self) -> Self:
        """Return iterator for this scenario."""
        self.timestep = 0
        return self

    def __next__(self: Self) -> dict[str, pd.DataFrame]:
        """Return the data corresponding to the current timestep in the scenario."""
        try:
            ts_data = {
                label: df.loc[pd.IndexSlice[:, self.timestep], :]
                for label, df in self.data.items()
            }
            self.timestep += 1
            return ts_data
        except KeyError as err:
            raise StopIteration from err

    def __str__(self: Self) -> str:
        """Return a string representation of this scenario."""
        return (
            f"A scenario with the following timeseries data: {list(self.data.keys())}."
            f"Current timestep: {self.timestep}"
        )

    def keys(self: Self) -> list[str]:
        """Return the available keys in the data dictionary."""
        return self.data.keys()

    def update(self: Self, timestep: int) -> None:
        """Update the scenario for the specified timestep."""
        pass


def _number_of_days(start_year: int, num_years: int) -> int:
    """Return the total number of days in the specified years."""
    return np.sum(
        [
            monthrange(y, m + 1)[1]
            for y in range(start_year, start_year + num_years)
            for m in range(12)
        ],
    )
