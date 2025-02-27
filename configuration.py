# -*- coding: utf-8 -*-
"""Module with helper functions for creating digital twins."""

from __future__ import annotations

import logging
import os
from datetime import date
from itertools import islice
from random import choice
from typing import TYPE_CHECKING

import geopandas as gpd
import networkx as nx
import pandas as pd
from hydra.utils import instantiate
from scipy import stats

from model import network as dtn
from model.agents import (
    ExtensionOfficer,
    Field,
    Grower,
)
from model.model import Model
from model.scenario import Scenario
from model.pest_control import BioControl

if TYPE_CHECKING:
    from .sim_config import Config

GROWER_SINGLE_CROP_FIELDS = 10
PESTS = ["rgb", "qfly"]
NUM_PEST_TRAPS = 5


logger = logging.getLogger(__name__)


def configure_dt(
    gdf: gpd.GeoDataFrame,
    cfg: Config,
    verbose: bool = True,
) -> Model:
    """Configure the digital twin `dt` so its ready for simulation."""
    verbose and logger.info("configuring the DT ...")

    start_date = date.fromisoformat(cfg["start_date"])

    # initialise the DT with a base scenario object
    scenario = Scenario()
    scenario.start_year = start_date.year
    dt = Model(gdf=gdf, scenario=scenario, start_date=start_date)

    # assign owners for all cropping fields
    dt.cropping_mask = dt.gdf["cropping"].copy()

    dt.num_growers = dt.gdf["grower"].nunique()
    assign_fields_to_growers(dt, dt.cropping_mask, dt.gdf["grower"].to_numpy())

    # num_growers = cfg["social"]["num_growers"]
    # gids = utils.cluster_shapes(dt.gdf[dt.cropping_mask], num_clusters=num_growers)
    # assign_fields_to_growers(dt, dt.cropping_mask, gids)

    # pest control methods
    configure_pest_control(dt, cfg)

    # GROWERS
    for g in dt.agents(agent_type=Grower):
        g.risk_aversion = cfg["grower"]["risk_aversion"]

    # setup grower network
    initialise_grower_network(dt, cfg)

    # how frequently growers interact with each other
    dt.grower_interaction_frequency = cfg["social"]["gif"]

    # cache growers' connections
    for g in dt.agents(agent_type=Grower):
        g.connections = dtn.get_network_connections(dt, g)

    # growers' pest control preferences
    set_control_method_preferences(dt, cfg)

    # setup grower dataframe
    init_grower_dataframe(dt)

    config_extension(dt, cfg)

    # save the configuration within the digital twin for later access.
    dt.cfg = cfg

    return dt


def config_extension(dt: Model, cfg: Config) -> None:
    """Configure the extension info for the `dt` using configuration `cfg`."""
    for ext_info in cfg["social"]["extension"]:
        policy = instantiate(ext_info["interaction_policy"])(network=dt.agent_network)
        for _ in range(ext_info["num_ext_officers"]):
            eo = ExtensionOfficer(interaction_policy=policy)
            eo.pest_control = dt.available_pest_controls[0]
            dt.add_agent(eo)


def init_grower_dataframe(dt: Model) -> None:
    """Initialise a dataframe for recording grower info in the `dt`."""
    biocontrol = dt.available_pest_controls[0]
    rows = [
        (dt.timestep, g.unique_id, g.control_method_preference[biocontrol])
        for g in dt.agents(agent_type=Grower)
    ]
    dtypes = {
        "timestep": "uint16",
        "grower_id": pd.StringDtype(),
        "pcb_Biocontrol": "float32",
    }
    dt.data["grower"] = (
        pd.DataFrame(rows, columns=dtypes.keys())
        .astype(dtypes)
        .set_index(["grower_id", "timestep"])
    )


def config_scenario(
    dt: Model,
    cfg: Config,
    start_date: date,
    input_dir: str,
    verbose: bool,
) -> Scenario:
    """Configure the passed scenario instance based on the provided configuration.

    Args:
        dt: The DT to which this scenario will be applied.
        cfg: A dictionary like object with configuration values.
        start_date: The start date of this simulation.
        input_dir: Where to find the input files to be read.
        verbose: If verbose, show logs of different processes during configuration.
    """


def assign_fields_to_growers(
    dt: Model,
    field_filter: pd.Series,
    grower_ids: list[int],
) -> None:
    """Assign fields to growers.

    This will create both the field and grower objects in the DT.

    Args:
        dt: The DT to be acted on.
        field_filter: The filter for obtaining fields in the DT.
        grower_ids: The ids of growers that will be assigned fields.
    """
    dt.gdf.loc[~field_filter, "grower"] = pd.NA
    dt.gdf.loc[field_filter, "grower"] = grower_ids

    grower_ids = {}

    # grower's propensity to change their practice with new information.
    propensity_to_change = _gen_values_in_uniform_distribution(10000)

    grouped = dt.gdf.loc[field_filter].groupby("grower")
    for gid, group in grouped:
        grower = Grower()
        dt.add_agent(grower)
        grower_ids[gid] = grower.unique_id
        for fid in group.index.to_numpy():
            field = Field(fid)
            dt.add_spatial_object(field)
            field.assign_owner(grower)

        # propensity to change
        grower.propensity_to_change = choice(propensity_to_change)

    dt.gdf.loc[field_filter, "grower"] = dt.gdf.loc[field_filter, "grower"].replace(
        grower_ids,
    )


def _gen_values_in_uniform_distribution(N: int) -> list[int]:
    """Pick `N` values within a uniform distribution (0,1)."""
    lower, upper = 0, 1
    mu, sigma = 0.5, 0.2
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs(N)


def initialise_grower_network(dt: Model, cfg: Config) -> None:
    """Integrate an existing network of agents or initialise a new one.

    We'll check for an existing network as follows:
    - filename of the cfg['gis_file'].gml.gz

    We'll create the network of growers using:
    - network based on the crop industry
    - network based on grower groups

    Then we'll merge the two networks and assign to the DT.
    """
    # We'll check if a network file with the same name as 'gis_file' exists
    # yes: use that network, else: construct a new network
    network_file = f"{cfg['input_dir']}/{cfg['gis_file']}.gml.gz"
    if os.path.isfile(network_file):
        dtn.integrate_grower_network(dt, nx.read_gml(network_file))
        return

    networks = []
    # spatial network
    if cfg["social"]["spatial_network"]:
        networks += [dtn.create_grower_spatial_network(dt)]

    if cfg["social"]["social_network"]:
        # crop industry network
        networks += [dtn.create_crop_industry_network(dt)]

        # grower group network
        networks += [dtn.create_grower_group_network(dt)]

    dt.agent_network = nx.compose_all(networks)


def configure_pest_control(dt: Model, cfg: Config) -> None:
    """Add pest control methods available in the `dt` digital twin by parsing `cfg`."""
    dt.available_pest_controls = [BioControl("Biocontrol")]


def set_control_method_preferences(dt: Model, cfg: Config) -> None:
    """Read `cfg` and assign pest control preferences for agents in the `dt`."""
    # make sure grower distribution totals to 100
    control_cfg = cfg["grower"]["control_preference_distribution"]
    if sum([x["grower_proportion"] for x in control_cfg]) != 100:  # noqa: PLR2004
        raise ValueError("grower_proportion should total to 100")

    grower_iter = dt.agents(agent_type=Grower)
    slices = [round(x["grower_proportion"] * dt.num_growers / 100) for x in control_cfg]
    grower_iters = [islice(grower_iter, 0, i) for i in slices]
    for it, prefs in zip(grower_iters, control_cfg, strict=True):
        pref = prefs["control_preference"]
        control = next(
            x for x in dt.available_pest_controls if x.name == pref["control_name"]
        )
        for grower in it:
            grower.control_method_preference[control] = pref["preference"]
