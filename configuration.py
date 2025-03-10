# -*- coding: utf-8 -*-
"""Module with helper functions for creating models."""

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

from model import network as mnet
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


def configure_model(
    gdf: gpd.GeoDataFrame,
    cfg: Config,
    verbose: bool = True,
) -> Model:
    """Configure the `model` so its ready for simulation."""
    verbose and logger.info("configuring the model ...")

    start_date = date.fromisoformat(cfg["start_date"])

    # initialise the model with a base scenario object
    scenario = Scenario()
    scenario.start_year = start_date.year
    model = Model(gdf=gdf, scenario=scenario, start_date=start_date)

    # assign owners for all cropping fields
    model.cropping_mask = model.gdf["cropping"].copy()

    model.num_growers = model.gdf["grower"].nunique()
    assign_fields_to_growers(model, model.cropping_mask, model.gdf["grower"].to_numpy())

    # pest control methods
    configure_pest_control(model, cfg)

    # GROWERS
    for g in model.agents(agent_type=Grower):
        g.risk_aversion = cfg["grower"]["risk_aversion"]

    # setup grower network
    initialise_grower_network(model, cfg)

    # how frequently growers interact with each other
    model.grower_interaction_frequency = cfg["social"]["gif"]

    # cache growers' connections
    for g in model.agents(agent_type=Grower):
        g.connections = mnet.get_network_connections(model, g)

    # growers' pest control preferences
    set_control_method_preferences(model, cfg)

    # setup grower dataframe
    init_grower_dataframe(model)

    config_extension(model, cfg)

    # save the configuration within the model for later access.
    model.cfg = cfg

    return model


def config_extension(model: Model, cfg: Config) -> None:
    """Configure the extension info for the `model` using configuration `cfg`."""
    for ext_info in cfg["social"]["extension"]:
        policy = instantiate(ext_info["interaction_policy"])(
            network=model.agent_network
        )
        for _ in range(ext_info["num_ext_officers"]):
            eo = ExtensionOfficer(interaction_policy=policy)
            eo.pest_control = model.available_pest_controls[0]
            model.add_agent(eo)


def init_grower_dataframe(model: Model) -> None:
    """Initialise a dataframe for recording grower info in the `model`."""
    biocontrol = model.available_pest_controls[0]
    rows = [
        (model.timestep, g.unique_id, g.control_method_preference[biocontrol])
        for g in model.agents(agent_type=Grower)
    ]
    dtypes = {
        "timestep": "uint16",
        "grower_id": pd.StringDtype(),
        "pcb_Biocontrol": "float32",
    }
    model.data["grower"] = (
        pd.DataFrame(rows, columns=dtypes.keys())
        .astype(dtypes)
        .set_index(["grower_id", "timestep"])
    )


def config_scenario(
    model: Model,
    cfg: Config,
    start_date: date,
    input_dir: str,
    verbose: bool,
) -> Scenario:
    """Configure the passed scenario instance based on the provided configuration.

    Args:
        model: The model to which this scenario will be applied.
        cfg: A dictionary like object with configuration values.
        start_date: The start date of this simulation.
        input_dir: Where to find the input files to be read.
        verbose: If verbose, show logs of different processes during configuration.
    """


def assign_fields_to_growers(
    model: Model,
    field_filter: pd.Series,
    grower_ids: list[int],
) -> None:
    """Assign fields to growers.

    This will create both the field and grower objects in the model.

    Args:
        model: The model to be acted on.
        field_filter: The filter for obtaining fields in the model.
        grower_ids: The ids of growers that will be assigned fields.
    """
    model.gdf.loc[~field_filter, "grower"] = pd.NA
    model.gdf.loc[field_filter, "grower"] = grower_ids

    grower_ids = {}

    # grower's propensity to change their practice with new information.
    propensity_to_change = _gen_values_in_uniform_distribution(10000)

    grouped = model.gdf.loc[field_filter].groupby("grower")
    for gid, group in grouped:
        grower = Grower()
        model.add_agent(grower)
        grower_ids[gid] = grower.unique_id
        for fid in group.index.to_numpy():
            field = Field(fid)
            model.add_spatial_object(field)
            field.assign_owner(grower)

        # propensity to change
        grower.propensity_to_change = choice(propensity_to_change)

    model.gdf.loc[field_filter, "grower"] = model.gdf.loc[
        field_filter, "grower"
    ].replace(
        grower_ids,
    )


def _gen_values_in_uniform_distribution(N: int) -> list[int]:
    """Pick `N` values within a uniform distribution (0,1)."""
    lower, upper = 0, 1
    mu, sigma = 0.5, 0.2
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs(N)


def initialise_grower_network(model: Model, cfg: Config) -> None:
    """Integrate an existing network of agents or initialise a new one.

    We'll check for an existing network as follows:
    - filename of the cfg['gis_file'].gml.gz

    We'll create the network of growers using:
    - network based on the crop industry
    - network based on grower groups

    Then we'll merge the two networks and assign to the model.
    """
    # We'll check if a network file with the same name as 'gis_file' exists
    # yes: use that network, else: construct a new network
    network_file = f"{cfg['input_dir']}/{cfg['gis_file']}.gml.gz"
    if os.path.isfile(network_file):
        mnet.integrate_grower_network(model, nx.read_gml(network_file))
        return

    networks = []
    # spatial network
    if cfg["social"]["spatial_network"]:
        networks += [mnet.create_grower_spatial_network(model)]

    if cfg["social"]["social_network"]:
        # crop industry network
        networks += [mnet.create_crop_industry_network(model)]

        # grower group network
        networks += [mnet.create_grower_group_network(model)]

    model.agent_network = nx.compose_all(networks)


def configure_pest_control(model: Model, cfg: Config) -> None:
    """Add pest control methods available in the `model` by parsing `cfg`."""
    model.available_pest_controls = [BioControl("Biocontrol")]


def set_control_method_preferences(model: Model, cfg: Config) -> None:
    """Read `cfg` and assign pest control preferences for agents in the `model`."""
    # make sure grower distribution totals to 100
    control_cfg = cfg["grower"]["control_preference_distribution"]
    if sum([x["grower_proportion"] for x in control_cfg]) != 100:  # noqa: PLR2004
        raise ValueError("grower_proportion should total to 100")

    grower_iter = model.agents(agent_type=Grower)
    slices = [
        round(x["grower_proportion"] * model.num_growers / 100) for x in control_cfg
    ]
    grower_iters = [islice(grower_iter, 0, i) for i in slices]
    for it, prefs in zip(grower_iters, control_cfg, strict=True):
        pref = prefs["control_preference"]
        control = next(
            x for x in model.available_pest_controls if x.name == pref["control_name"]
        )
        for grower in it:
            grower.control_method_preference[control] = pref["preference"]
