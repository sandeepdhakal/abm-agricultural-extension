"""A script to run a very simple IPM simulation.

This simulation does the following:
    - reads a config file
    - create a model based on the config file
    - create pest models as specified by a trap observations file
    - simulation the pest population models and the growers' management actions
"""

import logging
import random
import statistics as st
from datetime import date
from functools import partial
from importlib import import_module
from typing import Optional

import geopandas as gpd
import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from numpy.typing import ArrayLike

from model.agents import Grower
from model.config import Config
from model.model import Model

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_path="config", config_name="config")
def ipm_sim(cfg: Config) -> Optional[Model]:
    """Read simulation config from the `cfg`."""
    setup_logger("model_simulation.log")

    # set the random seed
    random.seed(cfg.seed)
    np.random.default_rng(cfg.seed)

    input_dir = cfg["input_dir"]
    gdf = gpd.read_parquet(f"{input_dir}/{cfg['gis_file']}.parquet")

    # configure the landscape
    try:
        configurator = import_module(cfg["configurator"])
    except ModuleNotFoundError:
        logger.error(
            f"The configurator module {cfg['configurator']} was not found. "
            "Terminating!!",
        )
        return None

    model = configurator.configure_model(gdf, cfg, verbose=False)
    log_model_info(model)

    end_date = date.fromisoformat(cfg["end_date"])
    run_simulation(model, end_date)


def log_model_info(model: Model) -> None:
    """Log basic information about the model."""
    logger.info("Model created and configured.")


def run_simulation(model: Model, end_date: date) -> None:
    """Simulate the `model` for `timestep` timesteps."""
    # logger.info(
    #     f"Simulation duration: {model.start_date}" f" - {end_date}",
    # )

    # setup learning
    int_freq = model.cfg["social"]["gif"]
    learn_method = _learning_method(
        model.cfg["social"]["learning"]["social_learning_method"],
    )

    while model.date <= end_date:
        # ask all agents and sub-models to update themselves
        model.update()

        # learning
        if model.timestep % int_freq == 0:
            learn_from_peers(model, learn_method)

        # prepare the model for the next timestep
        model.step()

        # log grower data
        log_grower_data(model)

    logger.info("Simulation completed!!")
    save_model_data(model)


def log_grower_data(model: Model) -> None:
    """Log grower's data in the `model`."""
    ctrl_method = model.available_pest_controls[0]
    rows = [
        (model.timestep, g.unique_id, g.control_method_preference[ctrl_method])
        for g in model.agents(agent_type=Grower)
    ]
    dtypes = {
        "timestep": "uint16",
        "grower_id": pd.StringDtype(),
        "pcb_Biocontrol": "float32",
    }
    df = (
        pd.DataFrame(rows, columns=dtypes.keys())
        .astype(dtypes)
        .set_index(["grower_id", "timestep"])
    )
    model.data["grower"] = pd.concat([model.data["grower"], df])


def setup_logger(logfile: str) -> None:
    """Setup the logger to ouput to `logfile` and the console."""
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode="w")

    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    # console.setFormatter(formatter)
    # logging.getLogger("").addHandler(console)


def save_model_data(model: Model) -> None:
    """Write the model data to files."""
    OUTPUT_DIR = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    opts = {"compression": "brotli"}

    # growers' pest prefs data
    model.data["grower"].reorder_levels(["timestep", "grower_id"]).to_parquet(
        f"{OUTPUT_DIR}/model_growers.parquet",
        **opts,
    )


def learn_from_peers(model: Model, learning_helper: partial[list[float]]) -> None:
    """Implementation of peer learning of pest control preferences for growers."""
    learn_weight = model.cfg["social"]["learning"]["lw"]

    for g in model.agents(agent_type=Grower):
        others_prefs = np.array(
            [
                [ng.control_method_preference.get(k, 0) for ng in g.connections]
                for k in model.available_pest_controls
            ],
        )
        prefs_to_learn = learning_helper(others_prefs)

        for (cm, sp), op in zip(
            g.control_method_preference.items(),
            prefs_to_learn,
            strict=False,
        ):
            new_pref = np.average([sp, op], weights=[1 - learn_weight, learn_weight])
            g.control_method_preference[cm] = new_pref


def most_common(data: ArrayLike, bins: [float]) -> ArrayLike:
    """Return the mean of the most common bin."""
    x = np.digitize(data, bins, right=True)
    bin_pos = np.apply_along_axis(st.mode, 1, x)
    return (bins[bin_pos] + bins[bin_pos - 1]) / 2


def _learning_method(method: str) -> partial[list[float]]:
    match method:
        case "most_common":
            bins = np.linspace(0, 1, 11)
            return partial(most_common, bins=bins)
        case "average":
            return partial(np.mean, axis=1)


if __name__ == "__main__":
    ipm_sim()
