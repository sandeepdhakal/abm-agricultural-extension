"""Structured config definitions for the IPM simulation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import MISSING


@dataclass
class ControlPreference:
    """Config schema for pest control method preference."""

    pest_name: str = MISSING
    control_name: str = MISSING
    preference: float = MISSING


@dataclass
class ControlEfficacyPerPest:
    """Config schema for pest efficacy for a pest."""

    day: int = MISSING
    mortality: int = MISSING


@dataclass
class ControlEfficacy:
    """Config schema for efficacy of a pest control method."""

    pest_name: str = MISSING
    pest_efficacy: list[ControlEfficacyPerPest] = MISSING


@dataclass
class PestControl:
    """Config schema for a pest control method."""

    name: str = MISSING
    control_type: str = MISSING
    efficacy: Optional[list[ControlEfficacy]] = None


@dataclass
class GrowerControlPreferenceDistribution:
    """Config scheme for the proportion of growers with a specific preference."""

    control_preference: ControlPreference = MISSING
    grower_proportion: float = MISSING


@dataclass
class GrowerConfig:
    """Config schema for growers."""

    control_preference_distribution: list[GrowerControlPreferenceDistribution] = MISSING
    risk_aversion: int = 5


@dataclass
class ExtensionConfig:
    """Config schema for extension services."""

    num_ext_officers: int = 1

    # we'll instantiate ipm_dt.agents.InteractionPolicy objects using
    # hydra's hydra.utils.instantiate method.
    # see https://hydra.cc/docs/1.2/advanced/instantiate_objects/overview/
    interaction_policy: Dict[str, Any] = MISSING


@dataclass
class SocialConfig:
    """Config schema for demographics."""

    social_network: bool = False
    spatial_network: bool = True

    # learning
    social_learning_method: str = "average"

    # learning weight
    lw: float = 0.5

    # how frequently the growers interact with each other (in days)
    gif: int = 7

    # extension services available for growers
    extension: list[ExtensionConfig] = MISSING


@dataclass
class GeoConfig:
    """Config schema for geospatial information."""

    gis_file: str = MISSING


@dataclass
class Config:
    """Config scheme for DT simulation."""

    # config for setting up the DT
    input_dir: str = "./input"
    start_date: str = "${now:%Y-%m-%d}"
    end_date: str = "${now:%Y-%m-%d}"
    gis_file: str = MISSING
    configurator: str = MISSING
    data_sources: list[str] = MISSING

    # social config
    social: SocialConfig = MISSING
    grower: GrowerConfig = MISSING

    # scenario related configs
    pest_control: list[PestControl] = MISSING

    # repeating runs with different seeds
    seed: int = 10_000
