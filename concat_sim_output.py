"""A simple script to concat sim output files into a single dataframe."""

import os

import pandas as pd
from omegaconf import OmegaConf


def get_config(fpath: str) -> tuple[str, ...]:
    """Return the hydra/omega configurations from filepath `fpath`.

    The configuration is read from the .hydra/overrides.yaml file located in the same
    directory as fpath. For all :v pairs in the yaml file, a tuple of (v1, v2, ...) is
    returned so it can be used with pandas concat.
    """
    # the base directory where `fpath` is located
    base_dir = fpath.rsplit("/", maxsplit=1)[0]

    # read config file from .hydra/overrides.yaml
    conf = OmegaConf.load(f"{base_dir}/.hydra/overrides.yaml")

    # put all values in a tuple
    return tuple(tuple(x.split("="))[1] for x in conf)


def concat_outputs(output_path: str) -> pd.DataFrame:
    """Concat all output files in `output_path` as one dataframe."""
    # iterate through all files in the output_path and save dataframe paths
    dfs = []
    for root, _dirs, files in os.walk(output_path):
        for file in files:
            with open(os.path.join(root, file), "r") as f:
                if f.name.endswith(".parquet"):
                    dfs.append(f.name)

    # get configs
    keys = ["ext", "slw", "gif", "landscape", "seed"]
    configs = [get_config(x) for x in dfs]
    # then read dataframes
    dfs = [pd.read_parquet(x) for x in dfs]

    # concat all and return
    return pd.concat(dfs, keys=configs, names=keys)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "This script reads the output dataframes saved in the 'input' directory, "
            "concatenates them as one dataframe and saves the resulting dataframe as a "
            "parquet file using brotli compression."
        )
    )
    parser.add_argument("input", help="directory with all sim outputs", type=str)
    parser.add_argument("outfile", help="name of the concatenated dataframe.", type=str)
    args = parser.parse_args()

    df = concat_outputs(args.input)
    df.to_parquet(args.outfile, compression="brotli")
    print(f"Successfully saved otuput to {args.outfile}")
