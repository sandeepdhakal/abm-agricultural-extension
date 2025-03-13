This is an agent-based model for simulating the impact of extension services on growers' preference for novel pest control methods, such as Biocontrol.

# Installing Dependencies

[`uv`](https://docs.astral.sh/uv/) is used for managing the project. Therefore, installing the dependencies is as simple as:

```bash
# In the project directory
uv venv
source .venv/bin/activate
uv sync
```

# Configuring Simulations

## Configuration

When running the simulations, we need to provide simulation configurations in the form of YAML files. The configuration is done using the [`hydra`](https://hydra.cc/docs/intro/) library. Sample config files are provided in the [config](./config) directory.

The structure of the configuration is specified in [./model/config.py](./model/config.py).

## Simulation Inputs

The model also requires two input files: geospatial information about the fields in the landscape, and a network file with information about growers' connections with each other. Both files should have the same basename provided with the `gis_file` key in the configuration file.

1. A geopandas dataframe which takes the form <gis_file>.parquet. This dataframe is used to configure the landscape in the model on which the grower agents operate, and as such should contain information about each field in the landscape. An example file is provided in [./input/n10_m2.parquet](./input/n10_m2.parquet).

```python
                                                     geometry  area_sq_km  cropping grower
spatial_id
24          POLYGON ((16572508.594 -3905126.01, 16572512.6...    0.685917      True      1
25          POLYGON ((16576380.471 -3905939.778, 16576336....    0.154849      True      3
30          POLYGON ((16576089.023 -3906290.98, 16576024.5...    2.202346      True      3
31          POLYGON ((16577878.196 -3903978.572, 16577884....    2.462458      True      3
33          POLYGON ((16579827.639 -3905876.831, 16579887....    0.863511      True      3
```

As you can see this dataframe contains information about each field and the owner of that field, and each grower has a unique id.

2. A network file in the form <gis_file>.gml.gz, create using the `networkx` library.
   Each node in this network should correspond to a grower in the above dataframe.

```python
G = nx.read_gml('n10_m2.gml.gz')

G.nodes
# NodeView(('1', '3', '8', '9', '4', '0', '2', '7', '6', '5'))

G.edges
# EdgeView([('1', '3'), ('1', '8'), ('1', '4'), ('1', '0'), ('1', '7'), ('1', '6'), ('3', '9'), ('3', '4'),
# ('3' , '6'), ('3', '5'), ('8', '9'), ('8', '7'), ('8', '5'), ('9', '0'), ('9', '2'), ('4', '2')])
```

For examples of how to generate these networks, see [generate_networks.py](./generate_networks.py).

# Running Simulations

Once the input files and configuration files have been added, we can run the simulation by calling the [simulator.py](./simulator.py) file.

```python
python simulator.py
```

You can also leverage other `hydra` options if you want to provide additional options when running simulations.

## Simulation Outputs

All simulation outputs are saved to the `multirun` directory by default (which can be changed in the config file).
Each unique parameter configuration set has its own subdirectory, with the parent folders named after the date/time the simulation was started.
For details of the naming conventions used, see the [multirun documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/).
The outputs are saved as _parquet_ files with _brotli_ compression.

There is a sample script [concat_sim_output.py](./concat_sim_output.py) for concatenating the individual output dataframes as one multi-index dataframe.

```bash
➜ python concat_sim_output.py --help
usage: concat_sim_output.py [-h] input outfile

This script reads the output dataframes saved in the 'input' directory, concatenates them as one dataframe and saves the resulting dataframe as a parquet
file using brotli compression.

positional arguments:
  input       directory with all sim outputs
  outfile     name of the concatenated dataframe.

options:
  -h, --help  show this help message and exit
```
