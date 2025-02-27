# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: ag-ext
#     language: python
#     name: abm-extension
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import colorcet as cc  # noqa: F401 # used indirectly
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx

from model import utils

# %%
INPUT_DIR = "input"
IMG_DIR = "images"

# %% [markdown]
"""
## Geospatial

Read the geodataframe
"""

# %%
gdf = gpd.read_parquet("input/landscape.parquet")

# %% [markdown]
"""
# Allocate fields to growers
"""

# %%
n = 20

# separate the landscape into clusters of 'num_growers'
gdf["grower"] = utils.cluster_shapes(
    gdf[gdf["cropping"]],
    num_clusters=n,
).astype("str")

G = nx.Graph()


def get_neighbours(gdf, grower, distance):
    """Get neighhbours for the `grower` within a `distance` (in meters) in the landscape `gdf`."""
    fields_df = gdf[gdf.grower == grower]
    return set(
        gdf[gdf.cropping]
        .sjoin_nearest(
            fields_df[["geometry"]],
            how="inner",
            max_distance=distance,
            exclusive=True,
        )["grower"]
        .unique()
        .tolist()
    )


for index, rows in gdf.groupby("grower"):
    fields_df = gdf[gdf.grower == index]
    neighbours = get_neighbours(gdf, index, 5000)

    neighbours.remove(index)
    if not neighbours:
        neighbours = get_neighbours(gdf, index, 7000)

    G.add_edges_from([(index, n) for n in neighbours])
    G.remove_edges_from(nx.selfloop_edges(G))

# %%
"""Generate a spatial network of the growers in the landscape."""

# %%
# spatial location of nodes/grower
centroids = gdf.dissolve(by="grower").centroid
positions = {k: [v.x, v.y] for k, v in centroids.to_dict().items()}
nx.set_node_attributes(G, positions, "pos")

# %%
fig, ax = plt.subplots(figsize=(8, 8))
gdf.dissolve(by="grower", aggfunc="sum").plot(
    edgecolor="black", cmap="tab20", alpha=0.5, ax=ax
)
nx.draw(G, pos=nx.get_node_attributes(G, "pos"), ax=ax, node_size=20)
plt.savefig(f"{IMG_DIR}/spatial_20g.png", bbox_inches="tight", dpi=1200)
plt.show()

# %%
path = "./input"
nx.write_gml(G, f"{path}/spatial_20m.gml.gz")
gdf.to_parquet(f"{path}/spatial_20m.parquet", compression="brotli")


# %%
def create_barabasi_grower_spatial_network(
    n: int, m: int, gdf: gpd.GeoDataFrame, default_trust=1.0
):
    # separate the landscape into clusters of 'num_growers'
    gdf["grower"] = utils.cluster_shapes(
        gdf[gdf["cropping"]],
        num_clusters=n,
    ).astype("str")

    G = nx.barabasi_albert_graph(n=n, m=m)

    # connect growers and network nodes
    growers = gdf["grower"].unique()
    nx.relabel_nodes(G, {i: g for i, g in enumerate(growers)}, copy=False)

    # trust between nodes: unidirectional
    trust_attrs = {edge: {"trust": default_trust} for edge in G.edges}
    nx.set_edge_attributes(G, trust_attrs)

    # spatial location of nodes/grower
    centroids = gdf.dissolve(by="grower").centroid
    positions = {k: [v.x, v.y] for k, v in centroids.to_dict().items()}
    nx.set_node_attributes(G, positions, "pos")

    return (gdf, G)


# %%
num_growers = [10, 20, 30]
m = 2  # each grower is connected to at least 2 other growers

for n in num_growers:
    _gdf, G = create_barabasi_grower_spatial_network(n, m, gdf.copy())
    nx.write_gml(G, f"{path}/n{n}_m{m}.gml.gz")
    _gdf.to_parquet(f"{path}/n{n}_m{m}.parquet", compression="brotli")


# %%
def draw_grower_network_on_map(gdf, G, ax):
    # draw the map
    gdf.dissolve(by="grower", aggfunc="sum").plot(
        edgecolor="black", ax=ax, cmap="tab20", alpha=0.3
    )

    # draw network on top of the map
    nx.draw(G, pos=nx.get_node_attributes(G, "pos"), ax=ax, node_size=20)


# %%
# fig, axes = plt.subplots(2, 3, layout='constrained', figsize=(12,8), sharex=True, sharey=True)
# for n, ax in zip(num_growers, axes.flat):
#     gdf = gpd.read_parquet(f'{path}/n{n}_m{m}.parquet')
#     G = nx.read_gml(f'{path}/n{n}_m{m}.gml.gz')
#     draw_grower_network_on_map(gdf, G, ax)
#
#
# # delete ax without data
# for ax in axes.flat:
#     ## check if something was plotted
#     if not bool(ax.has_data()):
#         fig.delaxes(ax) ## delete if nothing is plotted in the axes obj
# plt.show()

# %%
for n in num_growers:
    gdf = gpd.read_parquet(f"{path}/n{n}_m{m}.parquet")
    G = nx.read_gml(f"{path}/n{n}_m{m}.gml.gz")

    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.dissolve(by="grower", aggfunc="sum").plot(
        edgecolor="black", cmap="tab20", alpha=0.5, ax=ax
    )
    nx.draw(G, pos=nx.get_node_attributes(G, "pos"), ax=ax, node_size=20)
    plt.savefig(f"{IMG_DIR}/social_{n}g.png", bbox_inches="tight", dpi=1200)
