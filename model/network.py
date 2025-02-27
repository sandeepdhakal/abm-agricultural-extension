"""Module that adds networking features to a model."""

from typing import Any

import networkx as nx
import numpy as np
from shapely.geometry import Point

from .agents import Grower
from .model import Model


def create_randomised_social_network(model: Model) -> nx.Graph:
    """Create a random social network of the growers in the model.

    This network is created using the Barabasi Albert method.
    See https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html

    Args:
        model: The model on which to create the network.
    """
    # Get all the growers in the model
    growers = list(model.agents(agent_type=Grower))

    min_connections = 3
    G = nx.barabasi_albert_graph(n=len(growers), m=min_connections)

    # establish a connection between the nodes and growers
    nx.relabel_nodes(G, {i: g.unique_id for i, g in enumerate(growers)}, copy=False)

    # establish a connection between the nodes and growers
    nx.relabel_nodes(G, {i: g.unique_id for i, g in enumerate(growers)}, copy=False)
    # add connection strength for connections:
    # for now, we only use trust and its unidirectional
    trust_attrs = {edge: {"trust": np.random.random()} for edge in G.edges}
    nx.set_edge_attributes(G, trust_attrs)

    return G


def create_grower_spatial_network(
    model: Model,
    neighbour_radius: int = 2000,
) -> nx.Graph:
    """Create a network of growers in the `model`.

    The network is created such that a grower *g* is connected to other growers that
    own fields within a `neighbour_radius` distance of the fields of *g*.

    Args:
        model: The model on which to create the network.
        neighbour_radius: The radius (in metres) for matching neighbouring fields.
    """
    G = nx.Graph()

    # Get all the growers in the model
    growers = list(model.agents(agent_type=Grower))

    # fields adjacent to those owned by a grower
    for grower in growers:
        fields_df = model.gdf[model.gdf.grower == grower.unique_id]
        neighbour_ids = (
            model.gdf[model.cropping_mask]
            .sjoin_nearest(
                fields_df[["geometry"]],
                how="inner",
                max_distance=neighbour_radius,
                exclusive=True,
            )["grower"]
            .unique()
            .tolist()
        )

        if not neighbour_ids:
            continue
        neighbours = model.agents(ids=neighbour_ids)
        G.add_edges_from(
            [
                (grower.unique_id, neighbour.unique_id, {"trust": np.random.random()})
                for neighbour in neighbours
            ],
            type="grower",
        )
        G.add_nodes_from(G.nodes, type="grower")

    # remove self edges
    G.remove_edges_from(nx.selfloop_edges(G))

    gdf = model.gdf.dissolve(by="grower")
    gdf["centroid"] = gdf.centroid

    positions = {
        k: (v.x, v.y) for k, v in gdf.loc[list(G.nodes), "centroid"].to_dict().items()
    }
    nx.set_node_attributes(G, positions, "pos")

    return G


def integrate_grower_network(model: Model, G: nx.Graph) -> None:
    """Integrate existing network `G` with `model`."""
    # integrating existing network with the model
    # mapping between grower ids in the model and the node id in the network.

    # set the type of all nodes to growers
    nx.set_node_attributes(G, name="type", values="grower")

    # change type of 'pos' from list to tuple.
    # why: because the gml format doesn't support tuples as attributes
    new_pos = {}
    for node, data in G.nodes(data=True):
        if "pos" in data:
            new_pos[node] = {"pos": tuple(data["pos"])}
    nx.set_node_attributes(G, new_pos)

    new_labels = {}
    centroids = model.gdf.dissolve(by="grower").centroid
    for node, data in G.nodes(data=True):
        pt = Point(data["pos"])
        match = centroids[centroids.geom_equals(pt)]
        if len(match):
            new_labels[node] = match.index[0]

    G = nx.relabel_nodes(G, new_labels)
    model.agent_network = G


def create_crop_industry_network(model: Model) -> nx.Graph:
    """Connect all growers belonging to a crop industry in the `model`."""
    G = nx.Graph()  # connections by crop industry

    # Industry Network: same crop type
    for idx, rows in model.gdf[model.cropping_mask][["grower", "crop"]].groupby("crop"):
        growers = rows["grower"].unique().tolist()

        # growers in the same industry are not directly connected to each other
        # G.add_edges_from(itertools.combinations(growers, 2), industry=idx)

        # connections between the industry itself and its members
        G.add_node(idx, type="crop_industry")
        G.add_edges_from([(idx, g) for g in growers], type="crop_industry")

    return G


def get_network_connections(model: Model, agent: Grower) -> list[Grower]:
    """Get the network connections and associated attributes for the specified agent.

    Args:
        model: The model where this operation will be performed.
        agent : Grower The agent for which to get the network connections.

    Returns:
        A dictionary where the key is the other agent and the value is a dictionary of
        connection attributes.
    """
    connections = model.agent_network[agent.unique_id]
    return {next(model.agents(ids=[k])): v for k, v in connections.items()}


def get_centrality_measures(G: nx.Graph) -> dict[str, Any]:
    """Return three different degrees of centrality for the network.

    Args:
        G: The network for which to calculate the centrality measures.

    Returns:
        A dictionary with {centrality_measure_name: value}.
    """
    # Betweenness centrality of a node v is the sum of the fraction of all-pairs
    # shortest paths that pass through v
    bc = nx.betweenness_centrality(G, weight="trust")
    bc = dict(sorted(bc.items(), key=lambda item: item[1], reverse=True))

    # Degree centrality for a node v is the fraction of nodes it is connected to
    dc = nx.degree_centrality(G)
    dc = dict(sorted(dc.items(), key=lambda item: item[1], reverse=True))

    # Closeness centrality indicates how close a node is to other nodes in the network.
    # It is calculated as the average of the shortest path length from the node to
    # every other node in the network.
    cc = nx.closeness_centrality(G)
    cc = dict(sorted(cc.items(), key=lambda item: item[1], reverse=True))

    return {
        "betweenness": bc,
        "degree": dc,
        "closeness": cc,
    }
