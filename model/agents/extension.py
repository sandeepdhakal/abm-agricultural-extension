# -*- coding: utf-8 -*-

"""Module to represent an extension officer."""

from collections import deque
from functools import lru_cache
from random import choices
from typing import Any, Protocol, Self, runtime_checkable

import networkx as nx
from shapely import Point, distance

from ..model import Model as Model
from .base import Agent
from .grower import Grower

MAX_DIST = 100_000


@lru_cache
def _distance_penalty(d: float, max_dist: float = MAX_DIST) -> float:
    return 1 - (max_dist - d) / max_dist


@runtime_checkable
class InteractionPolicy(Protocol):
    """Protocol for determining the nodes for an extension officer to interact with."""

    def next_nodes(self: Self, **kwargs: int) -> list[(str, float)]:
        """Next nodes to intearct with."""


def preference_ratio_for_control_method(model: Model, control_method: str) -> float:
    """Return preference among growers in the `model` for `control_method`."""
    grower_data = model.data["grower"]
    num_positive_growers = len(grower_data[grower_data["pcb_Biocontrol"] >= 0.5])  # noqa: PLR2004
    return num_positive_growers / len(grower_data)


@lru_cache
def shortest_path(G: nx.Graph, nodes_to_visit: tuple[str, ...]) -> [str]:
    """Return the shortest path for visiting `nodes_to_visit` in graph `G`."""
    # graph of nodes to visit
    # this should be a circular graph, with each edge weight corresponding to the
    # distance between the nodes.
    pG = nx.complete_graph(G.subgraph(nodes_to_visit))

    # calculate distance between edges in the graph
    for e in pG.edges:
        pG.edges[e]["weight"] = distance(
            Point(G.nodes[e[0]]["pos"]),
            Point(G.nodes[e[1]]["pos"]),
        )

    # now we calculate the tsp path
    tsp = nx.approximation.traveling_salesman_problem
    return tsp(pG, cycle=False)


class ExtensionOfficer(Agent):
    """An extension officer."""

    def __init__(self: Self, **attrs: dict[str, Any]) -> None:
        """Init extension officer."""
        Agent.__init__(self)

        if "interaction_policy" in attrs and isinstance(
            attrs["interaction_policy"],
            InteractionPolicy,
        ):
            self.interaction_policy = attrs["interaction_policy"]
        else:
            raise KeyError("'interaction_policy' required")

    def step(self: Self, **env_data: dict[str, Any]) -> None:
        """Take some actions for this timestep."""
        # interact with the growers determined by the interaction policy
        pref_ratio = preference_ratio_for_control_method(self.model, "pcb_Biocontrol")
        queue = self.interaction_policy.next_nodes(
            timestep=self.model.timestep,
            pref_ratio=pref_ratio,
        )
        gids = [q[0] for q in queue]
        times = [q[1] for q in queue]

        growers = self.model.agents(ids=gids)
        for grower, interaction_time in zip(growers, times, strict=True):
            self._update_grower_preference(grower, interaction_time)

    def _update_grower_preference(self: Self, grower: Grower, t: float) -> None:
        """Update the growers' preference to a control method."""
        # NOTE: currently assuming that all agents want to improve pref. for BioControl
        # TODO: make more configurable

        # NOTE: avoiding redundant calculation while we're assuming no negative effect
        old_pref = grower.control_method_preference[self.pest_control]
        if old_pref >= 1:
            return

        new_pref = _new_pref(old_pref, t, grower.propensity_to_change)
        grower.control_method_preference[self.pest_control] = new_pref


@lru_cache
def _new_pref(old_pref: float, t: float, propensity_to_change: float) -> float:
    new_pref = old_pref * (1 + (t**2) * propensity_to_change)
    return 1.0 if new_pref >= 1 else new_pref


class UniformInteraction:
    """Interaction policy in which ext officers interact equally with all growers.

    The extension officer prepares a list of growers to interact in one cycle, by
    randomly picking `percent_nodes` growers from the entire network. At the end of the
    cycle, this list is re-generated, which means it might be different from the
    previous list.
    """

    def __init__(
        self,
        network: nx.Graph,
        percent_nodes: int,
        time_per_node: float,
    ) -> None:
        """Init UniformInteraction.

        Args:
            network: The graph of growers for interaction.
            percent_nodes: The percent of nodes to interact with in one iteration.
            time_per_node: The time (/1) to be spent interacting with the grower.

        """
        self.network = network
        self.time_per_node = time_per_node

        self.num_nodes = round(len(self.network) * (percent_nodes / 100))
        self._gen_node_list()

    def _gen_node_list(self) -> None:
        nodes_to_visit = choices(list(self.network.nodes()), k=self.num_nodes)
        self.remaining_nodes = deque(
            nodes_to_visit
            if len(nodes_to_visit) == 1
            else shortest_path(self.network, tuple(nodes_to_visit)),
            maxlen=self.num_nodes,
        )

    def next_nodes(self, **kwargs: int) -> list[(str, float)]:
        """Return the nodes to interact with next.

        Returns:
            List of _(grower_id, interaction_time)_ tuples, with each tuple representing
            an interaction.
        """
        num_nodes = int(1 / self.time_per_node)
        ns = []
        last_loc = None
        for _ in range(num_nodes):
            if not self.remaining_nodes:
                self._gen_node_list()

            # apply distance penalty
            if last_loc is None:
                ns.append((self.remaining_nodes.popleft(), self.time_per_node))
            else:
                other_loc = self.network.nodes[self.remaining_nodes[0]]["pos"]
                dist = distance(Point(last_loc), Point(other_loc))
                penalty = _distance_penalty(dist)
                if penalty < self.time_per_node:
                    ns.append((self.remaining_nodes.popleft(), self.time_per_node))
            last_loc = self.network.nodes[ns[-1][0]]["pos"]
        return ns


class FocusedInteraction:
    """Interaction policy in which ext officers focus on central nodes."""

    def __init__(
        self,
        network: nx.Graph,
        percent_nodes: int,
        time_per_node: float,
    ) -> None:
        """Init FocusedInteraction.

        Args:
            network: The graph of growers for interaction.
            percent_nodes: The top percent of most central nodes to interact with.
            time_per_node: The time (/1) to be spent interacting with the grower.
        """
        self.network = network
        self.num_nodes = round(len(self.network) * (percent_nodes / 100))

        # determine degree centrality of nodes, sort and pick num_nodes
        deg_cent = nx.degree_centrality(self.network)
        deg_cent = [
            n[0] for n in sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)
        ]
        self.nodes = deg_cent[: self.num_nodes]

        self.remaining_nodes = deque(
            self.nodes
            if len(self.nodes) == 1
            else shortest_path(self.network, tuple(self.nodes)),
            maxlen=self.num_nodes,
        )
        self.time_per_node = time_per_node

    def next_nodes(self, **kwargs: int) -> list[str]:
        """Return the nodes to interact with next.

        Returns:
            List of _(grower_id, interaction_time)_ tuples, with each tuple representing
            an interaction.
        """
        num_nodes = int(1 / self.time_per_node)
        ns = []
        last_loc = None
        for _ in range(num_nodes):
            if not self.remaining_nodes:
                self.remaining_nodes.extend(self.nodes)

            # apply distance penalty
            if last_loc is None:
                ns.append((self.remaining_nodes.popleft(), self.time_per_node))
            else:
                other_loc = self.network.nodes[self.remaining_nodes[0]]["pos"]
                dist = distance(Point(last_loc), Point(other_loc))
                penalty = _distance_penalty(dist)
                if penalty < self.time_per_node:
                    ns.append((self.remaining_nodes.popleft(), self.time_per_node))
            last_loc = self.network.nodes[ns[-1][0]]["pos"]
        return ns
