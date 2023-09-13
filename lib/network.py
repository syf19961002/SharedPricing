# This file contains the `Network` class and functions that help construct the network object.

import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import copy
import graph_tool.all as gt
import time
import sys
import os

sys.path.append(os.getcwd() + "/..")


def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """

    # Deal with the value
    if isinstance(value, bool):
        tname = "bool"

    elif isinstance(value, int):
        tname = "float"
        value = float(value)

    elif isinstance(value, float):
        tname = "float"

    elif isinstance(value, dict):
        tname = "object"

    elif isinstance(value, list):
        tname = "list"

    else:
        tname = "string"
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname)  # Create the PropertyMap
        gtG.graph_properties[key] = prop  # Set the PropertyMap
        gtG.graph_properties[key] = value  # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set()  # cache keys to only add properties once
    for node in nxG.nodes():
        data = nxG.nodes[node]

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname)  # Create the PropertyMap
            gtG.vertex_properties[key] = prop  # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties["id"] = gtG.new_vertex_property("string")

    # Add the edge properties second
    eprops = set()  # cache keys to only add properties once
    for src, dst in nxG.edges():
        data = nxG.edges[src, dst]

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname)  # Create the PropertyMap
            gtG.edge_properties[key] = prop  # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {}  # vertex mapping for tracking edges later
    bad_node_keys = set()
    for node in nxG.nodes():
        data = nxG.nodes[node]

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data["id"] = str(node)
        for key, value in data.items():
            try:
                gtG.vp[key][v] = value  # vp is short for vertex_properties
            except:
                bad_node_keys.add(key)

    # Add the edges
    bad_edge_keys = set()
    for src, dst in nxG.edges():
        data = nxG.edges[src, dst]

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            try:
                gtG.ep[key][e] = value  # ep is short for edge_properties
            except:
                bad_edge_keys.add(key)

    if len(bad_node_keys) + len(bad_edge_keys) > 0:
        print("Failure on properties", bad_node_keys, bad_edge_keys)

    return gtG


class Network(object):
    """
    Represents a routing network as a graph.
    """

    def __init__(self, network, positions, one_way):
        self.G = network
        self.G2 = nx2gt(self.G)
        if "crs" not in self.G.graph:
            self.G.graph["crs"] = ""  # needed to work with osmnx
        self.one_way = one_way
        self.positions = positions

    def draw(self, size=5, edge_color="#999999", od=None, route_color="#ff0000"):
        "draws the network, for some reason needs two tries to work"
        maxX = max(p[0] for p in self.positions)
        maxY = max(p[1] for p in self.positions)
        minX = min(p[0] for p in self.positions)
        minY = min(p[1] for p in self.positions)
        xspan = max(1, maxX - minX)
        yspan = max(1, maxY - minY)
        if xspan < yspan:
            fig = plt.figure(figsize=(size * xspan / yspan, size))
        else:
            fig = plt.figure(figsize=(size, size * yspan / xspan))
        fig.patch.set_visible(False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        detail = len(self.G.nodes) <= 200
        if detail:
            nx.draw_networkx_nodes(
                self.G, pos=self.positions, node_size=50, node_color="#999999"
            )
        if type(edge_color) == str or self.one_way or not detail:
            edge_alpha = 1
        else:
            edge_alpha = 0.5
        if detail:
            arrow_style = "-|>"
        else:
            arrow_style = "-"
        nx.draw_networkx_edges(
            self.G,
            pos=self.positions,
            edge_color=edge_color,
            alpha=edge_alpha,
            arrowstyle=arrow_style,
        )
        if od != None:
            path = self.shortest_path(od[0], od[1])
            nx.draw_networkx_edges(
                self.G,
                pos=self.positions,
                edgelist=[(path[k], path[k + 1]) for k in range(len(path) - 1)],
                edge_color=route_color,
            )

    def draw_requests(
        self,
        size=5,
        edge_color="#999999",
        ods=None,
        route_color=["red", "green", "blue"],
    ):
        "draws the network, for some reason needs two tries to work"
        maxX = max(p[0] for p in self.positions)
        maxY = max(p[1] for p in self.positions)
        minX = min(p[0] for p in self.positions)
        minY = min(p[1] for p in self.positions)
        xspan = max(1, maxX - minX)
        yspan = max(1, maxY - minY)
        if xspan < yspan:
            fig = plt.figure(figsize=(size * xspan / yspan, size))
        else:
            fig = plt.figure(figsize=(size, size * yspan / xspan))
        fig.patch.set_visible(False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        detail = len(self.G.nodes) <= 200
        if detail:
            nx.draw_networkx_nodes(
                self.G, pos=self.positions, node_size=50, node_color="#999999"
            )
        if type(edge_color) == str or self.one_way or not detail:
            edge_alpha = 1
        else:
            edge_alpha = 0.5
        if detail:
            arrow_style = "-|>"
        else:
            arrow_style = "-"
        nx.draw_networkx_edges(
            self.G,
            pos=self.positions,
            edge_color=edge_color,
            alpha=edge_alpha,
            arrowstyle=arrow_style,
        )
        for i in range(len(ods)):
            od = ods[i]
            if od != None:
                path = self.shortest_path(od[0], od[1])
                nx.draw_networkx_edges(
                    self.G,
                    pos=self.positions,
                    edgelist=[(path[k], path[k + 1]) for k in range(len(path) - 1)],
                    edge_color=route_color[i],
                )

    def calculate_shortest_times(self):
        if not hasattr(self, "shortest_time_matrix"):
            print("Calculating shortest time matrix...")
            t0 = time.time()
            # Get travel times
            weights = self.G2.new_edge_property("float")
            for e in self.G2.edges():
                weights[e] = float(self.G.edges[e]["travel_time"])
            V = len(self.G.nodes)
            nodes = range(V)
            shortest_time_matrix = np.zeros((V, V))
            for u in nodes:
                shortest_time_matrix[u, :] = gt.shortest_distance(
                    self.G2, source=u, target=nodes, weights=weights, pred_map=False
                )
            self.shortest_time_matrix = shortest_time_matrix
            self.shortest_time_paths = [dict() for u in range(V)]
            print(
                "... Done calculating shortest time matrix,",
                time.time() - t0,
                "seconds",
            )

    def shortest_path(self, u, v):
        self.calculate_shortest_times()
        if v not in self.shortest_time_paths[u]:
            self.shortest_time_paths[u][v] = nx.shortest_path(
                self.G, source=u, target=v, weight="travel_time"
            )
        return self.shortest_time_paths[u][v]


def osm(G, edge_keys):
    """
    Returns a simplified network from OSM graph
    """
    G = nx.convert_node_labels_to_integers(G)
    positions = [(G.nodes[u]["x"], G.nodes[u]["y"]) for u in G.nodes()]
    G2 = nx.DiGraph()
    G2.add_nodes_from(range(len(positions)))
    for u in range(len(positions)):
        for attr in G.nodes[u]:
            G2.nodes[u][attr] = G.nodes[u][attr]
    for u, v in G.edges():
        G2.add_edge(u, v)
        k = edge_keys[u, v]
        for attr in G.edges[u, v, k]:
            G2.edges[u, v][attr] = G.edges[u, v, k][attr]
    G2.graph["crs"] = G.graph["crs"]
    return Network(G2, positions, False)
