"""1-tree lower bound on tour distance for the 28-node network (Section 5.11, Eq. 24).

LB = weight of a minimum spanning tree on the 27 non-depot nodes + the two shortest depot arcs.
Every tour contains a spanning path on the non-depot nodes plus two depot arcs, so no tour is
shorter than LB. Distances are symmetric. The bound concerns distance only, not the objective F.
"""
import os

import networkx as nx
import numpy as np
import pandas as pd

from gtst_common import DATA_DIR, save_json


def main():
    edges = pd.read_csv(os.path.join(DATA_DIR, "hcmc_edges_static.csv"))
    n = len(pd.read_csv(os.path.join(DATA_DIR, "hcmc_nodes.csv")))
    dist = np.zeros((n, n))
    dist[edges.source_node.values, edges.target_node.values] = edges.physical_distance_km.values
    g = nx.Graph()
    g.add_weighted_edges_from((i, j, dist[i, j]) for i in range(1, n) for j in range(i + 1, n))
    mst = sum(w for _, _, w in nx.minimum_spanning_tree(g).edges(data="weight"))
    d0 = np.sort(dist[0, 1:])
    lb = mst + d0[0] + d0[1]
    print(f"1-tree lower bound on tour distance: {lb:.2f} km")
    save_json({"mst_km": round(float(mst), 3), "two_shortest_depot_arcs_km": [float(d0[0]), float(d0[1])],
               "lower_bound_km": round(float(lb), 2)}, "distance_lower_bound.json")


if __name__ == "__main__":
    main()
