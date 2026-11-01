"""
E-Gait skeleton graph definition.

Wraps the hierarchical adjacency construction from graph.tools into a single
Graph object, instantiated once and passed into the model.
"""

import sys
import numpy as np

sys.path.extend(["../"])
from graph import tools

num_node = 16


class Graph:
    def __init__(self, CoM=0, labeling_mode="spatial"):
        self.num_node = num_node
        self.CoM = CoM
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            edges = tools.get_edgeset(CoM=self.CoM)
            A = tools.get_hierarchical_graph(num_node, edges)
        else:
            raise ValueError(f"Unsupported labeling mode: {labeling_mode}")
        return A, self.CoM
