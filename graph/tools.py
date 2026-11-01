"""
Graph construction utilities.

Builds a multi-layer hierarchical adjacency structure for a skeleton graph.
Joints are partitioned into concentric "rings" by hop-distance from a chosen
center-of-mass (CoM) joint. Each pair of adjacent rings forms one hierarchy
layer, and within each layer three directional sub-relations are built:
identity (self), forward (inner ring -> outer ring), and reverse (outer
ring -> inner ring).
"""

import numpy as np


def edge2mat(link, num_node):
    """Convert a list of (target, source) joint-index pairs into a binary
    adjacency matrix of shape (num_node, num_node)."""
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    """Column-normalize a directed adjacency matrix (in-degree normalization)."""
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_groups(CoM=0):
    """
    Return the concentric-ring partition of the 16-joint E-Gait skeleton,
    ordered outward from the chosen center-of-mass (CoM) joint.

    Joint order:
      0=Root, 1=Spine, 2=Neck, 3=Head,
      4=LeftShoulder, 5=LeftElbow, 6=LeftHand,
      7=RightShoulder, 8=RightElbow, 9=RightHand,
      10=LeftHip, 11=LeftKnee, 12=LeftFoot,
      13=RightHip, 14=RightKnee, 15=RightFoot

    Groups are given 1-indexed (shifted to 0-indexed internally by
    get_edgeset), matching the convention used throughout this module.
    """
    groups = []

    if CoM == 0:
        groups.append([1])
        groups.append([2, 11, 14])
        groups.append([3, 5, 8, 12, 15])
        groups.append([4, 6, 9, 13, 16])
        groups.append([7, 10])

    elif CoM == 1:
        groups.append([2])
        groups.append([1, 3, 5, 8])
        groups.append([11, 14, 4, 6, 9])
        groups.append([12, 15, 7, 10])
        groups.append([13, 16])

    elif CoM == 2:
        groups.append([3])
        groups.append([2, 4])
        groups.append([1, 5, 8])
        groups.append([11, 14, 6, 9])
        groups.append([12, 15, 7, 10])
        groups.append([13, 16])

    else:
        raise ValueError(f"Unsupported CoM index: {CoM}")

    return groups


def get_edgeset(CoM=0):
    """
    Build the per-hierarchy-layer edge sets: for every pair of adjacent
    rings, construct identity / forward / reverse joint-pair lists.
    """
    groups = get_groups(CoM=CoM)

    for i, group in enumerate(groups):
        group = [i - 1 for i in group]
        groups[i] = group

    identity = []
    forward_hierarchy = []
    reverse_hierarchy = []

    for i in range(len(groups) - 1):
        self_link = groups[i] + groups[i + 1]
        self_link = [(i, i) for i in self_link]
        identity.append(self_link)

        forward_g = []
        for j in groups[i]:
            for k in groups[i + 1]:
                forward_g.append((j, k))
        forward_hierarchy.append(forward_g)

        reverse_g = []
        for j in groups[-1 - i]:
            for k in groups[-2 - i]:
                reverse_g.append((j, k))
        reverse_hierarchy.append(reverse_g)

    edges = []
    for i in range(len(groups) - 1):
        edges.append([identity[i], forward_hierarchy[i], reverse_hierarchy[-1 - i]])

    return edges


def get_graph(num_node, edges):
    """Turn one hierarchy layer's three edge-sets into a (3, V, V) matrix stack."""
    I = edge2mat(edges[0], num_node)
    Forward = normalize_digraph(edge2mat(edges[1], num_node))
    Reverse = normalize_digraph(edge2mat(edges[2], num_node))
    A = np.stack((I, Forward, Reverse))
    return A


def get_hierarchical_graph(num_node, edges):
    """Stack every hierarchy layer's (3, V, V) matrix into (num_layers, 3, V, V)."""
    A = []
    for edge in edges:
        A.append(get_graph(num_node, edge))
    A = np.stack(A)
    return A
