import numpy as np
import matplotlib.pyplot as plt
from mds import *

from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.metrics.pairwise import euclidean_distances

def create_adjacency(data, k=3):
    """
    creates a graph by connecting each instance to its k nearest neighbors
    """
    n, m = data.shape
    dist = euclidean_distances(data)
    sorted_indexes = np.argsort(dist, axis=1)[:,:k]
    adj = np.zeros((n, n)) + np.inf
    for i in range(n):
        adj[i, sorted_indexes[i]] = dist[i, sorted_indexes[i]]
    short = graph_shortest_path(adj, method="FW")
    return short


def isomap(data, dim=2, k=3):
    """
    Reduces dimensionality while trying to preserve the geodesic distances between
    the instances.
    """
    # compute geodisc distances
    d = create_adjacency(data, k)
    # square of the distances
    D = np.square(d)
    # double centring of D
    S = double_centring(D)
    # applying mds
    X = mds(S, dim)
    return X

def evaluate_isomap(df, red_df, k=3):
    # compute geodisc distances
    d_adj = create_adjacency(df, k)
    d_red = create_adjacency(red_df, k)
    # square of the distances
    D_adj = np.square(d_adj)
    D_red = np.square(d_red)
    # double centring of D
    S_adj = double_centring(D_adj)
    S_red = double_centring(D_red)
    return MSE(D_adj, D_red)


def MSE(A, B):
    return ((A-B)**2).mean()
