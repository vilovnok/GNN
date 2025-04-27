import numpy as np
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities

def negative_log_likelihood(pred, labels):
    return -np.log(pred)[np.arange(pred.shape[0]), np.argmax(labels, axis=1)]

def glorot_init(nin, nout):
    sd = np.sqrt(6.0 / (nin + nout))
    return np.random.uniform(-sd, sd, size=(nin, nout))


def load_data():
    g = nx.karate_club_graph()
    colors = np.zeros(g.number_of_nodes())
    communities = greedy_modularity_communities(g)

    for i, com in enumerate(communities):
        colors[list(com)] = i

    n_classes = np.unique(colors).shape[0]
    labels = np.eye(n_classes)[colors.astype(int)]

    adj_matrix = nx.to_numpy_array(g)
    X = np.eye(g.number_of_nodes())

    return g, n_classes, labels, adj_matrix, X, colors