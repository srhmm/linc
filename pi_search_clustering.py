import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, LpInteger, GUROBI, CPLEX_PY, CPLEX_CMD, LpSolverDefault
import itertools

from sklearn.cluster import AgglomerativeClustering

from pi_search_mdl import pi_mdl
from utils_pi import pi_map_to_pi

"""Help functions for ILP-based search for a partition Pi of contexts into groups"""


def pi_search_clustering(pair_distances, c_n, gpr, regression_per_group, subsample_size):

    """ Formulate partition search as a kmeans problem and solve it.
    Objective: find the clustering and number of clusters that minimizes pairwise distances.

    :param pair_distances: dist(i,j) for (i,j) in itertools.combinations(contexts, 2).
        MDL gain of using a joint model M_ij for both contexts cmp. to separate models M_i, M_j
    :param c_n: number of contexts
    :return: pistar, pistar_mdl, pistar_model, pistar_data, pistar_penalty
    """
    contexts = range(c_n)
    dists_ij = np.array([[0 for i in range(c_n)] for j in range(c_n)])
    comb_i = -1

    for pair in itertools.combinations(contexts, 2):
        comb_i = comb_i + 1
        i, j = pair[0], pair[1]
        dists_ij[i][j] = pair_distances[comb_i].astype(np.int64)

    pistar, pistar_mdl, pistar_model, pistar_data, pistar_penalty = np.inf, np.inf, np.inf, np.inf, np.inf
    for k in range(2, c_n):
        pi_at_k = pi_search_clustering_k(dists_ij, k)
        mdl, mdl_model, mdl_data, mdl_penalty, _ = pi_mdl(gpr, pi_at_k, regression_per_group, subsample_size)
        if mdl < pistar_mdl:
            pistar = pi_at_k
            pistar_mdl = mdl
            pistar_model = mdl_model
            pistar_data = mdl_data
            pistar_penalty = mdl_penalty

    return pistar, pistar_mdl, pistar_model, pistar_data, pistar_penalty


def pi_search_clustering_k(D, k):
    agg = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
    agg.fit_predict(D)
    pi = pi_map_to_pi(agg.labels_)

    return pi