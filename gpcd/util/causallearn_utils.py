"""Tools for other functions and methods"""

import numpy as np
from causaldag import DAG, PDAG


def check_2d(X):
    if X is not None and X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def dag_true_orientations(true_dag, cpdag):
    """Number of correctly oriented edges / number of edges"""
    # np.gen.py.assert_array_equal(true_dag, np.tril(true_dag))
    tp = len(np.where((true_dag + cpdag - cpdag.T) == 2)[0])
    n_edges = np.sum(true_dag)
    return tp / n_edges


def dag_false_orientations(true_dag, cpdag):
    """Number of falsely oriented edges / number of edges"""
    # np.gen.py.assert_array_equal(true_dag, np.tril(true_dag))
    fp = len(np.where((true_dag + cpdag.T - cpdag) == 2)[0])
    n_edges = np.sum(true_dag)
    return fp / n_edges


def dag_precision(true_dag, cpdag):
    tp = len(np.where((true_dag + cpdag - cpdag.T) == 2)[0])
    fp = len(np.where((true_dag + cpdag.T - cpdag) == 2)[0])
    return tp / (tp + fp) if (tp + fp) > 0 else 1


def dag_recall(true_dag, cpdag):
    tp = len(np.where((true_dag + cpdag - cpdag.T) == 2)[0])
    return tp / np.sum(true_dag)


def average_precision_score(true_dag, pvalues_mat):
    """
    Computes average precision score from pvalue thresholds
    """

    thresholds = np.unique(pvalues_mat)
    dags = np.asarray(cpdag2dags(dag2cpdag(true_dag)))

    # ap_score = 0
    # prior_recall = 0

    precisions = []
    recalls = []

    for t in thresholds:
        axis = tuple(np.arange(1, pvalues_mat.ndim))
        n_changes = np.sum(pvalues_mat <= t, axis=axis) / 2
        min_idx = np.where(n_changes == np.min(n_changes))[0]
        cpdag = (np.sum(dags[min_idx], axis=0) > 0).astype(int)
        precisions.append(dag_precision(true_dag, cpdag))
        recalls.append(dag_recall(true_dag, cpdag))

        # ap_score += precision * (recall - prior_recall)
        # prior_recall = recall

    # if len(thresholds) == 1:
    #     ap_score = precisions[0] * recalls[0]
    # else:
    sort_idx = np.argsort(recalls)
    recalls = np.asarray(recalls)[sort_idx]
    precisions = np.asarray(precisions)[sort_idx]
    ap_score = (np.diff(recalls, prepend=0) * precisions).sum()

    return ap_score


def dags2mechanisms(dags):
    """
    Returns a dictionary of variable: mechanisms from
    a list of DAGs.
    """
    m = len(dags[0])
    mech_dict = {i: [] for i in range(m)}
    for dag in dags:
        for i, mech in enumerate(dag.T):  # Transpose to get parents
            mech_dict[i].append(mech)

    # remove duplicates
    for i in range(m):
        mech_dict[i] = np.unique(mech_dict[i], axis=0)

    return mech_dict


def create_causal_learn_dag(G):
    """Converts directed adj matrix G to causal graph"""
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode

    n_vars = G.shape[0]
    node_names = [("X%d" % (i + 1)) for i in range(n_vars)]
    nodes = [GraphNode(name) for name in node_names]

    cl_dag = Dag(nodes)
    for i in range(n_vars):
        for j in range(n_vars):
            if G[i, j] != 0:
                cl_dag.add_directed_edge(nodes[i], nodes[j])

    return cl_dag


def create_causal_learn_cpdag(G):
    """Converts adj mat of cpdag to a causal learn graph object"""
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode

    n_vars = G.shape[0]
    node_names = [("X%d" % (i + 1)) for i in range(n_vars)]
    nodes = [GraphNode(name) for name in node_names]

    cl_cpdag = GeneralGraph(nodes)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if G[i, j] == 1 and G[j, i] == 1:
                cl_cpdag.add_edge(
                    Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL)
                )
            elif G[i, j] == 1 and G[j, i] == 0:
                cl_cpdag.add_edge(
                    Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                )
            elif G[i, j] == 0 and G[j, i] == 1:
                cl_cpdag.add_edge(
                    Edge(nodes[i], nodes[j], Endpoint.ARROW, Endpoint.TAIL)
                )

    return cl_cpdag


def dag2cpdag(adj, targets=None):
    """Converts an adjacency matrix to the cpdag adjacency matrix, with potential interventions"""
    dag = DAG().from_amat(adj)
    cpdag = dag.cpdag()
    if targets is None:
        return cpdag.to_amat()[0]
    else:
        return dag.interventional_cpdag([targets], cpdag=cpdag).to_amat()[0]


def cpdag2dags(adj):
    """Converts a cpdag adjacency matrix to a list of all dags"""
    adj = np.asarray(adj)
    dags_elist = list(PDAG().from_amat(adj).all_dags())
    dags = []
    for elist in dags_elist:
        G = np.zeros(adj.shape)
        elist = np.asarray(list(elist))
        if len(elist) > 0:
            G[elist[:, 0], elist[:, 1]] = 1
        dags.append(G)

    return dags


"""
Useful causal-learn utils for reference

# Orients edges in a pdag, to find a dag (not necessarily possible)
causallearn.utils.PDAG2DAG import pdag2dag

# Returns the CPDAG of a DAG (the MEC!)
from causallearn.utils.DAG2CPDAG import dag2cpdag

# Checks if two dags are in the same MEC
from causallearn.utils.MECCheck import mec_check

# Runs meek's orientation rules over a DAG, with optional background
# knowledge. definite_meek examines definite unshielded triples
from causallearn.utils.PCUtils.Meek import meek, definite_meek

"""
