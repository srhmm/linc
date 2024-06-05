import itertools
from typing import List

import causaldag as cd
import numpy as np
from cdt.metrics import SHD, SID

from gpcd.util.upq import UPQ
from .util import is_insignificant


def dag_n_edges(adj):
    assert adj.shape[0] == adj.shape[1]
    return sum([len(np.where(adj[i] != 0)[0]) for i in range(len(adj))])
    # return sum(
    #    [sum([1 if (adj[i][j] != 0) else 0 for j in range(len(adj[i]))]) for i in range(len(adj))])


def mec_from_dag(dag: cd.DAG):
    """Returns all dags in the Markov Equivalence class of a dag"""
    cpdag = dag.cpdag()
    adj = cpdag.to_amat()[0]
    mec = cpdag2dags(adj)
    return mec


def cpdag2dags(adj):
    """Converts a cpdag adjacency matrix to a list of all dags"""
    adj = np.asarray(adj)
    dags_elist = list(cd.PDAG().from_amat(adj).all_dags())
    dags = []
    for elist in dags_elist:
        G = np.zeros(adj.shape)
        elist = np.asarray(list(elist))
        if len(elist) > 0:
            G[elist[:, 0], elist[:, 1]] = 1
        dags.append(G)
    return dags


def directional_f1(true_dag, test_dag):
    tp = sum(
        [
            sum(
                [
                    1 if (test_dag[i][j] != 0 and true_dag[i][j] != 0) else 0
                    for j in range(len(true_dag[i]))
                ]
            )
            for i in range(len(true_dag))
        ]
    )
    tn = sum(
        [
            sum(
                [
                    1 if (test_dag[i][j] == 0 and true_dag[i][j] == 0) else 0
                    for j in range(len(true_dag[i]))
                ]
            )
            for i in range(len(true_dag))
        ]
    )
    fp = sum(
        [
            sum(
                [
                    1 if (test_dag[i][j] != 0 and true_dag[i][j] == 0) else 0
                    for j in range(len(true_dag[i]))
                ]
            )
            for i in range(len(true_dag))
        ]
    )
    fn = sum(
        [
            sum(
                [
                    1 if (test_dag[i][j] == 0 and true_dag[i][j] != 0) else 0
                    for j in range(len(true_dag[i]))
                ]
            )
            for i in range(len(true_dag))
        ]
    )
    den = tp + 1 / 2 * (fp + fn)
    if den > 0:
        f1 = tp / den
    else:
        f1 = 1
    return f1, tp, tn, fn, fp


def match_dags(true_dag, test_dag):
    return not False in [
        (
            not False
            in [
                (
                    (test_dag[i][j] == 0 and true_dag[i][j] == 0)
                    or (test_dag[i][j] != 0 and true_dag[i][j] != 0)
                )
                for j in range(len(test_dag[i]))
            ]
        )
        for i in range(len(test_dag))
    ]


def gen_test_dags(
    true_dag: cd.DAG,
    true_edge_p: int,
    oracle_mec: bool,
    oracle_nedges: bool,
    random_size: int,
    edge_prbs: List,
):
    """Generates DAGs to test, including the true DAG

    Args:
        true_dag: true DAG
        oracle_mec: whether DAGs should be sampled only from the MEC or at random
        oracle_nedges: whether DAGs should be sampled with the true num edges or at random
        random_size: how many random DAGs should be included per edge probability
        edge_prbs: edge probabilities to be considered

    Returns: list of test dags

    """
    assert not (oracle_mec and oracle_nedges)
    N = len(true_dag.to_amat())
    true_edges = dag_n_edges(true_dag.to_amat())

    if oracle_mec:
        # all DAGs in the MEC
        test_dags = mec_from_dag(true_dag)
    # sample random DAGs
    elif oracle_nedges:
        test_dags = [true_dag.to_amat()]
        for _ in range(random_size):
            found = False
            for _ in range(1000):
                arcs = cd.rand.directed_erdos(N, true_edge_p)
                dag = cd.rand.rand_weights(arcs)
                if dag_n_edges(dag.to_amat()) != true_edges:
                    continue
                else:
                    test_dags.append(dag.to_amat())
                    found = True
                    break
        if not found:
            raise Warning(f"not enough DAGs with {true_edges} found")
    else:
        test_dags = [true_dag.to_amat()]
        for edge_prb in edge_prbs:
            for _ in range(random_size):
                arcs = cd.rand.directed_erdos(len(true_dag.to_amat()), edge_prb)
                dag = cd.rand.rand_weights(arcs)
                test_dags.append(dag.to_amat())
        """
        for n_edges in range(N**2):
            for _ in range(random_size):

                found=False
                for edge_p in edge_prbs:  #todo set edge_p directly?
                    for _ in range(100):
                        arcs = cd.rand.directed_erdos(N, edge_p)
                        dag = cd.rand.rand_weights(arcs)
                        if dag_n_edges(dag.to_amat())!=true_edges:
                            continue
                        else:
                            test_dags.append(dag.to_amat())
                            found=True
                            break
                if not found: 
                    raise Warning(f'not enough DAGs with {n_edges} found')
        """
    return test_dags


def gen_dags_from_queue(dag_search, q: UPQ):
    test_dags = []
    edges = q.all_entries
    # for pi_edge in edges.values():
    #    node, parent = pi_edge.j, pi_edge.i
    # gain, score, pa, score_cur, pa_cur = dag_search.eval_edge_addition(node, parent)
    # assert not is_insignificant(gain)
    for i in range(1, len(edges) + 1):
        edge_combos = itertools.combinations(edges.values(), i)
        for combo in edge_combos:
            for edge in combo:
                node, parent = edge.j, edge.i
                # Check whether adding the edge would result in a cycle
                if dag_search.has_cycle(parent, node):
                    continue
                gain, score, pa, score_cur, pa_cur = dag_search.eval_edge_addition(
                    node, parent
                )

                # todo Optional to speed things up: Check whether gain is significant
                if is_insignificant(gain):
                    continue

                dag_search.add_edge(
                    parent, node, score
                )  # want true_adj[parent][node] > 0
            test_dags.append(dag_search.get_adj())
            dag_search.remove_all_edges()
    return test_dags


def eval_old_jpcmci_links(true_dag, jpcmci_links, N, enable_SID_call):
    assert true_dag.shape[1] == N

    undir_t = int(
        sum(
            [
                sum(
                    [
                        (
                            1
                            if (
                                jpcmci_links[i][j][0]
                                and jpcmci_links[j][i][0]
                                and (true_dag[i][j] != 0 or true_dag[j][i] != 0)
                            )
                            else 0
                        )
                        for j in range(N)
                    ]
                )
                for i in range(N)
            ]
        )
        / 2
    )

    undir_f = int(
        sum(
            [
                sum(
                    [
                        (
                            1
                            if (
                                jpcmci_links[i][j][0]
                                and jpcmci_links[j][i][0]
                                and true_dag[i][j] == 0
                                and true_dag[j][i] == 0
                            )
                            else 0
                        )
                        for j in range(N)
                    ]
                )
                for i in range(N)
            ]
        )
        / 2
    )

    jpcmci_dag = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if jpcmci_links[i][j][0]:
                jpcmci_dag[i][j] = 1

    tp = sum(
        [
            sum(
                [
                    (
                        1
                        if (
                            jpcmci_links[i][j][
                                0
                            ]  # or stricter with> and (not jpcmci_links[j][i][0])
                            and true_dag[i][j] != 0
                        )
                        else 0
                    )
                    for j in range(N)
                ]
            )
            for i in range(N)
        ]
    )
    tn = sum(
        [
            sum(
                [
                    1 if ((not jpcmci_links[i][j][0]) and true_dag[i][j] == 0) else 0
                    for j in range(N)
                ]
            )
            for i in range(N)
        ]
    )
    fp = sum(
        [
            sum(
                [
                    (
                        1
                        if (
                            jpcmci_links[i][j][0]  # count bidirectional edges as FP
                            and true_dag[i][j] == 0
                        )
                        else 0
                    )
                    for j in range(N)
                ]
            )
            for i in range(N)
        ]
    )
    fn = sum(
        [
            sum(
                [
                    1 if ((not jpcmci_links[i][j][0]) and true_dag[i][j] != 0) else 0
                    for j in range(N)
                ]
            )
            for i in range(N)
        ]
    )
    den = tp + 1 / 2 * (fp + fn)
    if den > 0:
        f1 = tp / den
    else:
        f1 = 1
    tpr = 0 if (tp + fn == 0) else tp / (tp + fn)
    fpr = 0 if (tn + fp == 0) else fp / (tn + fp)
    fdr = 0 if (tp + fp == 0) else fp / (tp + fp)
    true_ones = np.array(
        [[1 if i != 0 else 0 for i in true_dag[j]] for j in range(len(true_dag))]
    )
    res_ones = np.array(
        [[1 if i != 0 else 0 for i in jpcmci_dag[j]] for j in range(len(jpcmci_dag))]
    )

    shd = SHD(true_ones, jpcmci_dag)
    if enable_SID_call:
        sid = np.float(SID(true_ones, jpcmci_dag))
    else:
        sid = 1  # call to R script currently not working

    checksum = tp + tn + fn + fp
    assert checksum == N**2
    return shd, sid, f1, tpr, fpr, fdr, tp, tn, fn, fp, undir_t, undir_f


def eval_dags(true_dag, result, N, enable_SID_call):
    assert true_dag.shape[0] == N and true_dag.shape[1] == N
    if len(result) == 0:
        sum_gain, res_dag = 0, np.zeros((N, N))
    else:
        result = sorted(result.items(), key=lambda item: item[1][0])
        sum_gain, res_dag = result[0][1]

    return eval_dag(true_dag, res_dag, N, enable_SID_call)


def eval_dag(true_dag, res_dag, N, enable_SID_call):
    assert true_dag.shape[1] == res_dag.shape[1] and res_dag.shape[1] == N
    assert true_dag.shape[0] == res_dag.shape[0]

    f1, tp, tn, fn, fp = directional_f1(true_dag, res_dag)

    tpr = 0 if (tp + fn == 0) else tp / (tp + fn)
    fpr = 0 if (tn + fp == 0) else fp / (tn + fp)
    fdr = 0 if (tp + fp == 0) else fp / (tp + fp)

    acc = match_dags(true_dag, res_dag)

    true_ones = np.array(
        [[1 if i != 0 else 0 for i in true_dag[j]] for j in range(len(true_dag))]
    )
    shd = SHD(true_ones, res_dag)
    if enable_SID_call:
        sid = np.float(SID(true_ones, res_dag))
    else:
        sid = 1  # call to R script currently not working

    checksum = tp + tn + fn + fp
    assert checksum == true_dag.shape[0] * true_dag.shape[1]

    return acc, shd, sid, f1, tpr, fpr, fdr, tp, tn, fn, fp
