import itertools
import logging
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy import ndarray

from exp.exptypes import ExpType
from gpcd.dag import DAG
from gpcd.other.util import is_insignificant
from gpcd.other.util_dag import cpdag2dags
from gpcd.scoring import GPParams
from gpcd.util.causallearn_utils import dag2cpdag
from gpcd.util.upq import UPQ


def dag_search(
    exp_type: ExpType,
    truths: SimpleNamespace,
    data_c: dict,
    gp_hyperparams: GPParams,
    logger: logging.Logger,
    verbosity: int,
    is_true_edge: object = lambda i: lambda j: "",
) -> tuple[ndarray, DAG]:
    """
    Search over DAG minimising GP score
    :param exp_type: MEC or greedy
    :param data_c: data in each context, dict {'c0': data0, 'c1': data1 ...}
    :param gp_hyperparams: params in GPParams
    :param logger: logging
    :param verbosity: verbosity
    :param is_true_edge: for gen.py
    :return: adjacency, DAG object (with no edges but all scores computed during search)
    """
    if exp_type == ExpType.MEC:
        dags = cpdag2dags(dag2cpdag(truths.dag))
        return exhaustive_search(
            dags, data_c, gp_hyperparams, logger, verbosity, is_true_edge
        )
    elif exp_type == ExpType.DAG:
        return greedy_dag_search(
            data_c, gp_hyperparams, logger, verbosity, is_true_edge
        )
    else:
        raise ValueError(exp_type)


def exhaustive_search(
    candidate_dags: list[np.ndarray[Any, Any]],
    data_c: dict[np.ndarray[float, float]],
    gp_hyperparams: GPParams,
    logger: logging.Logger,
    verbosity: int,
    is_true_edge=lambda i: lambda j: "",
) -> tuple[ndarray, DAG]:
    """
    Exhaustive search over candidate causal DAGs
    :param candidate_dags: list of adjs, e.g. MEC
    :param data_c: data in each context, dict {'c0': data0, 'c1': data1 ...}
    :param gp_hyperparams: params in GPParams
    :param logger: logging
    :param verbosity: verbosity
    :param is_true_edge: for gen.py
    :return: adjacency, DAG object (with no edges but all scores computed during search)
    """

    logger.info("\n*** MEC Search ***")
    dag_s = DAG(data_c, gp_hyperparams, logger, verbosity, is_true_edge)
    dag_scores = {}
    for i, adj in enumerate(candidate_dags):
        logger.info(f"Scoring Dag {i}" + r"/" + f"{len(candidate_dags)+1}")
        dag_scores[i] = dag_s.eval_other_dag(
            adj, rev=True
        )  # rev: convention whether adj[i][j] means i->j or vice versa

    min_idx = min(dag_scores, key=dag_scores.get)
    min_arg = candidate_dags[min_idx]
    return min_arg, dag_s


def greedy_dag_search(
    data_c: dict,
    gp_hyperparams: GPParams,
    logger: logging,
    verbosity: int,
    is_true_edge=lambda i: lambda j: "",
) -> tuple[ndarray, DAG]:
    """
    Greedy tree search for causal DAGs
    :param data_c: data in each context, dict {'c0': data0, 'c1': data1 ...}
    :param gp_hyperparams: params in GPParams
    :param logger: logging
    :param verbosity: verbosity
    :param is_true_edge: for gen.py
    :return: adjacency, DAG object (with edges as in adjacency and all scores computed during search)
    """
    if verbosity > 0:
        logger.info("\n*** DAG Search ***")

    q = UPQ()
    dag_s = DAG(data_c, gp_hyperparams, logger, verbosity, is_true_edge)
    q = dag_s.initial_edges(q)

    q, dag_s = _dag_forward_phase(q, dag_s, logger, verbosity - 1)
    q, dag_s = _dag_backward_phase(q, dag_s, logger, verbosity - 1)

    if verbosity > 0:
        logger.info(f"Result:")
        dag_s.info_adj()
    return dag_s.get_adj(), dag_s


def _dag_forward_phase(
    q: UPQ, dag_search: DAG, logger: logging.Logger, verbosity: int
) -> (UPQ, DAG):
    st = time.perf_counter()
    if verbosity > 0:
        logger.info("Forward Phase ...")

    while q.pq:
        try:
            pi_edge = q.pop_task()
            node, parent = pi_edge.j, pi_edge.pa

            # Check whether adding the edge would result in a cycle
            if dag_search.has_cycle(parent, node):
                continue
            if dag_search.exists_anticausal_edge(parent, node):  # todo refine
                if verbosity > 0:
                    logger.info(
                        f"\tSkip cycle {parent} -> {node}, existing edge(s) {dag_search.parents_of(parent)} -> {parent} \t{dag_search.is_true_edge(parent)(node)}"
                    )
                continue
            gain, score, pa, score_cur, pa_cur = dag_search.eval_edge_addition(
                node, parent
            )

            # Check whether gain is significant
            if is_insignificant(gain):
                # if verbosity > 0:
                #    print(
                #    f'\tSkip edge {parent} -> {node}: s={np.round(gain[0][0], 2)} pa={dag_search.parents_of(node)} \t{dag_search.is_true_edge(parent)(node)}')
                continue

            dag_search.add_edge(
                parent, node, score, gain
            )  # want true_adj[parent][node] > 0

            # Reconsider children under current model and remove if reversing the edge improves score
            for ch in dag_search._nodes:
                if not dag_search.is_edge(node, ch):
                    continue
                gain = dag_search.eval_edge_flip(node, ch)

                if not is_insignificant(gain):
                    # Remove the edge, update the gain of both edges
                    dag_search.remove_edge(node, ch)

                    edge_fw = dag_search.pair_edges[node][ch]
                    edge_bw = dag_search.pair_edges[ch][node]
                    assert edge_fw.i == node and edge_fw.j == ch
                    assert edge_bw.i == ch and edge_bw.j == node

                    assert not (q.exists_task(edge_fw))
                    if q.exists_task(edge_bw):
                        q.remove_task(edge_bw)

                    gain_bw, _, _, _, _ = dag_search.eval_edge_addition(
                        edge_bw.i, edge_bw.j
                    )
                    gain_fw, _, _, _, _ = dag_search.eval_edge_addition(
                        edge_fw.i, edge_fw.j
                    )
                    q.add_task(edge_bw, gain_bw * 100)
                    q.add_task(edge_fw, gain_fw * 100)

            # Reconsider edges Xk->Xj in q given the current model as their score changed upon adding Xi->Xj
            for mom in dag_search._nodes:
                # Do not consider Xi,Xj, or current parents/children of Xi
                if (
                    node == mom
                    or parent == mom
                    or dag_search.is_edge(mom, node)
                    or dag_search.is_edge(node, mom)
                ):
                    continue
                edge_candidate = dag_search.pair_edges[mom][
                    node
                ]  # pi_dag.init_edges[mom][target]
                gain_mom, score, _, _, _ = dag_search.eval_edge_addition(node, mom)

                if q.exists_task(edge_candidate):  # ow. insignificant /skipped
                    q.remove_task(edge_candidate)
                    q.add_task(edge_candidate, gain_mom * 100)
        except KeyError:  # empty or all remaining entries are tagged as removed
            pass

    if verbosity > 0:
        logger.info(f"Forward: {np.round(time.perf_counter() - st, 2)}s ")
    return q, dag_search


def _dag_backward_phase(
    q: UPQ, dag_search: DAG, logger: logging.Logger, verbosity: int
) -> (UPQ, DAG):
    st = time.perf_counter()
    if verbosity > 0:
        logger.info("Backward Phase ...")

    for j in dag_search._nodes:
        parents = dag_search.parents_of(j)
        if len(parents) <= 1:
            continue
        max_gain = -np.inf
        arg_max = None

        # Consider all graphs G' that use a subset of the target's current parents
        min_size = 1  # todo min_size = 0 allowed?
        for k in range(min_size, len(parents) + 1):
            parent_sets = itertools.combinations(parents, k)
            for parent_set in parent_sets:
                gain = dag_search.eval_edges(j, parent_set)
                if gain > max_gain:
                    max_gain = gain
                    arg_max = parent_set
                # print(f'\tconsidering {parent_set} -> {j}, {np.round(gain[0][0],2)}')
        if (arg_max is not None) and (not is_insignificant(max_gain)):
            if verbosity > 0:
                logger.info(f"\tupdating {parents} to {arg_max} -> {j}")
            dag_search.update_edges(j, arg_max)

    if verbosity > 0:
        logger.info(f"Backward: {np.round(time.perf_counter() - st, 2)}s ")

    return q, dag_search
