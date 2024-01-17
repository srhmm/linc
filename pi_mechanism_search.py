import itertools
import time

import numpy as np

from pi_search_clustering import pi_search_clustering
from pi_search_mdl import pi_mdl, pi_mdl_conf, pi_mdl_best
from pi_tree import is_insignificant
from pi import PI
from pi_search_ilp import pi_search_ILP
from utils_pi import pi_convertfrom_pair

def pi_mechanism_search(Xyc, target, candidates, vsn):
    """ For target node Xi and subsets Xs, discovers the mechanisms fpi : Xs -> Y (with best context partition and MDL score)"""
    result_pis = [None for _ in range(len(candidates))]
    result_scores = [0 for _ in range(len(candidates))]
    for i, set in enumerate(candidates):
        result_pis[i],result_scores[i] = pi_partition_search(Xyc, target, set, vsn)
    return result_pis, result_scores


def pi_partition_search(Xyc, target, parents, vsn):
    """ For target node Xi and one subset Xs, discovers the mechanisms fpi : Xs -> Y (with best context partition and MDL score)"""
    C_n = len(Xyc)
    Xc = np.array([Xyc[c_i][:, [p for p in parents]] for c_i in range(C_n)])
    yc = np.array([Xyc[c_i][:, target] for c_i in range(C_n)])

    # LINC GP or RFF Regression -----------------
    start_time = time.perf_counter()
    gpr = PI(Xc, yc, np.random.RandomState(1), skip_regression=False,
                       skip_regression_pairs=not vsn.regression_per_pair, rff=vsn.rff, pi_search=True)
    gpr.cmp_distances(from_G_ij=vsn.regression_per_pair)
    print(f"Regression (RFF={vsn.rff}):", round(time.perf_counter() - start_time, 5), "sec")


    # LINC ILP search -----------------
    start_time = time.perf_counter()

    # LINC search using clustering approach -----------------

    if vsn.ilp:
        if vsn.clustering:
            pistar, pistar_mdl, pistar_model, pistar_data, pistar_penalty = \
                pi_search_clustering(gpr.pair_mdl_gains, C_n, gpr, vsn.regression_per_group)
        else:
            pair_indicator_vars, _, pair_contexts = pi_search_ILP(gpr.pair_mdl_gains, C_n, wasserstein=False)
            pistar = pi_convertfrom_pair(pair_indicator_vars, pair_contexts, C_n)
            pistar_mdl, pistar_model, pistar_data, pistar_penalty, _ = pi_mdl(gpr, pistar,                             regression_per_group=vsn.regression_per_group)



        # LINC Exhaustive search -----------------
    else:
            pistar, pistar_mdl, pistar_model, pistar_data, pistar_penalty, _ = \
                pi_mdl_best(gpr, vsn)

    print(f"Partition search (ILP={vsn.ilp}):", round(time.perf_counter() - start_time, 5), 'sec')

    return pistar, pistar_mdl