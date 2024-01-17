import statistics
from enum import Enum
from statistics import mean, stdev

import numpy as np
from graphical_models import rand

from function_types import FunctionType
from gen_context_data import gen_arc_weights_pi
from intervention_types import IvType
from linear_dag import LinearDAG
from nonlinear_dag import NonlinearDAG
from tree_search import tree_search
from utils import cantor_pairing, data_scale
from utils_pi import pi_rand, pi_group_map
from vsn import Vsn


class SparsityBivariate(Enum):
    IID = 0
    #invariance = indep change of parent
    INVARIANCE_1 = 1
    INVARIANCE_2 = 2
    INVARIANCE_3 = 3
    INVARIANCE_4 = 4
    INVARIANCE_5 = 5
    # indep change of the child while parent invariant
    IC_1 = 6
    IC_2 = 7
    IC_3 = 8
    IC_4 = 9
    IC_5 = 10
    # both change
    BOTH_A = 11
    BOTH_B = 12
    BOTH_C = 13
    BOTH_D = 14
    # adversarial changes
    ADVERSE_A = 15
    ADVERSE_B = 16
    ADVERSE_C = 17

def sparsity_to_partitions(spars: SparsityBivariate):
    pi1 =  [[0,1,2,3,4]]
    pi0 =  [[0], [1], [2], [3], [4]]
    if spars is SparsityBivariate.IID:
        return pi1, pi1
    if spars is SparsityBivariate.INVARIANCE_5:
        return [[0], [1], [2], [3], [4]], pi1
    if spars is SparsityBivariate.INVARIANCE_4:
        return [[0], [1], [2], [3, 4]], pi1
    if spars is SparsityBivariate.INVARIANCE_3:
        return [[0], [1, 2], [3, 4]], pi1
    if spars is SparsityBivariate.INVARIANCE_2:
        return [[0], [1,2,3,4]], pi1
    if spars is SparsityBivariate.INVARIANCE_1:
        return [[0,1],[2,3,4]], pi1

    if spars is SparsityBivariate.IC_5:
        return pi1, [[0], [1], [2], [3], [4]]
    if spars is SparsityBivariate.IC_4:
        return pi1, [[0], [1], [2], [3, 4]]
    if spars is SparsityBivariate.IC_3:
        return pi1, [[0], [1, 2], [3, 4]]
    if spars is SparsityBivariate.IC_2:
        return pi1, [[0], [1, 2, 3, 4]]
    if spars is SparsityBivariate.IC_1:
        return pi1, [[0,1], [2, 3, 4]]

    if spars is SparsityBivariate.BOTH_A:
        return  [[1], [0, 2, 3, 4]] , [[0], [1, 2, 3, 4]]
    if spars is SparsityBivariate.BOTH_B:
        return  [[0, 1], [2, 3, 4]],  [[2], [0, 1, 3, 4]]
    if spars is SparsityBivariate.BOTH_C:
        return  [[0, 1], [2, 3, 4]], [[2, 3], [0, 1, 4]],
    if spars is SparsityBivariate.BOTH_D:
        return  [[0, 1, 2], [3, 4]], [[1, 2], [0, 1, 4]],

    if spars is SparsityBivariate.ADVERSE_B:
        return [[0,1], [2,3], [4]], [[0,2], [3,1], [4]]

    if spars is SparsityBivariate.ADVERSE_C:
        return [[0, 1, 2], [3,4]], [[0,4], [1,2,3]]

    if spars is SparsityBivariate.ADVERSE_A:
        return pi0, pi0


def gen_bivariate( C_n, D_n, fun_type, iv_type,
                   pi_parent, pi_child,
                   seed, scale=True):
    dag = rand.directed_erdos(nnodes=2, density=1)

    dag = rand.rand_weights(dag)

    if dag.weight_mat[0,1] != 0:
        assert dag.weight_mat[1,0] == 0
        pa, ch = 0, 1
    else:
        assert dag.weight_mat[1,0] != 0
        ch, pa = 1, 0

    Dc = [None for _ in range(C_n)]
    Gc = [None for _ in range(C_n)]

    partitions_X = [None for _ in range(2)]
    arc_weights_X = [None for _ in range(2)]

    # Partition for each node ----------
    partitions_X[pa] = pi_parent
    partitions_X[ch] = pi_child
                      #        permute=True, k_min=intervX_n+1, k_max=intervX_n+1, single_context_groups=False) # [[c_i for c_i in range(C_n)]]
    arc_weights_X[0] = gen_arc_weights_pi(partitions_X[0], dag, seed, iv_type)
    arc_weights_X[1] = gen_arc_weights_pi(partitions_X[1], dag, seed, iv_type)

    # n_dag = NonlinearDAG(dag.nodes, dag.arc_weights)
    # DAG and data per context ----------
    for c_i in range(C_n):
        lin_dag_c = LinearDAG(dag.nodes, dag.arcs)

        for node_X in dag.nodes:
            pi_X = partitions_X[node_X]
            obs_X = 0
            pi_map = pi_group_map(pi_X, C_n)
            pi_k = pi_map[c_i]

            change_parameters = iv_type is IvType.PARAM_CHANGE  # or iv_type_X is IvType.CONST)

            # Nothing to be done for an observational context (except setting arc weight if param change)
            if c_i in pi_X[obs_X] and not change_parameters:
                continue

            # Interventional Group: arc weight of DAG changes
            if change_parameters:
                weights_to_p = arc_weights_X[node_X][pi_k]
                for (i, j) in dag.arcs:
                    if j == node_X:
                        w_ij = weights_to_p[(i, j)]
                        lin_dag_c.set_arc_weight(i, j, w_ij)  # a new DAG where causal weight in context is the group's
            else:
                if iv_type == IvType.SHIFT:
                    lin_dag_c.set_node_bias(node_X, 2)
                if iv_type == IvType.SCALE:
                    lin_dag_c.set_node_variance(node_X, 20)

        # Sample data in context c ----------
        # Linear Gaussian Models
        if fun_type is FunctionType.LINEAR_GAUSS:
            data_c = lin_dag_c.sample(nsamples=D_n)
            if scale:
                data_c = data_scale(data_c)
            Dc[c_i] = data_c
            Gc[c_i] = lin_dag_c

    if fun_type is FunctionType.GP_PRIOR:
        ndag = NonlinearDAG(dag.nodes, dag.arc_weights)
        Dc = ndag.sample_data(D_n, C_n, seed, partitions_X, 3, iv_type,  iv_type)

        for c_i in range(C_n):
            Gc[c_i] = ndag
    return Dc, Gc, partitions_X, pa, ch

def causal_gain_sparsity(reps = 10):
    gains = [0 for _ in range(len(SparsityBivariate))]
    var = [0 for _ in range(len(SparsityBivariate))]
    TP, TN, FN, FP = 0,0,0,0
    for i, spars in enumerate(SparsityBivariate):
        gains_sub = [0 for _ in range(reps)]
        for j in range(reps):
            gain,  tp, tn, fn, fp  = bivariate_tree_search(spars, seed=j, rff=True)
            gains_sub[j] = gain
            TP, TN, FN, FP = TP + tp, TN + tn, FN + fn, FP + fp
        gains[i] = mean(gains_sub)
        var[i] = stdev(gains_sub)

    for i, spars in enumerate(SparsityBivariate):
        print("\n", spars)
        pi_parent, pi_child = sparsity_to_partitions(spars)
        print("\tPartitions:" + str(pi_parent) + str(pi_child))
        print("\tGain:", gains[i], "+-:", var[i])


def bivariate_tree_search(spars,seed, rff=False):
    C_n = 5
    D_n = 750

    pi_parent, pi_child = sparsity_to_partitions(spars)

    Dc, Gc, pis, pa, ch = gen_bivariate(C_n, D_n, FunctionType.GP_PRIOR, IvType.PARAM_CHANGE, pi_parent, pi_child, seed=seed)

    T = tree_search(Dc, vsn = Vsn(rff=rff, ilp_in_tree_search=True, regression_per_pair=False, regression_per_group=False,
                                mdl_gain=True), revisit_children=True, revisit_queue=True, revisit_parentsets=True, prnt=Gc[0].arcs, vb=False, tofile=False)
    #s= ("\n" +  str(spars) +#  "\tPartitions:" + str(pi_parent) + str(pi_child) +
     #"\tGain:" + str(round(get_causal_gain(T, pa, ch),2)))
    adj  = T.get_adj()
    tp, tn, fn, fp = 0,0,0,0

    if adj[ch, pa]==0:
        tn = tn + 1

    if adj[pa,ch]==1:
        tp = tp + 1
    else:
        if adj[ch,pa]==1:
            fp = fp + 1
        else:
            fn = fn + 1

    return get_causal_gain(T, pa, ch), tp, tn, fn, fp


def get_causal_gain(T, pa, ch):
    _, good, _, _, _ = (T.eval_edge(T.init_edges[ch][pa])) #smaller is better
    _, bad, _, _, _ = (T.eval_edge(T.init_edges[pa][ch])) #the smaller, the stronger indication for the anticausal direction
    return bad-good # large if causal direction is preferred over the anticausal