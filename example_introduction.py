from statistics import mean, stdev

import numpy as np
from graphical_models import rand

from gen_context_data import gen_arc_weights_pi
from upq import UPQ
from intervention_types import IvType
from linear_dag import LinearDAG
from nonlinear_dag import NonlinearDAG
from out import Out
from pi_tree import PiTree, edge_score_pi
from sparse_shift import dag2cpdag
from exp_synthetic.methods_linc import LINC
from utils_pi import pi_group_map, pi_enum
from vsn import Vsn


def example_identifiability(reps = 2):
    scores = [[[np.inf for _ in range(reps)] for _ in range (7)] for _ in range(5) ]

    dag = rand.directed_erdos(nnodes=6, density=0)
    dag.add_arc(0, 1)
    dag.add_arc(1, 2)
    dag.add_arc(1, 4)
    dag.add_arc(1, 6)
    #dag.add_arc(1, 8)
    dag.add_arc(2, 3)
    dag.add_arc(2, 5)
    dag.add_arc(2, 7)
    #dag.add_arc(2, 9)
    dag = NonlinearDAG(dag.nodes, dag.arcs)
    C_n = 5
    node_n = len(dag.nodes)

    # CAUSE 0 , EFFECTinterv1: 2 , effect interv 5 : 4, unrel interv1:3, unrel interv5:5
    partitions_X = [[[0], [1, 2, 3, 4]],
                    [[1], [0, 2, 3, 4]],
                    [[0, 1, 4], [2, 3]], [[0, 3, 4], [1, 2]],
                    [[0, 1],[4], [2, 3]], [[0, 4],[3], [1, 2]],
                    [[0], [1], [2], [3], [4]], [[0], [1], [2], [3], [4]]]#,
                   # [[0,1,2,3,4]], [[0,1,2,3,4]]]
    partitions_X = [[[0, 1, 2, 3, 4]],[[0, 1, 2, 3, 4]],[[0, 1, 2, 3, 4]],[[0, 1, 2, 3, 4]],
                    [[0, 1, 2, 3, 4]],[[0, 1, 2, 3, 4]],[[0, 1, 2, 3, 4]],
                    [[0, 1, 2, 3, 4]]]
    # TODO : do this for all partitions at k and report the best one!
    candidate_partitions = [[[0, 1, 2, 3, 4]], [[1], [0, 2, 3, 4]],
                            [[0, 1], [2, 3], [4]], [[0, 1], [2], [3, 4]], [[1], [0, 2], [3, 4]], [[1], [2], [0, 3, 4]], [[1], [2], [0, 3], [4]],
                            [[0, 1], [2], [3], [4]], [[0], [1], [2], [3], [4]]]
    candidate_partitions = [[[0,3],[1],[2],[4]],[[0,4],[1],[2],[3]],[[3,4],[1],[2],[0]]]
    candidate_partitions = [[[0, 1, 2, 3, 4]], [[1], [0, 2, 3, 4]], [[1], [2], [0, 3, 4]],
                            [[3,4],[1],[2],[0]], [[0], [1], [2], [3], [4]]   ]

    #candidate_partitions = pi_enum(5, permute=True)
    for rep in range(reps):
        Dc  = dag.sample_data(500, C_n, 0, partitions_X, rep, IvType.PARAM_CHANGE, IvType.PARAM_CHANGE, scale=True)

        vsn = Vsn(rff=True, ilp_partitions=True, mdl_gain=False)
        T = PiTree(C_n, len(partitions_X), Dc, vsn)
        _ = T.initial_edges(UPQ(), Out("", vb=False, tofile=False), subsample_size=len(Dc[0]),
                            cmp_score=False)  # , mdl_cache=mdl_cache, score_cache=score_cache, pi_cache=pi_cache )  # skips computing initial scores for all node pairs

        # Candidate causal mechanisms for node 1: no parent, true parent 0, wrong parent 2
        # In each case, consider partition with no groups, with the correct group, with 3 or 4 or 5 groups

        candidate_parents =[[7], [9]]# [[0], [2], [4], [3], [5]]
        s = "*** REP: "+ str(rep)
        j = 1
        for pa_i, pa in enumerate(candidate_parents):
            s = s + str("\n\t Parent Set: " + str(pa))
            for k in range(5):

                candidate_partitions = pi_enum(5, k_min=k+1, k_max=k+1, permute=True)
                s = s + str("\n\t k: " + str(k+1))
                for pi in candidate_partitions:
                    s = s + str("\n\t\t pi: " + str(pi))
                    score, mdl, pi = edge_score_pi(Dc, pa, j, pi, vsn, C_n)  #
                    if score < scores[k][pa_i][rep]:
                        scores[k][pa_i][rep] = score
                    # full = mdl + others
                # T.add_edge(i, j, score, mdl, pi)
                    s = s + ", score:" + str(round(mdl, 1))  # + ", full: " + str(round(full, 1))
        print(s)

    mean_scores, std_scores = [[0  for _ in range (7)] for _ in range(5) ], [[0  for _ in range (7)] for _ in range(5) ]
    for rep in range(reps):
        for k in range(len(scores[0])):
            for p in range(len(scores[0][k])):
                mean_scores[k][p] , std_scores[k][p] = mean(scores[k][p]) , stdev(scores[k][p])

    return scores, mean_scores, std_scores
def example_introduction():
    dag = rand.directed_erdos(nnodes=6, density=0)
    dag.add_arc(0, 1)
    dag.add_arc(1, 2)
    dag.add_arc(1, 4)
    dag.add_arc(2, 3)
    dag.add_arc(2, 5)
    dag = NonlinearDAG(dag.nodes, dag.arcs)
    lindag = LinearDAG(dag.nodes, dag.arcs)
    C_n = 6
    node_n = len(dag.nodes)

    #CAUSE 0 , EFFECTinterv1: 2 , effect interv 5 : 4, unrel interv1:3, unrel interv5:5
    partitions_X = [[[0], [1, 2, 3, 4, 5]], [[1], [0, 2, 3, 4, 5]], [[0, 1], [2, 3], [4], [5]],
                    [[0], [4], [1, 2], [3, 5]], [[0], [1], [2], [3], [4], [5]], [[0], [1], [2], [3], [4], [5]]]
    # TODO : do this for all partitions at k and report the best one!
    candidate_partitions = [[[0, 1, 2, 3, 4, 5]], [[1], [0, 2, 3, 4, 5]], [[0, 1], [2, 3], [4, 5]],
                            [[0, 1], [2], [3], [4, 5]], [[0], [1], [2], [3], [4, 5]], [[0], [1], [2], [3], [4], [5]]]
    Dc_nonlinear = dag.sample_data(500, C_n, 0, partitions_X, 0, IvType.PARAM_CHANGE, IvType.PARAM_CHANGE)

    Dc = [None for _ in range(C_n)]

    arc_weights_X = [None for _ in range(node_n)]
    seed = 0
    for node_X in dag.nodes:
        partition_X = partitions_X[node_X]
        arc_weights_X[node_X] = gen_arc_weights_pi(partition_X, lindag, seed, IvType.PARAM_CHANGE)
    for c_i in range(C_n):
        lin_dag_c = LinearDAG(lindag.nodes, lindag.arcs)  # lin_dag.copy()

        for node_X in dag.nodes:
            pi_X = partitions_X[node_X]
            obs_X = 0
            pi_map = pi_group_map(pi_X, C_n)
            pi_k = pi_map[c_i]

            weights_to_p = arc_weights_X[node_X][pi_k]
            for (i, j) in dag.arcs:
                if j == node_X:  # and not (i == confounder and node_X == target):
                    w_ij = weights_to_p[(i, j)]
                    lin_dag_c.set_arc_weight(i, j, w_ij)  # a new DAG where causal weight in context is the group's

        data_c = lin_dag_c.sample(500)
        Dc[c_i] = data_c

    #li = LINC(dag2cpdag(dag.weight_mat), dag.weight_mat, None, True)
    #for Xc in Dc:
    #    li.add_environment(Xc)

    # min_dag = li.get_min_dags(False)

    vsn = Vsn(rff=False, ilp_partitions=True, mdl_gain=False)
    T = PiTree(C_n, len(partitions_X), Dc, vsn)
    _ = T.initial_edges(UPQ(), Out("", vb=False, tofile=False), subsample_size=len(Dc[0]),
                        cmp_score=False)  # , mdl_cache=mdl_cache, score_cache=score_cache, pi_cache=pi_cache )  # skips computing initial scores for all node pairs

    # Candidate causal mechanisms for node 1: no parent, true parent 0, wrong parent 2
    # In each case, consider partition with no groups, with the correct group, with 3 or 4 or 5 groups

    candidate_parents = [[0], [2], [3], [4]]  # [[], [0], [2], [0,2]]
    s = ""
    j = 1
    for pa in candidate_parents:
        s = s + str("\n\t Parent Set" + str(pa))
        for pi in candidate_partitions:
            s = s + str("\n\t k" + str(len(pi)))
            score, mdl, pi = edge_score_pi(Dc, pa, j, pi, vsn, C_n)  #
            # full = mdl + others
            # T.add_edge(i, j, score, mdl, pi)
            s = s + ", score:" + str(round(mdl, 1))  # + ", full: " + str(round(full, 1))
    print(s)
# def example_introduction():
#
#     dag = rand.directed_erdos(nnodes=4, density=0)
#     dag.add_arc(0,1)
#     dag.add_arc(1,2)
#     dag.add_arc(2,3)
#     dag = NonlinearDAG(dag.nodes, dag.arcs)  # lin_dag.copy()
#     dag = LinearDAG(dag.nodes, dag.arcs)
#
#
#     C_n = 3
#     partitions_X = [[[0], [1,2]], [[1], [0,2]], [[2], [0,1]]]
#     candidate_partitions= [[[0,1,2]], [[1], [0,2]], [[0],[1],[2]]]
#     Dc = dag.sample_data(1000, C_n, 3, partitions_X, 0, IvType.PARAM_CHANGE, IvType.PARAM_CHANGE)
#
#
#     C_n = 5
#     partitions_X = [[[0], [1, 2, 3, 4]], [[1], [0, 2, 3, 4]], [[2], [4], [1,0], [3]],  [[0], [4], [1,2], [3]],]
#     candidate_partitions= [[[0,1,2,3,4]], [[1],[0,2,3,4]], [[0,1],[2,3], [4]], [[0,1],[2],[3],[4]], [[0],[1],[2],[3],[4]]]
#     C_n = 6
#     partitions_X = [[[0], [1, 2, 3, 4, 5]], [[1], [0, 2, 3, 4, 5]], [[2], [4, 5], [1, 0], [3]],  [[0], [4], [1,2], [3,5]],]
#     #TODO : do this for all partitions at k and report the best one!
#     candidate_partitions= [[[0,1,2,3,4,5]], [[1],[0,2,3,4,5]], [[0,1],[2,3], [4,5]], [[0,1],[2],[3],[4,5]], [[0],[1],[2],[3],[4,5]], [[0],[1],[2],[3],[4],[5]]]
#     Dc = dag.sample_data(500, C_n, 0, partitions_X, 0, IvType.PARAM_CHANGE, IvType.PARAM_CHANGE)
#
#     li = LINC(dag2cpdag(dag.weight_mat), dag.weight_mat, None, True)
#     for Xc in Dc:
#         li.add_environment(Xc)
#
#     #min_dag = li.get_min_dags(False)
#
#     vsn = Vsn(rff=False, ilp_partitions=True, mdl_gain=False)
#     T = PiTree(C_n, len(partitions_X), Dc, vsn)
#     _ = T.initial_edges(UPQ(), Out("", vb=False, tofile=False), subsample_size=len(Dc[0]),
#                         cmp_score=False)  # , mdl_cache=mdl_cache, score_cache=score_cache, pi_cache=pi_cache )  # skips computing initial scores for all node pairs
#
#     # Candidate causal mechanisms for node 1: no parent, true parent 0, wrong parent 2
#     # In each case, consider partition with no groups, with the correct group, with 3 or 4 or 5 groups
#
#     candidate_parents = [[0,2]]#[[], [0], [2], [0,2]]
#     s= ""
#     j = 1
#     for pa in candidate_parents:
#         s = s+str("\n\t Parent Set"+str(pa))
#         for pi in candidate_partitions:
#             s = s+str("\n\t k"+str(len(pi)))
#             score, mdl, pi= edge_score_pi(Dc, pa, j, pi, vsn, C_n)#
#             #full = mdl + others
#             #T.add_edge(i, j, score, mdl, pi)
#             s = s +", score:" + str(round(mdl,1)) #+ ", full: " + str(round(full, 1))
#     print(s)