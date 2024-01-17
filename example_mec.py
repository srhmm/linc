import numpy as np
from typing import List

import upq
from function_types import FunctionType
from gen_context_data import gen_context_data
from intervention_types import IvType
from out import Out
from sparse_shift import cpdag2dags, dag2cpdag
from pi_tree import PiTree
from utils_tp import TP_tree_dag_cpdag
from vsn import Vsn

#seed 1, runs20:
#match:13, nomatch:7
#TP:39, TN:39
#FP:20, FPrev:20
# seed 100:
#match:65, nomatch:35
#TP:225, TN:225
#FP:92, FPrev:92
#precis, rec: 0.7



def test_mec():
    args = gen_context_data(5, 750, 5, 1, FunctionType.GP_PRIOR, IvType.CONST, IvType.CONST, iid_contexts=False,
                            partition_search=False, iv_per_node=[1, 1])
    Dc, dag, Gc, \
    target, parents, children, confounder, \
    partitions_X, observational_X = args
    from exp_synthetic.methods_linc import LINC
    li = LINC(dag2cpdag(dag.weight_mat), dag.weight_mat, None, True)
    for Xc in Dc:
        li.add_environment(Xc)

    min_dag = li.get_min_dags(False)

    vsn = Vsn(rff=True, ilp_partitions=True)
    T = PiTree(5, len(partitions_X), Dc, vsn)
    #TODO consider not cmp initial edges at all but then not calling get_mdl_graph_cost
    _ = T.initial_edges(upq.UPQ(), Out("", vb=False, tofile=False), subsample_size=len(Dc[0]),
                        cmp_score=False) #, mdl_cache=mdl_cache, score_cache=score_cache, pi_cache=pi_cache )  # skips computing initial scores for all node pairs

    for i in range(len(dag.weight_mat)):
        for j in range(len(dag.weight_mat[i, :])):
            if dag.weight_mat[i, j] != 0:
                gain, score, mdl, pi, _ =  T.eval_edge(T.init_edges[i][j])#i=i, j=j)
                print(i, j, gain, score, mdl, pi)
                T.add_edge(i, j, score, mdl, pi)
    mdl = T.get_graph_mdlcost()
    mdl_cache, score_cache, pi_cache = T.mdl_cache,  T.score_cache, T.pi_cache

class TP_metric():
    def __init__(self):
        self.match, self.nomatch, self.TP, self.TN, self.FP, self.FP_anticausal, self.FN = 0, 0, 0, 0, 0, 0, 0
    def inc(self, tp, tn, fp, fpa, fn):
        match = 1 if (fp + fpa + fn == 0) else 0
        nomatch = 0 if (match == 1) else 1
        self.match, self.nomatch = self.match + match, self.nomatch  + nomatch
        self.TP, self.TN  = self.TP + tp, self.TN + tn
        self.FP, self.FP_anticausal, self.FN   =  self.FP + fp, self.FP_anticausal + fpa, self.FN + fn
    def precis(self):
        return(self.TP /(self.TP + self.FP))
    def recall(self):
        return(self.TP /(self.TP + self.FN))
    def __str__(self):
        return ("match:"+str(self.match)+", nomatch:"+str(self.nomatch)+
                "\nTP:"+str(self.TP)+", TN:"+str(self.TN)+"\nFP:"+str(self.FP)+", FPrev:"+str(self.FP_anticausal))
def test_equivalence_search(runs = 2, seed = 200):

    res = TP_metric()
    for _ in range(runs):
        seed, res = equivalence_search(seed, res, iv_per_node=[0,1], rff=True)
        seed = seed + 1
    return res


def equivalence_search(seed: int, res: TP_metric, iv_per_node: List[int], rff: bool):

    C_n = 5
    D_n = 750
    node_n = 5
    iv_type = IvType.CONST

    Dc, dag, Gc, _, _, _, _, pis, _ = gen_context_data(C_n, D_n, node_n, seed, fun_type=FunctionType.GP_PRIOR, iv_type_target=iv_type,
                                                       iv_per_node=iv_per_node,
                                                       iv_type_covariates=iv_type,
                                                       iid_contexts=False, partition_search=False)
    dags = cpdag2dags(dag2cpdag(dag.weight_mat))
    it = 0
    while len(dags)==1:
        seed, it  = seed + 1, it + 1
        Dc, dag, Gc, _, _, _, _, pis, _ = gen_context_data(C_n, D_n, node_n, seed, fun_type=FunctionType.GP_PRIOR, iv_type_target=iv_type,
                                                       iv_type_covariates=iv_type, iv_per_node=[0, 1],
                                                       iid_contexts=False, partition_search=False)
        dags = cpdag2dags(dag2cpdag(dag.weight_mat))
        if (it > 1000):
            raise(Exception)
    cpdag = dag2cpdag(dag.weight_mat)
    vsn = Vsn(rff=rff)
    mdl_min = np.inf
    arg_min = None
    for cand in dags:
        T = PiTree(C_n, len(pis), Dc, vsn)
        _ = T.initial_edges(upq.UPQ(), Out("", vb=False, tofile=False), D_n, cmp_score=False) # skips computing initial scores for all node pairs

        for i in range(len(cand)):
            for j in range(len(cand[i, :])):
                if cand[i,j] != 0:
                    gain, score, mdl, pi, _ = T.eval_edge(T.init_edges[j][i])
                    T.add_edge(i, j , score, mdl, pi)
        mdl = T.get_graph_mdlcost()
        if mdl < mdl_min:
            mdl_min = mdl
            arg_min = T

    # total number of TP
    #tp, tn,fp, fpa, fn  = TP_tree_dag(arg_min, dag)

    # TP over those edges that remain undirected in the MEC
    tp, tn, fp, fpa, fn = TP_tree_dag_cpdag(arg_min, dag, cpdag)

    res.inc(tp, tn, fp, fpa, fn)


    return seed, res


