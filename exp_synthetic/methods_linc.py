import numpy as np
from pulp import PulpSolverError

import upq
from out import Out
from pi_dag_search import pi_dag_search
from pi_mec import PiMEC
from pi_mechanism_search import pi_mechanism_search
from sparse_shift import cpdag2dags
from pi_tree import PiTree
from sparse_shift import dag_precision, dag_recall
from vsn import Vsn


class LINC:
    """
    LINC (A wrapper for experiments)
    """

    def __init__(self, cpdag, dag, rff, pi_search, ILP, known_mec, clus):
        self.domains_ = []
        self.cpdag = cpdag  # adj matrix
        self.dag = dag
        self.maxenv_only = True
        self.known_mec=known_mec
        #self.tp_metric = tp_metric
        self.rff = rff
        self.pi_search=pi_search
        self.min_dags_ = np.zeros((len(cpdag), len(cpdag)))
        self.min_gains_, self.min_sig_ = None, None
        self.min_obj_ = None
        self.min_conf_ = None
        self.min_mdl_node_ = None
        self.vsn = Vsn(rff=rff, clus=clus, ilp=ILP)

    def add_environment(self, interventions):
        self.domains_.append(interventions)

    def get_mechanisms(self, y, subsets):
        return self._linc_mechanismsearch(self.domains_,y, subsets)


    # LINC for DAG search
    def get_min_dags(self, soft):
        mec, dag, mdl, gains, sig, mdl_node = self._linc_safe(self.domains_)
        self.min_obj_ = mec
        self.min_dags_ = dag
        self.min_mdl_ = mdl
        self.min_gains_, self.min_sig_ = gains, sig
        self.min_mdl_node_ = mdl_node

        #self.min_gains_, self.min_sig_ = mec.conf_mat(dag)
        return self.min_dags_

    #TODO should this return some undirected edges if not conf?
    def get_min_cpdag(self, soft):
        cpdag =self.min_dags_# (np.sum(self.min_dags_, axis=0) > 0).astype(int)
        return cpdag

    def _linc_safe(self, Xs):
        if not self.known_mec:
            return self._linc_dagsearch(Xs)
        try:
            return self._linc(Xs)
        except (PulpSolverError):
            print("(LINC) Switching from ILP to clus")
            self.vsn.clustering=True
            #
            #print("(LINC) Switching from ILP to exhaustive")
            #self.vsn.ilp_partitions=False
            return self._linc(Xs)

    # LINC for DAG search (unknown MEC)
    def _linc_dagsearch(self, Xs):
        dag = pi_dag_search(Xs, self.vsn)
        adj = dag.get_adj().T
        mdl = dag.get_graph_mdlcost()
        gain_mat, sig_mat =dag.conf_mat(adj)
        mdl_node = None #not Implemented!
        return dag, adj, mdl, gain_mat, sig_mat, mdl_node

    # LINC for a node in the DAG
    def _linc_mechanismsearch(self, Xs, y, subsets):
        pis, scores = pi_mechanism_search(Xs, y, subsets, self.vsn)
        imin = min(range(len(scores)), key=scores.__getitem__)
        pistar, pistar_score = pis[imin], scores[imin]
        return pistar, pistar_score, pis, scores

    # MEC search
    def _linc(self, Xs):
        C_n = len(Xs)
        D_n = Xs[0].shape[0]
        X_n = Xs[0].shape[1]

        # Search over all DAGs in true MEC
        dags = cpdag2dags(self.cpdag)

        mdl_min = np.inf
        mdl_node_min = None
        dag_min = np.zeros((X_n, X_n))
        mec = PiMEC(C_n, X_n, Xs, self.vsn)

        for cand in dags:
            mdl, mdl_node = mec.eval_adj(cand, gain=self.vsn.mdl_gain)
            if mdl < mdl_min:
                mdl_min = mdl
                dag_min = cand
                mdl_node_min = mdl_node

        gain_min, sig_min = mec.conf_mat(dag_min, gain=self.vsn.mdl_gain)


        # Mechanism scores for each node: DAG[i][j] gets score for j's mechanism f:PA_j ->j.
        mdl_node_mat = np.zeros((X_n, X_n))
        for j in mec.nodes:
            for i in mec.nodes:
                if dag_min[i][j] == 1:
                    mdl_node_mat[i][j] = mdl_node_min[j]

        return mec, dag_min, mdl_min, gain_min, sig_min, mdl_node_mat

    #The same logic but hacky code (passing score caches from each tree class to the next)
    def _linc_tree_for_mec(self, Xs, rff, gain):
        C_n = len(Xs)
        D_n = Xs[0].shape[0]
        X_n = Xs[0].shape[1]

        # Search over all DAGs in true MEC
        dags = cpdag2dags(self.cpdag)
        T = PiTree(C_n, X_n, Xs, self.vsn)
        score_cache, mdl_cache, pi_cache = T.score_cache, T.mdl_cache, T.pi_cache
        mdl_min = np.inf
        dag_min = np.zeros((X_n, X_n))
        tree_min = T

        for cand in dags:
            T = PiTree(C_n, X_n, Xs, self.vsn)
            _ = T.initial_edges(upq.UPQ(), Out("", vb=False, tofile=False), D_n,
                                cmp_score=False, score_cache = score_cache, mdl_cache=mdl_cache, pi_cache=pi_cache)  # skips computing initial scores for all node pairs

            for i in range(len(cand)):
                for j in range(len(cand[i, :])):
                    if cand[i, j] != 0:
                        gain, score, mdl, pi, _ = T.eval_edge(T.init_edges[j][i])
                        T.add_edge(i, j, score, mdl, pi)
            mdl = T.get_graph_mdlcost()
            if gain:
                mdl= T.get_graph_score()
            #cache = T.score_cache
            if mdl < mdl_min:
                mdl_min = mdl
                dag_min = cand
                tree_min = T

        gain_min, sig_min = dag_min, dag_min #TODO no signif computed here yet

        return tree_min, dag_min, mdl_min, gain_min, sig_min

def dag_tpfpfnbi(true_dag, cpdag):

    tp = len(np.where((true_dag + cpdag - cpdag.T) == 2)[0]) #only causal edge
    fp = len(np.where((true_dag + cpdag.T - cpdag) == 2)[0]) #only noncausal edge
    fn = len(np.where((true_dag - cpdag - cpdag.T) == 1)[0])
    bi = len(np.where((cpdag + cpdag.T) == 2)[0])

    #TODO bidirected edges not used in the metrics?
    return tp, fp, fn, bi

def dag_fpr(true_dag, cpdag):
    tp, fp, fn, _ = dag_tpfpfnbi(true_dag, cpdag)
    return tp / (tp + fn) if (tp + fn) > 0 else 1


def pr_scores_optimistic(true_dag, estim_dag):

    # ap_score = 0
    # prior_recall = 0

    precisions = []
    recalls = []
    tpr = []
    fpr = []
    labels_true = []
    labels_soft = []

    for _ in range(1):
        cpdag = estim_dag
        precisions.append(dag_precision(true_dag, cpdag))
        recalls.append(dag_recall(true_dag, cpdag))
        labels_true = labels_true + list(true_dag.ravel())

        tpr.append(dag_recall(true_dag, cpdag))
        fpr.append(dag_fpr(true_dag, cpdag))
        #estim_dag_weighted = (cpdag - cpdag.T) * (1-t)# disregard bidirected edges and weight by decr pval

        mx = max(cpdag.ravel())
        estim_dag_weighted = cpdag
        mins = estim_dag_weighted[np.nonzero(estim_dag_weighted)]
        mn = .1
        if len(mins) > 0:
            mn = min(min(mins), .1)
        estim_dag_weighted[np.isnan(estim_dag_weighted)] = 0
        estim_dag_weighted = estim_dag_weighted.ravel() * 1.0
        estim_dag_weighted[estim_dag_weighted == estim_dag_weighted.transpose()] *= mn

        labels_soft = labels_soft + list(estim_dag_weighted.ravel())
        # ap_score += precision * (recall - prior_recall)
        # prior_recall = recall


    sort_idx = np.argsort(recalls)
    recalls = np.asarray(recalls)[sort_idx]
    precisions = np.asarray(precisions)[sort_idx]

    labels_soft = np.asarray(labels_soft)[sort_idx]
    labels_true = np.asarray(labels_true)[sort_idx]

    from sklearn.metrics import auc, roc_curve, roc_auc_score
    if (len(recalls)>1):
        aupr = auc(recalls, precisions)

        if (1 in labels_true):
            fprate, tprate, _ = roc_curve(labels_true, labels_soft)
            auroc_2 = auc(fprate, tprate)
            auroc_1 = roc_auc_score(labels_true, labels_soft)
            auroc = auc(fpr, tpr)
        else:
            if (len(np.nonzero(labels_soft))):
                auroc, auroc_1 = 0,0
            else:
                auroc, auroc_1  = 1, 1
        #ap = average_precision_score(recalls, precisions)

    else:
        score = precisions[0] * recalls[0]
        aupr, auroc, auroc_1 = score, score, score
    # if len(thresholds) == 1:
    #     ap_score = precisions[0] * recalls[0]
    # else:
    ap = (np.diff(recalls, prepend=0) * precisions).sum()

    return ap, aupr, auroc #, auroc_1

def pr_scores_(true_dag, estim_dag, soft_scores):

    # ap_score = 0
    # prior_recall = 0

    precisions = []
    recalls = []
    tpr = []
    fpr = []
    labels_true = []
    labels_soft = []

    precisions.append(dag_precision(true_dag, estim_dag))
    recalls.append(dag_recall(true_dag, estim_dag))
    labels_true = labels_true + list(true_dag.ravel())

    tpr.append(dag_recall(true_dag, estim_dag))
    fpr.append(dag_fpr(true_dag, estim_dag))
    #estim_dag_weighted = (cpdag - cpdag.T) * (1-t)# TODO disregard bidirected edges and weight by decr pval

    #soft_scores[np.isnan(soft_scores)] = 0
    labels_soft = labels_soft + list(soft_scores.ravel())
    # ap_score += precision * (recall - prior_recall)
    # prior_recall = recall

    sort_idx = np.argsort(recalls)
    recalls = np.asarray(recalls)[sort_idx]
    precisions = np.asarray(precisions)[sort_idx]

    sort_idx = np.argsort(labels_soft)
    sort_idx = sort_idx[::-1]
    labels_soft = np.asarray(labels_soft)[sort_idx]
    labels_true = np.asarray(labels_true)[sort_idx]


    from sklearn.metrics import auc, roc_auc_score

    # AUROC: considers labels
    if (1 in labels_true):
        auroc = roc_auc_score(labels_true, labels_soft)
    else:
        if (len(np.nonzero(labels_soft))):
            auroc = 0
        else:
            auroc = 1

    # Area under precision recall curve
    if (len(recalls)>1):
        aupr = auc(recalls, precisions)
    else:
        aupr = precisions[0] * recalls[0]

    # Same as average precision

    # if len(thresholds) == 1:
    #     ap_score = precisions[0] * recalls[0]
    # else:
    ap = (np.diff(recalls, prepend=0) * precisions).sum()

    return ap, aupr, auroc


def pr_scores_old(true_dag, soft_scores):
    import sklearn
    ap = sklearn.metrics.average_precision_score(true_dag.ravel(), soft_scores.ravel())

    fpr_, tpr_, _ = sklearn.metrics.roc_curve(true_dag.ravel(), soft_scores.ravel())
    roc = sklearn.metrics.auc(fpr_, tpr_)
    precision_, recall_, _ = sklearn.metrics.precision_recall_curve(true_dag.ravel(), soft_scores.ravel())
    auc = sklearn.metrics.auc(recall_, precision_)

    return ap, roc, auc


'''
def aep_score_pval(true_dag, soft_scores):
    """
    Considers all edge decisions, ordered by pvalue
    - scores = mch.soft_scores_, len(scores)= len(mch.n_dags_)
    - pvalues_mat.shape = (mch.n_dags_, mch.n_vars_)
    """

    from sparse_shift.utils import dag2cpdag, cpdag2dags
    dags = np.asarray(cpdag2dags(dag2cpdag(true_dag)))

    min_idx = np.where(soft_scores == np.min(soft_scores))[0] # index of the best dag in the mec according to min changes
    if len(min_idx) > 1:
        min_idx = min_idx[0]

    estim_dag = dags[min_idx]
    pval_var = soft_scores[min_idx]

    p_edge = [0 if estim_dag[i][j]==0 else 1-pval_var[j]
                          for i in range(len(estim_dag))
                          for j in range(len(estim_dag))]

    labels = true_dag.reshape(-1) #[1 if true_dag[i][j]==1 else 0 for i in range(len(true_dag))
        #for j in range(len(true_dag))]

    from sklearn.metrics import precision_recall_curve, auc, average_precision_score

    ap = average_precision_score(labels, p_edge)
    return ap
'''

#def aep_score_linc(true_dag, dag_gains):#
#
#    p_edge = dag_gains.reshape(-1)
#    labels = true_dag.reshape(-1)

#    import sklearn
#    ap = sklearn.metrics.average_precision_score(labels, p_edge)
#    return ap
