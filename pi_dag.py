from collections import defaultdict

import numpy as np
from cdt.metrics import SHD, SID

from out import Out
from pi_search_clustering import pi_search_clustering
from pi_search_mdl import pi_mdl, pi_mdl_conf, pi_mdl_best
from pi_tree import is_insignificant
from pi import PI
from pi_search_ilp import pi_search_ILP
from upq import UPQ
from utils_pi import pi_convertfrom_pair, pi_enum
from vsn import Vsn



class PiDAG:
    def __init__(self, C_n, X_n, XYc, vsn):
        self.C_n = C_n
        self.X_n = X_n
        self.XYc = XYc
        self.vsn = vsn
        self._nodes = set([i for i in range(X_n)])
        #self._edge_list = defaultdict(defaultdict)

        # Each node Y has a score and partition for its cur parents {X1, ...Xn}
        self.node_scores = defaultdict(float)
        self.node_mdl = defaultdict(float)
        self.node_pi = defaultdict(list)
        self.node_pa = defaultdict(set)
        self.node_ch = defaultdict(set)

        self.pair_edges = [[None for j in range(X_n)] for i in range(X_n)]

        # Record all child-parent combinations {X1, ... Xn} -> Y seen so far
        self.score_cache = {}
        self.mdl_cache = {}
        self.pi_cache = {}
    def children_of(self, j):
        return [i for i in self._nodes if self.is_edge(j, i)]#[i for i in self.node_ch[j]] #
    def parents_of(self, j):
        return [i for i in self._nodes if self.is_edge(i, j)] #[i for i in self.node_pa[j]]
    def get_nodes(self):
        return self._nodes
    def get_adj(self):
        adj = np.array([np.zeros(len(self._nodes)) for _ in range(len(self._nodes))])
        for i in self._nodes:
            for j in self._nodes:
                if self.is_edge(i,j):
                    adj[i][j] = 1
        return adj


    def get_graph_mdlcost(self):
        return self.eval_other_dag(self.get_adj(), rev=False)
        #TODO return sum(self.node_mdl.values())
    def get_shd(self, truth):
        return SHD(truth, self.get_adj())

    def get_sid(self, truth):
        return SID(truth, self.get_adj())

    def is_edge(self, i, j):
        return i in self.node_pa[j]#(i,j) in self._edge_list


    def init_node(self, j, score_0, mdl_0, pi_0):
        self.node_scores[j] = score_0
        self.node_mdl[j] = mdl_0
        self.node_pi[j] = pi_0

    def add_edge(self, i: int, j: int,
                 score: int, mdl: int, pi: list):
        """
        Add Xi -> Xj

        :param i: parent
        :param j: target
        :param score: score(parents(Xj) + {Xi}, Xj).  _, score, _, _ = T.eval_edge(edge)
        :param mdl : mdl(parents(Xj) + {Xi}, Xj). _, _, mdl, _ = T.eval_edge(edge)
        :param pi: partition(parents(Xj) + {Xi}, Xj). _, _, pi, _ = T.eval_edge(edge)
        :return:
        """
        self.node_scores[j] = score
        self.node_mdl[j] = mdl
        self.node_pi[j] = pi
        self.node_pa[j].add(i)
        self.node_ch[i].add(j)

    def remove_edge(self, i, j):
        """
        Remove Xi -> Xj. This will also update the score and partition for Xj to be that of Xpa(j)\Xi -> Xj.

        :param i: parent
        :param j: parent
        :return:
        """
        assert (i in self.node_pa[j])
        self.node_pa[j].remove(i)
        self.node_ch[i].remove(j)
        pa_up = self.parents_of(j) #[i for i in self.node_pa[j]]
        self.node_scores[j], self.node_mdl[j], self.node_pi[j] = self.eval_edge(j, pa_up)


    # case: iterating over DAGs in MEC
    def eval_other_dag(self, adj, gain=False, rev=True):
        mdl = 0
        for j in self._nodes:
            if rev: # whether adj[j][i]==1 means i->j or vice versa
                pa = [i for i in self._nodes if adj[j][i]==1] #adj[i][j]==1]
            else:
                pa = [i for i in self._nodes if adj[i][j]==1]
            score_j, mdl_j, _ = self.eval_edge(j, pa)
            if gain:
                mdl = mdl + score_j
            else:
                mdl = mdl + mdl_j
        return mdl

    def eval_edge_addition(self, j, i) : #pa):
        pa_cur = self.parents_of(j) #[p for p in self._nodes if self.is_edge(p,j)]
        pa_up = [p for p in self._nodes if (p in pa_cur or p==i)]

        score_cur, mdl_cur, pi_cur = self.eval_edge(j, pa_cur)
        score_up, mdl_up, pi_up = self.eval_edge(j, pa_up)
        self.add_edge(j, i, score_up, mdl_up, pi_up)
        cost_up = self.get_graph_mdlcost() #self.eval_other_dag()
        self.remove_edge(j, i)
        cost_cur = self.get_graph_mdlcost() #self.eval_other_dag()
        gain = (cost_cur - cost_up)
        return gain, score_up, mdl_up, pi_up, pa_up, score_cur, mdl_cur, pi_cur, pa_cur

    def eval_edge_addition_gain(self, j, i) : #pa):
        pa_cur = self.parents_of(j) #[p for p in self._nodes if self.is_edge(p,j)]
        pa_up = [p for p in self._nodes if (p in pa_cur or p==i)]

        score_cur, mdl_cur, pi_cur = self.eval_edge(j, pa_cur)
        score_up, mdl_up, pi_up = self.eval_edge(j, pa_up)
        gain = (score_cur - score_up)
        return gain, score_up, mdl_up, pi_up, pa_up, score_cur, mdl_cur, pi_cur, pa_cur

    def eval_edge_flip(self, j, ch):
        ''' Evaluates {j} <- {ch},pa_j against {j} u pa_ch -> pa_j '''

        cost_cur = self.get_graph_mdlcost()
        assert self.is_edge(j, ch)
        self.remove_edge(j, ch)
        gain, score_ji, mdl_ji, pi_ji ,  _, _, _, _, _ = self.eval_edge_addition(ch, j) #j -> ch
        self.add_edge(ch, j, score_ji, mdl_ji, pi_ji)
        cost_flip = self.get_graph_mdlcost()

        assert self.is_edge(ch, j)
        self.remove_edge(ch, j)
        gain_ij, score_ij, mdl_ij, pi_ij, _, _, _, _, _ = self.eval_edge_addition(j, ch) #ch -> j
        self.add_edge(j, ch, score_ij, mdl_ij, pi_ij )


        #Compares the wrong scores: (once locally for j, once for ch)
        #score_cur, mdl_cur, pi_cur = self.eval_edge_addition(ch, j) #j -> ch
        #score_flip, mdl_flip, pi_flip = self.eval_edge_addition(j, ch) #ch -> j
        gain = (cost_cur - cost_flip)
        return gain

    def eval_edge(self, j, pa)-> (int, int, int, list, list):
        """
        Considers a multiedge pa(Xj)->Xj.

        :param edge: pa(Xj)->Xj
        :return: score_up=score(Xpa->Xj), mdl_up, pi_up=partition(Xpa->Xj)
        """
        hash_key = f'j_{str(j)}_pa_{str(pa)}'

        if self.score_cache.__contains__(hash_key):
            assert self.mdl_cache.__contains__(hash_key) and self.pi_cache.__contains__(hash_key)
            score_up = self.score_cache[hash_key]
            mdl_up = self.mdl_cache[hash_key]
            pi_up = self.pi_cache[hash_key]
            return score_up, mdl_up, pi_up

        score_up, mdl_up, pi_up = local_edge_score(self.XYc, covariates=pa, target=j, vsn=self.vsn, C_n=self.C_n)

        # Weird case of inf scores - use worst-case (no parent) score to avoid inf scores over the whole causal model
        if score_up == np.inf:
            hash_key = f'j_{str(j)}_pa_{str([])}'
            if self.score_cache.__contains__(hash_key):
                #print("(LINC) setting score to", self.score_cache[hash_key])
                return self.score_cache[hash_key], self.mdl_cache[hash_key],  self.pi_cache[hash_key]

            score_up, mdl_up, pi_up = local_edge_score(self.XYc, covariates=[], target=j, vsn=self.vsn, C_n=self.C_n)
            #print("(LINC) setting score to", round(score_up,2))

        self.score_cache[hash_key] = score_up
        self.mdl_cache[hash_key] = mdl_up
        self.pi_cache[hash_key] = pi_up
        return score_up, mdl_up, pi_up

    def initial_edges(self, q : UPQ)-> (UPQ):

        for j in self._nodes:
            pa = []
            score, mdl, pi = self.eval_edge(j, pa)
            others = [i for i in self._nodes if not(i==j)]
            for i in others:
                score_ij, mdl_ij, pi_ij = self.eval_edge(j, [i])

                edge_ij = q_entry(i, j, score_ij, mdl_ij, pi_ij, score, mdl, pi)

                q.add_task(task=edge_ij, priority=-score_ij * 100)  #negating score for prioritization in q
                self.pair_edges[i][j] = edge_ij
        return q

    def eval_local_edges(self, j, new_pa, new_ch):
        """
        Evaluate graph changes  locally  around a target Xj, i.e. adding new parents and or children.
        Args:
            j: Target Xj
            T: Tree
            new_parents: list of causal parents (during search: a subset of current parents)
            new_children: list of causal children (during search: a subset of current parents, edges to be flipped)
            always_mdl: whether to use the full MDL score in the backward phase (always, even if mdl gain =True)

        Returns:

        """
        old_ch = self.children_of(j)#[i for i in self._nodes if self.is_edge(j, i)]
        old_pa = self.parents_of(j) #[i for i in self._nodes if self.is_edge(i, j)]

        old_score = self.get_graph_mdlcost()

        self.update_local_edges(j, new_pa, new_ch)
        new_score = self.get_graph_mdlcost()

        self.update_local_edges(j, old_pa, old_ch)
        #self.unupdate_edges(j, old_pa, new_pa, new_ch)
        ancient_score = self.get_graph_mdlcost()

        #assert (ancient_score == old_score)
        for i in old_pa:
            assert self.is_edge(i, j)
        for i in old_ch:
            assert self.is_edge(j, i)
        gain = old_score - new_score
        return gain


    def update_local_edges(self, j, new_parents, new_children=None):
        """ Update graph around Xj, adding new parents and children.

            Old parents: remove, strict superset of Xs_1 cup Xs_2.
            New parents Xs_1: add these parents, i.e. keep old edges.
            New children Xs_2: remove from parents and add as children instead, i.e. flip old edges.

        :param j: target Xj
        :param new_parents: new parent set Xs_1
        :param new_children: new child set Xs_2, Xs1 and Xs2 strict subset of old parents
        :return:
        """

        old_parents = self.parents_of(j)#[i for i in self._nodes if self.is_edge(i,j)]
        for i in old_parents:
            self.remove_edge(i, j)
        old_children = self.children_of(j)#[i for i in self._nodes if self.is_edge(j, i)]
        for i in old_children:
            self.remove_edge(j, i)

        for i in new_parents:
            _, score, mdl, pi, _,_,_,_,_ = self.eval_edge_addition(i, j)
            self.add_edge(i, j, score, mdl, pi)

        if new_children is None:
            return

        for i in new_children:
            _, score, mdl, pi, _,_,_,_,_ = self.eval_edge_addition(j, i)
            self.add_edge(j, i, score, mdl, pi)


    #
    # def unupdate_edges(self, j, old_parents, new_parents, new_children=None):
    #     for i in new_parents:
    #         self.remove_edge(i, j)
    #     if new_children is not None:
    #         for i in new_children:
    #             self.remove_edge(j, i)
    #
    #     for i in old_parents:
    #         edge = self.init_edges[j][i]
    #         _, score, mdl, pi, _ = self.eval_edge(edge)
    #         self.add_edge(i, j, score, mdl, pi)

    def conf_mat(self, adj):
        sig_mat = np.zeros((len(adj), len(adj)))
        gain_mat = np.zeros((len(adj), len(adj)))
        for i in range(len(adj)):
            for j in range(len(adj)):
                if adj[i][j] == 1:
                    adj_rev = adj.copy()
                    adj_rev[i][j] = 0
                    adj_rev[j][i] = 1

                    mdl_causal, mdl_rev = self.eval_other_dag(adj), self.eval_other_dag(adj_rev)
                    gain = mdl_rev - mdl_causal
                    gain_mat[i][j] = gain
                    if is_insignificant(gain, False, False, mdl_gain=False, alpha=0.05):
                        sig_mat[i][j] = 0
                    else:
                        sig_mat[i][j] = gain
        return gain_mat, sig_mat

    def has_cycle(self, i,j): # from https://www.geeksforgeeks.org/detect-cycle-in-a-graph/
        visited = [False] * (len(self._nodes) + 1)
        recStack = [False] * (len(self._nodes) + 1)
        for node in range(len(self._nodes)):
            if visited[node] == False:
                if self._has_cycle_util(node,visited,recStack,i,j) == True:
                    return True
        return False

    def _has_cycle_util(self, v, visited, recStack, i,j):
        visited[v] = True
        recStack[v] = True
        neighbors = []
        if v in self.node_ch:
            neighbors = self.node_ch[v]
        if v==i:
             neighbors = [n for n in range(len(self._nodes)) if n in neighbors or n==j]

        for neighbour in neighbors:
            if visited[neighbour] == False:
                if self._has_cycle_util(neighbour, visited, recStack, i,j) == True:
                    return True
            elif recStack[neighbour] == True:
                return True

        recStack[v] = False
        return False


# SCORES

def local_edge_score(XYc, covariates, target:int,
                     vsn: Vsn, C_n :int, speedup = True)-> (int, int, list):

    if vsn.subsample_size is None: sub_size = len(XYc[0])
    else: sub_size = vsn.subsample_size

    Yc = np.array([XYc[c_i][:, target][: sub_size] for c_i in range(C_n)])
    if len(covariates) == 0:
        #Regress Y onto itself, since there is no information from a parent
        #if vsn.rff:
            Xc = np.array([[[Yc[c_i][x_i], ] for x_i in range(sub_size)] for c_i in range(C_n)])
        #else:
        #    Xc = np.array([[[0, ] for x_i in range(sub_size)] for c_i in range(C_n)])
    else:
        Xc = np.array([XYc[c_i][:, covariates][: sub_size] for c_i in range(C_n)])

    #score, mdl_score, pi = cmp_edge_score(Xc, Yc, C_n, sub_size, vsn, (speedup and len(covariates) == 0))
    speedup_no_parents = (speedup and len(covariates) == 0)
    #return score, mdl_score, pi

#def cmp_edge_score(Xc, Yc, C_n, subsample_size, vsn, speedup_no_parents):
    if vsn.vario_in_tree_search:
        linreg = PI(Xc, Yc, np.random.RandomState(1), skip_regression=True)
        score = -np.inf
        pi = None
        for pi_cur in pi_enum(C_n, permute=True):
            score_cur = linreg.cmp_distances_linear(pi_cur, emp=vsn.emp)
            if score_cur > score: score, pi = score_cur, pi_cur
        mdl_score = 0
    else:
        gpr = PI(Xc, Yc, np.random.RandomState(1), skip_regression_pairs=not vsn.regression_per_pair, rff=vsn.rff,pi_search=vsn.pi_search)

        if speedup_no_parents:
            pi = [[c_i for c_i in range(C_n)]]
            mdl_score, _, _, _, _ = pi_mdl(gpr, pi, vsn.regression_per_group, sub_size)
        else:
            if vsn.ilp:
                if vsn.ilp_partitions:
                    gpr.cmp_distances()

                    if vsn.clustering:
                        pi, mdl_score, _, _, _ = \
                                pi_search_clustering(gpr.pair_mdl_gains, C_n, gpr,
                                                     vsn.regression_per_group, sub_size)
                    else:
                        pair_indicator_vars, _, pair_contexts = pi_search_ILP(gpr.pair_mdl_gains, C_n, wasserstein=False)
                        pi = pi_convertfrom_pair(pair_indicator_vars, pair_contexts, C_n)
                        mdl_score, _, _, _, _ = pi_mdl(gpr, pi, vsn.regression_per_group, sub_size)
                else:
                    pi = [[c_i for c_i in range(C_n)]]
                    mdl_score, _, _, _, _ = pi_mdl(gpr, pi, vsn.regression_per_group, sub_size)
            else: pi, mdl_score, _, _, _, _ = pi_mdl_best(gpr, vsn)

        if vsn.mdl_gain:
            if (len(covariates)==0):
                score=0
            else:
                gain_over_0, _, _ = pi_mdl_conf(gpr, pi, C_n, vsn.regression_per_group,
                                                sub_size) #simply does MDL(model, pi0) - MDL(model, pi)
                score = -gain_over_0 #to make this consistent version above: smaller is better
        else:
            score = mdl_score

    return score, mdl_score, pi

class q_entry:
    def __init__(self, i, j, score_ij, mdl_ij, pi_ij,
                 score_0, mdl_0, pi_0):
        self.i = i
        self.j = j

        # Partition and Score of edge i->j in the empty graph
        self.score_ij = score_ij
        self.mdl_ij = mdl_ij
        self.pi_ij = pi_ij
        # Partition and score of []->j
        self.score_0 = score_0
        self.mdl_0 = mdl_0
        self.pi_0 = pi_0

    def __hash__(self):
        return hash((self.i, self.j))

    def __eq__(self, other):
        return (self.i == other.i
                & self.j == other.j)
