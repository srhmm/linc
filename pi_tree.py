from math import isclose
from collections import defaultdict
import numpy as np

from out import Out
from pi_search_clustering import pi_search_clustering
from pi_search_mdl import pi_mdl, pi_normalized_gain, pi_mdl_conf, pi_mdl_best
from cdt.metrics import SHD, SID

from upq import UPQ

from pi import PI
from pi_search_ilp import pi_search_ILP
from utils_pi import pi_convertfrom_pair, pi_enum
from vsn import Vsn


class PiEdge:
    def __init__(self, i, j,
                 score_ij, mdl_ij, pi_ij,
                 score_0, mdl_0, pi_0,
                 initial_gain):
        self.i = i
        self.j = j

        # Partition and Score of edge i->j in the empty graph
        self.score_ij = score_ij #score(i->j)
        self.mdl_ij = mdl_ij
        self.pi_ij = pi_ij # pi(i->j)
        self.score_0 = score_0 # score []->j
        self.mdl_0 = mdl_0
        self.pi_0 = pi_0
        self.initial_gain = initial_gain # score_0 - score_ij in the empty graph

    def __hash__(self):
        return hash((self.i, self.j))

    def __eq__(self, other):
        return (self.i == other.i
                & self.j == other.j)
        #no scores, edges will be added to prio queue and should be unique for each i,j, with updatable score/gain.

    def __str__(self):
        return '%s--> %s : %s,  %s' % (self.i, self.j, round(self.score_ij,2),  self.pi_ij)


def print_current(e, T):
    return '%s--> %s <--%s\t\tedge: %s, noedge: %s\t\t%s' % (e.i, e.j, [node for node in T.edges[e.j]],
                                                        round(T.eval_edge(e)[0], 1),
                                                        round(T.node_scores[e.j], 1), T.node_pi[e.j])
def print_deletion(i, j, T, score1, score2):
    return '%s    %s <--%s\tedge: %s, noedge: %s' % (i,j, [node for node in T.edges[j]],
                                                      round(score1,1), round(score2,1))
def print_update(j, T, new_parents, score1, score2):
    return '%s--> %s    %s\tnow: %s, before: %s' % ([node for node in new_parents], j, [node for node in T.edges[j]],
                                                 round(score1, 1), round(score2, 1))


class PiTree:
    def __init__(self, C_n, X_n, XYc, vsn):# mdl_gain, ilp, regression_per_group):

        # Invariants:
        # - "score" means the score used to evaluate edges against one another (using gain=score_current - score_updated),
        # can be either of:
        # - mdl: plain MDL score,
        # - mdl gain: gain in MDL score over no partition, MDL_pi0 - MDL_piupdated.
        #
        # Scores of edge (i->j) relative to empty model: stored in PiEdge
        # Scores of edges (ikl..->j) relative to the current model: stored in store_cache and edge_scores

        self.C_n = C_n
        self.X_n = X_n
        self.XYc = XYc
        self.vsn = vsn

        self.nodes = set([i for i in range(X_n)])
        self.adj = set()
        self.pa = defaultdict(set)
        self.ch = defaultdict(set)
        self.edges = defaultdict(defaultdict)

        # Each node Y has a score and partition for its cur parents {X1, ...Xn}
        self.node_scores = defaultdict(float)
        self.node_mdl = defaultdict(float)
        self.node_pi = defaultdict(list)

        # Record all child-parent combinations {X1, ... Xn} -> Y seen so far
        self.score_cache = {}
        self.mdl_cache = {}
        self.pi_cache = {}

        # the initial values for {}->Y
        self.empty_scores = defaultdict(float)
        self.empty_mdl = defaultdict(list)
        self.empty_pi = defaultdict(list)

        # the initial edges Xi->Y
        self.init_edges = defaultdict(defaultdict)

    def init_node(self, j, score_0, mdl_0, pi_0):
        self.node_scores[j] = score_0
        self.node_mdl[j] = mdl_0
        self.node_pi[j] = pi_0

        self.empty_scores[j] = score_0
        self.empty_mdl[j] = mdl_0
        self.empty_pi[j] = pi_0

    def get_edges(self):
        return self.edges #todo make priv

    def get_adj(self):
        adj = np.array([np.zeros(len(self.nodes)) for _ in range(len(self.nodes))])
        for (i,j) in self.adj:
            adj[i][j] = 1
        return adj

    def get_edge_list(self):
        return self.adj #todo make priv

    def get_node_partitions(self):
        return self.node_pi

    def get_node_mdlcost(self):
        return self.node_mdl

    def get_node_scores(self):
        return self.node_scores

    def get_graph_mdlcost(self):
        return sum(self.node_mdl.values())

    def get_graph_score(self): #score=score used to eval edges during search, depending on version, either mdl plain or mdl gain
        return sum(self.node_scores.values())

    def get_shd(self, truth):
        return SHD(truth, self.get_adj())

    def get_sid(self, truth):
        return SID(truth, self.get_adj())

    def is_edge(self, i, j):
        return (i,j) in self.adj

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

        self.adj.add((i, j))
        if not(j in self.edges):
            self.edges[j] = defaultdict(PiEdge)
        self.edges[j][i] = self.init_edges[j][i]

        self.node_scores[j] = score
        self.node_mdl[j] = mdl
        self.node_pi[j] = pi
        self.ch[i].add(j)
        self.pa[j].add(i)

    def remove_edge(self, i, j):
        """
        Remove Xi -> Xj. This will also update the score and partition for Xj to be that of Xpa(j)\Xi -> Xj.

        :param i: parent
        :param j: parent
        :return:
        """
        edge = self.init_edges[j][i]
        assert (self.is_edge(i,j))

        self.adj.remove((i,j))
        self.edges[j].pop(i)
        self.ch[i].remove(j)
        self.pa[j].remove(i)

        self.node_scores[j], self.node_mdl[j], self.node_pi[j] = self.eval_j(j, edge.score_0, edge.mdl_0, edge.pi_0)


    def update_edges(self, j, new_parents, new_children=None):
        """ Update graph around Xj, adding new parents and children.

            Old parents: remove, strict superset of Xs_1 cup Xs_2.
            New parents Xs_1: add these parents, i.e. keep old edges.
            New children Xs_2: remove from parents and add as children instead, i.e. flip old edges.

        :param j: target Xj
        :param new_parents: new parent set Xs_1
        :param new_children: new child set Xs_2, Xs1 and Xs2 strict subset of old parents
        :return:
        """

        old_parents = [i for i in self.nodes if self.is_edge(i,j)]
        for i in old_parents:
            self.remove_edge(i, j)

        for i in new_parents:
            edge = self.init_edges[j][i]
            _, score, mdl, pi, _ = self.eval_edge(edge)
            self.add_edge(i, j, score, mdl, pi)

        if new_children is None:
            return

        for i in new_children:
            edge = self.init_edges[i][j]
            _, score, mdl, pi, _ = self.eval_edge(edge)
            self.add_edge(j, i, score, mdl, pi)


    def unupdate_edges(self, j, old_parents, new_parents, new_children=None):
        for i in new_parents:
            self.remove_edge(i, j)
        if new_children is not None:
            for i in new_children:
                self.remove_edge(j, i)

        for i in old_parents:
            edge = self.init_edges[j][i]
            _, score, mdl, pi, _ = self.eval_edge(edge)
            self.add_edge(i, j, score, mdl, pi)

    def eval_j(self, j, score_0, mdl_0, pi_0) -> (int, int, list):
        """ Evaluate Xj's ingoing edges

        :param j: Xj
        :param score_0: initial score
        :param mdl_0:
        :param pi_0:
        :return: score (using score by which we build the tree, either full mdl or mdl gain), mdl (full mdl score), pi (partition for Xj)
        """
        pa = [node for node in range(self.X_n) if node in self.pa[j]]
        hash_key = f'j_{str(j)}_pa_{str(pa)}'

        if self.score_cache.__contains__(hash_key):
            assert self.mdl_cache.__contains__(hash_key) and self.pi_cache.__contains__(hash_key)
            score = self.score_cache[hash_key]
            mdl = self.mdl_cache[hash_key]
            pi = self.pi_cache[hash_key]
            return score, mdl, pi

        # CASE: no parents
        pa = [n for n in self.edges[j].keys()]
        if len(pa) == 0 or j not in self.edges:
            return score_0, mdl_0, pi_0

        # CASE: score parents
        score, mdl, pi = edge_gain_target(self.XYc, covariates=pa, target=j, vsn=self.vsn, C_n=self.C_n)
        # For score, lower is better. either direct MDL score (=mdl), or negative gain over the null model.

        self.score_cache[hash_key] = score
        self.mdl_cache[hash_key] = mdl
        self.pi_cache[hash_key] = pi
        return score, mdl, pi

    def eval_edge(self, edge: PiEdge =None, pa=None, j = None)-> (int, int, int, list, list):
        """
        Considers adding an edge Xi -> Xj to the current model.
        Computes or looks up the new partition for Xj, and the score gain of the model
        where Xi is a new parent of Xj in addition the current parents Xpa.

        :param edge: Xi->Xj
        :return: gain=score(Xpa->Xj) - score(Xpa,Xi->Xj), score_up=score(Xpa,Xi->Xj), similarly mdl_up, pi_up
        """
        if edge is not None:
            i, j = edge.i, edge.j
            pa = [node for node in range(self.X_n)
              if node in self.pa[j] or node == i]
        else:
            assert pa is not None and j is not None

        hash_key = f'j_{str(j)}_pa_{str(pa)}'

        if self.score_cache.__contains__(hash_key):
            assert self.mdl_cache.__contains__(hash_key) and self.pi_cache.__contains__(hash_key)
            score_up = self.score_cache[hash_key]
            mdl_up = self.mdl_cache[hash_key]
            pi_up = self.pi_cache[hash_key]

            score_cur = self.node_scores[edge.j]
            gain = (score_cur - score_up)
            return gain, score_up, mdl_up, pi_up, pa

        gain, score_up, mdl_up, pi_up, pa_up = self._edge_gain(edge, j, pa)

        self.score_cache[hash_key] = score_up
        self.mdl_cache[hash_key] = mdl_up
        self.pi_cache[hash_key] = pi_up
        return gain, score_up, mdl_up, pi_up, pa_up


    def _edge_gain(self, edge: PiEdge =None, pa=None, j = None)-> (int, int, int, list, list):
        """
        Gain of adding an edge to the model (private since we first want to check whether score is cached)

        :param edge: Xi->Xjc
        :return: gain=score(Xpa->Xj) - score(Xpa,Xi->Xj), score_up=score(Xpa,Xi->Xj), pi_up=partition(Xpa,Xi->Xj),
        """
        #TODO what was this for?
        #if edge.j not in self.edges:
        #    return edge.initial_gain, edge.score_ij, edge.mdl_ij, edge.pi_ij, [edge.j]
        if edge is not None:
            j = edge.j
        score_cur = self.node_scores[j]
        pi_cur = self.node_pi[j]
        mdl_cur = self.node_mdl[j]
        pa_cur = self.edges[j].keys()

        #if len(pa_cur) == 0:
            #TODO return edge.initial_gain, edge.score_ij, edge.mdl_ij, edge.pi_ij, [edge.i]

        if edge.i in pa_cur:
            return 0, score_cur, mdl_cur, pi_cur, pa_cur

        pa_up = [n for n in range(len(self.nodes)) if n in pa_cur or n == edge.i]

        score_up, mdl_up, pi_up = edge_gain_target(self.XYc, covariates=pa_up, target=edge.j, vsn=self.vsn, C_n=self.C_n)

        gain = (score_cur - score_up)
        return gain, score_up, mdl_up, pi_up, pa_up


    def initial_edges(self, q : UPQ, out : Out, subsample_size : int, cmp_score=True, score_cache=None
    )-> (UPQ):
        """
        Given the empty model, consider (1) score for Xj if there are no ingoing edges,
        (2) edges Xi->Xj for all pairs, their partition, and (1)-(2) the gain of adding the edge over no edge.

        :param q: Updatable Priority Queue
        :param out: printing
        :param subsample_size: how many samples in each context are considered (for efficiency, default=100).
        :return: q
         """

        #if score_cache is not None:
        #    self.score_cache= score_cache
        # We describe the "noise" in Y, given no input variables X (imitate this by setting X=0)
        sub_size = subsample_size
        #if self.vsn.rff: # special case X=0: instead of directly using rffs, subsample the number of rff features and do exact GP regression
        #    sub_size = 200 #TODO adaptable num features, TODO necessary or use rffs below?
        #Xic0 = np.array([[[0, ] for _ in range(sub_size)] for _ in range(self.C_n)])

        gpr_0 = None

        for target in self.nodes:

            # case: no ingoing edges
            Yc = np.array([self.XYc[c_i][:, target][: sub_size] for c_i in range(self.C_n)])
            Xic0 = np.array([[[Yc[c_i][x_i], ] for x_i in range(sub_size)] for c_i in range(self.C_n)])

            if self.vsn.vario_in_tree_search:
                lin_0 = PI(Xic0, Yc, np.random.RandomState(1), skip_regression=True)
                pi_0 = [[c_i] for c_i in range(self.C_n)]
                score_0 = lin_0.cmp_distances_linear(pi_0, emp=self.vsn.emp)
                mdl_0 = 0 #placeholder
            else:
                gpr_0 = PI(Xic0, Yc, np.random.RandomState(1), skip_regression_pairs=not self.vsn.regression_per_pair, rff=self.vsn.rff,pi_search=self.vsn.pi_search)
                gpr_0.cmp_distances()
                # Find the best partition and its basic MDL score
                if self.vsn.ilp_in_tree_search:
                    #if self.vsn.rff:
                    #    pair_indicator_vars, _, pair_contexts = pi_search_ILP(gpr_0.pair_mdl_dists, self.C_n, wasserstein=True)
                    #else:

                    if self.vsn.clustering:
                        pi_0, mdl_0, _, _, _ = \
                                pi_search_clustering(gpr_0.pair_mdl_gains, self.C_n, gpr_0,
                                                     self.vsn.regression_per_group, subsample_size)
                    else:
                        pair_indicator_vars, _, pair_contexts = pi_search_ILP(gpr_0.pair_mdl_gains, self.C_n, wasserstein=False)

                        pi_0 = pi_convertfrom_pair(pair_indicator_vars, pair_contexts, self.C_n)
                        mdl_0, _, _, _, _ = pi_mdl(gpr_0, pi_0, regression_per_group=self.vsn.regression_per_group,
                                                       subsample_size=subsample_size)
                else:
                    pi_0, mdl_0, _, _, _, _ = pi_mdl_best(gpr_0, self.vsn)
                #Alternative:
                # pi_0 = [[c_i] for c_i in self.C_n]
                #mdl_0, _, _, _, _ = pi_mdl(gpr_0, pi_0, regression_per_group=self.vsn.regression_per_group,
                #                           subsample_size=subsample_size)
                if self.vsn.mdl_gain:
                    score_0 = 0
                #elif self.vsn.rff:
                #    score_0 = -mdl_0
                else:
                    #assert not self.vsn.rff and not self.vsn.mdl_gain
                    score_0 = mdl_0


            hash_key = f'j_{str(target)}_pa_{str([])}'

            self.score_cache[hash_key] = score_0
            self.mdl_cache[hash_key] = mdl_0
            self.pi_cache[hash_key] = pi_0

            self.init_node(target, score_0, mdl_0, pi_0)

            # case: ingoing edge X->target
            q, edges_target = initial_edges_target(self.XYc, target, score_0, mdl_0, pi_0, gpr_0, q,
                                                   vsn=self.vsn, out=out, C_n=self.C_n, score=cmp_score)
            self.init_edges[target] = edges_target

            # Cache scores X->target
            if True : #cmp_score:
                for edge in edges_target.values():
                    assert (edge.j == target)
                    pa = [edge.i]
                    hash_key = f'j_{str(target)}_pa_{str(pa)}'

                    self.score_cache[hash_key] = edge.score_ij
                    self.mdl_cache[hash_key] = edge.mdl_ij
                    self.pi_cache[hash_key] = edge.pi_ij
        return q

    def eval_adj(self, adj):
        mdl = 0
        for j in self.nodes:
            pa = [i for i in self.nodes if adj[i][j]==1]
            _, mdl_j, _ = 0, 0,0 #self.eval_edge(edge=self.edges[j][i])#None, pa=pa, j=j) #TODO update
            mdl = mdl + mdl_j
        return mdl

    def conf_mat(self, adj):
        sig_mat = np.zeros((len(adj), len(adj)))
        gain_mat = np.zeros((len(adj), len(adj)))
        for i in range(len(adj)):
            for j in range(len(adj)):
                if adj[i][j] == 1:
                    adj_rev = adj.copy()
                    adj_rev[i][j] = 0
                    adj_rev[j][i] = 1

                    mdl_causal, mdl_rev = self.eval_adj(adj), self.eval_adj(adj_rev)
                    gain = mdl_rev - mdl_causal
                    gain_mat[i][j] = gain
                    if is_insignificant(gain, False, False, mdl_gain=False, alpha=0.05):
                        sig_mat[i][j] = 0
                    else:
                        sig_mat[i][j] = gain
        return gain_mat, sig_mat

    def has_cycle(self, i,j): # from https://www.geeksforgeeks.org/detect-cycle-in-a-graph/
        visited = [False] * (len(self.nodes) + 1)
        recStack = [False] * (len(self.nodes) + 1)
        for node in range(len(self.nodes)):
            if visited[node] == False:
                if self._has_cycle_util(node,visited,recStack,i,j) == True:
                    return True
        return False

    def _has_cycle_util(self, v, visited, recStack, i,j):
        visited[v] = True
        recStack[v] = True
        neighbors = []
        if v in self.ch:
            neighbors = self.ch[v]
        if v==i:
             neighbors = [n for n in range(len(self.nodes)) if n in neighbors or n==j]

        for neighbour in neighbors:
            if visited[neighbour] == False:
                if self._has_cycle_util(neighbour, visited, recStack, i,j) == True:
                    return True
            elif recStack[neighbour] == True:
                return True

        recStack[v] = False
        return False

    def __str__(self):
        substrings = []
        for node in self.nodes:
            if self.pa[node]:
                parents_str = ','.join(map(str, self.pa[node]))
                gain = self.empty_scores[node] - self.node_scores[node]
                substrings.append('\n\t%s->%s : %s = %s - %s, %s' % (parents_str, node, round(gain, 2),  round(self.node_scores[node] ,2),  round(self.empty_scores[node], 2), self.node_pi[node]))
            else:
                substrings.append('\n\t[%s] : %s ' % (node, round(self.node_scores[node], 2)))
        return ''.join(substrings)



# UTILS

def is_insignificant (gain, vario_score, emp, mdl_gain, alpha = 0.05):
    if vario_score and emp:
            return False #TODO can use precomputed conf ints
    if mdl_gain:
        return (gain < 0 or 2**(-gain) > alpha) # TODO does this fit the gains over null models as well??
    return (gain < 0 or 2**(-gain) > alpha)


def eval_edge_orientation(i, j, T:PiTree):
    edge_ij = T.init_edges[j][i]
    edge_ji =  T.init_edges[i][j]

    if T.is_edge(i,j):
        T.remove_edge(i, j)
        gain_ij, score_ij, mdl_ij, pi_ij, _ = T.eval_edge(edge_ij)
        T.add_edge(j, i, score_ij, mdl_ij, pi_ij)
    else:
        gain_ij, score_ij, mdl_ij, pi_ij, _ = T.eval_edge(edge_ij)
    if T.is_edge(j, i):
        T.remove_edge(j, i)
        gain_ji, score_ji, mdl_ji, pi_ji, _ = T.eval_edge(edge_ji)
        T.add_edge(j, i, score_ji, mdl_ji, pi_ji)
    else:
        gain_ji, score_ji, mdl_ji, pi_ji, _ = T.eval_edge(edge_ji)

    print("\t\t\t GAIN corr orientation; false orientation: ", gain_ij, "; ", gain_ji)
    return gain_ij, score_ij, mdl_ij, pi_ij, \
           gain_ji, score_ji, mdl_ji, pi_ji,

def eval_reverse_edge(edge_forward,  T: PiTree):
    """ Gain of flipping an edge i->j.

        Have: {i, pa(j)-{i}}-> j in T.
        Returns: gain of {j, pa(j)u{i}}-> i over {j, pa(j)}-> i.
            Note: no need to remove edge i->j here, since we only need local gains around i here, for which the parents of j are irrelevant.

    :param edge_forward: edge i->j that we consider flipping
    :param T: tree

    """

    j = edge_forward.j
    i = edge_forward.i

    #mdl_current = T.get_graph_score()# T.get_graph_mdlcost()
    #T.remove_edge(i, j)

    #Score for i<-j
    edge_backward = T.init_edges[i][j]
    #new_pa = [n for n in T.nodes if T.is_edge(n,i) or n==j]

    gain, score, mdl, pi, _ = T.eval_edge(edge_backward)
    print("\t\t\t GAIN", gain)
    #T.add_edge(j, i, score, mdl, pi)
    # T.remove_edge(j,i) #(edge_backward)

    #mdl_updated = T.get_graph_score() #_mdlcost()
    #gain, score, mdl, pi, _ = T.eval_edge(edge_forward)
    #T.add_edge(i, j, score, mdl, pi) #add again since T is a reference
    #assert( T.get_graph_mdlcost() ==mdl_current)

    #gain = mdl_current - mdl_updated
    return gain

def eval_local_edges(j, T:PiTree,
                     new_parents, new_children=None,
                     always_mdl=True): # with false, less edges removed during edge removal
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
    old_parents = [i for i in T.nodes if T.is_edge(i, j)]

    if always_mdl: old_score = T.get_graph_mdlcost()
    else: old_score = T.get_graph_score()

    T.update_edges(j, new_parents, new_children)
    if always_mdl: new_score = T.get_graph_mdlcost()
    else: new_score = T.get_graph_score()

    T.unupdate_edges(j, old_parents, new_parents, new_children)
    ancient_score = T.get_graph_score()
    if always_mdl: ancient_score = T.get_graph_mdlcost()

    assert(ancient_score==old_score)
    for i in old_parents:
        assert T.is_edge(i, j)
    gain = old_score - new_score
    return gain


def initial_edges_target(XYc, target: int,
                         score_0 : int, mdl_0: int, pi_0: list,
                         gpr_0,
                         q : UPQ, vsn : Vsn,
                         C_n: int, out : Out, score =True )-> (UPQ, list):
    vb = False
    if vsn.subsample_size is None:
        subsample_size = min([len(XYc[c_i]) for c_i in range(C_n)])
    else:
        subsample_size = vsn.subsample_size
    if vb:
        out.printto("\tY = "+ str(target))
    covariates = [i for i in range(len(XYc[0][1]))
                  if i is not target]
    init_edges = defaultdict(PiEdge)

    # Consider DAG with a single edge (Xi -> Xj) for each Xi
    for node_i in covariates:
        if not score:
            pi_edge_i = PiEdge(node_i, target, 0, 0, [[c_i for c_i in range(C_n)]], score_0, mdl_0, pi_0, 0)

            q.add_task(task=pi_edge_i, priority=0)
            init_edges[node_i] = pi_edge_i
            continue

        Xic = np.array([XYc[c_i][:, [node_i]][: subsample_size] for c_i in range(C_n)])
        Yc = np.array([XYc[c_i][:, target][: subsample_size] for c_i in range(C_n)])

        if vsn.vario_in_tree_search:
            # Linear regression in each context
            linreg = PI(Xic, Yc, np.random.RandomState(1), skip_regression=True)
            score = -np.inf
            pi = None
            for pi_cur in pi_enum(C_n, permute=True):
                score_cur = linreg.cmp_distances_linear(pi_cur, emp=vsn.emp)
                if score_cur > score:
                    score = score_cur
                    pi = pi_cur
            gain = score_0 - score
            mdl_score = score #placeholder
        else:
            # GP regression in each context
            gpr = PI(Xic, Yc, np.random.RandomState(1), skip_regression_pairs=not vsn.regression_per_pair, rff=vsn.rff,pi_search=vsn.pi_search)
            gpr.cmp_distances()

            # Find the best partition for Xj and its full MDL score
            if vsn.ilp_in_tree_search:
                #if vsn.rff: #TODO fix rff partitions
                #    pair_indicator_vars, _, pair_contexts = pi_search_ILP(gpr.pair_mdl_dists, C_n, wasserstein=True)
                #else:
                pair_indicator_vars, _, pair_contexts = pi_search_ILP(gpr.pair_mdl_gains, C_n, wasserstein=False)
                pi = pi_convertfrom_pair(pair_indicator_vars, pair_contexts, C_n)
                mdl_score, _, _, _, _ = pi_mdl(gpr, pi, regression_per_group=vsn.regression_per_group,
                                               subsample_size=subsample_size)
            else:
                pi, mdl_score, _, _, _, _ = pi_mdl_best(gpr, vsn)

            # Invariance(good Partition/model): gain pf the partition over no groups
            if vsn.mdl_gain:
                gain_over_pi0, _, _ = pi_mdl_conf(gpr, pi, C_n, regression_per_group=vsn.regression_per_group,
                                                  subsample_size=subsample_size)
                gain = gain_over_pi0 #gains: larger=better
                gain2, _, _ = pi_normalized_gain(pi, pi_0, gpr, gpr_0, vsn.regression_per_group, C_n, subsample_size)
                #print("\t\t" + str(node_i)+ "->"+ str( target) + ": L= "  + str(round(gain, 2))+ "="+ str(round(-gain2,2))+"(smaller is better)")
                #assert (isclose(gain, -gain2, rel_tol=1e-9, abs_tol=0.0))
                score = -gain_over_pi0 #scores:  smaller = better
            else:
                score = mdl_score
                gain = score_0-score
        if vb:
            out.printto("\t\t" + str(node_i)+ "->"+ str( target) + ": L = "  + str(round(gain, 2)))

        negative_gain = -gain * 100 # hack for prioritization in q

        pi_edge_i = PiEdge(node_i, target, score, mdl_score,
                           pi,score_0, mdl_0, pi_0, gain)

        q.add_task(task=pi_edge_i, priority=negative_gain)
        init_edges[node_i] = pi_edge_i
    return q, init_edges


def edge_gain_target(XYc, covariates, target:int,
                     vsn: Vsn, C_n :int)-> (int, int, list):

    if vsn.subsample_size is None:
        subsample_size = min([len(XYc[c_i]) for c_i in range(C_n)])
    else: subsample_size = vsn.subsample_size

    if len(covariates) == 0:
        #if vsn.mdl_gain: sub_size = 200
        #else:
        sub_size = subsample_size
        #Xc = np.array([[[0, ] for _ in range(sub_size)] for _ in range(C_n)])
        Yc = np.array([XYc[c_i][:, target][: sub_size] for c_i in range(C_n)])
        Xc = np.array([[[Yc[c_i][x_i], ] for x_i in range(sub_size)] for c_i in range(C_n)])
    else:
        Yc = np.array([XYc[c_i][:, target][: subsample_size] for c_i in range(C_n)])
        Xc = np.array([XYc[c_i][:, covariates][: subsample_size] for c_i in range(C_n)])

    if vsn.vario_in_tree_search:
        linreg = PI(Xc, Yc, np.random.RandomState(1), skip_regression=True)
        score = -np.inf
        pi = None
        for pi_cur in pi_enum(C_n, permute=True):
            score_cur = linreg.cmp_distances_linear(pi_cur, emp=vsn.emp)
            if score_cur > score: score, pi = score_cur, pi_cur
        mdl_score = 0 #placeholder
    else:
        if len(covariates) == 0 & vsn.mdl_gain: rff = False
        else: rff = vsn.rff

        gpr = PI(Xc, Yc, np.random.RandomState(1), skip_regression_pairs=not vsn.regression_per_pair, rff=rff,pi_search=vsn.pi_search)

        gpr.cmp_distances()
        if vsn.ilp_in_tree_search:
            #if vsn.rff:
            #    pair_indicator_vars, _, pair_contexts = pi_search_ILP(gpr.pair_mdl_dists, C_n, wasserstein=True)
            #else:
            pair_indicator_vars, _, pair_contexts = pi_search_ILP(gpr.pair_mdl_gains, C_n, wasserstein=False)
            pi = pi_convertfrom_pair(pair_indicator_vars, pair_contexts, C_n)
            mdl_score, _, _, _, _ = pi_mdl(gpr, pi, vsn.regression_per_group, subsample_size)
        else: pi, mdl_score, _, _, _, _ = pi_mdl_best(gpr, vsn)

        if vsn.mdl_gain:
            gain_over_0, _, _ = pi_mdl_conf(gpr, pi, C_n, vsn.regression_per_group,
                                            subsample_size) #simply does MDL(model, pi0) - MDL(model, pi)
            score = -gain_over_0 #to make this consistent version above: smaller is better

        else:
            score = mdl_score


    return score, mdl_score, pi



def edge_score_pi(XYc, covariates, target:int, pi:list,
                     vsn: Vsn, C_n :int)-> (int, int, list):

    if vsn.subsample_size is None: subsample_size = len(XYc[0])
    else: subsample_size = vsn.subsample_size

    if len(covariates) == 0:
        #if vsn.mdl_gain: sub_size = 200
        #else:
        sub_size = subsample_size
        Xc = np.array([[[0, ] for _ in range(sub_size)] for _ in range(C_n)])
        Yc = np.array([XYc[c_i][:, target][: sub_size] for c_i in range(C_n)])
    else:
        Yc = np.array([XYc[c_i][:, target][: subsample_size] for c_i in range(C_n)])
        Xc = np.array([XYc[c_i][:, covariates][: subsample_size] for c_i in range(C_n)])

    if vsn.vario_in_tree_search:
        linreg = PI(Xc, Yc, np.random.RandomState(1), skip_regression=True)
        score = linreg.cmp_distances_linear(pi, emp=vsn.emp)
        mdl_score = 0 #placeholder
    else:
        #if len(covariates) == 0 & vsn.mdl_gain: rff = False
        #else:
        rff = vsn.rff

        gpr = PI(Xc, Yc, np.random.RandomState(1), skip_regression_pairs=not vsn.regression_per_pair, rff=rff,pi_search=vsn.pi_search)
        gpr.cmp_distances()

        mdl_score, _, _, _, _ = pi_mdl(gpr, pi, vsn.regression_per_group, subsample_size)


        if vsn.mdl_gain:
            gain_over_0, _, _ = pi_mdl_conf(gpr, pi, C_n, vsn.regression_per_group,
                                            subsample_size) #simply does MDL(model, pi0) - MDL(model, pi)
            score = -gain_over_0 #to make this consistent version above: smaller is better

        else:
            score = mdl_score


    return score, mdl_score, pi
