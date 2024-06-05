import logging
from collections import defaultdict

import numpy as np

from gpcd.other.util import is_insignificant
from gpcd.scoring import score_edge, GPParams
from gpcd.util.upq import UPQ


class DAG:
    def __init__(
        self,
        data_C,
        gp_hyperparams: GPParams,
        logger: logging,
        verbosity=0,
        is_true_edge=lambda i: lambda j: "",
    ):
        self.C = len(data_C)
        self.N = [data_C[k].shape[1] for k in data_C][0]
        self._nodes = set([i for i in range(self.N)])
        self.data_C = data_C
        self.gp_hyperparams = gp_hyperparams
        self.is_true_edge = is_true_edge
        self.logger = logger
        self.verbosity = verbosity

        # Each node Y with its current parents {X1,...Xn}
        self.node_mdl = defaultdict(float)
        self.node_pa = defaultdict(set)
        self.node_ch = defaultdict(set)

        self.pair_edges = [[None for _ in range(self.N)] for _ in range(self.N)]

        # Record all child-parent combinations {X1, ... Xn} -> Y seen so far
        self.mdl_cache = {}

    def children_of(self, j):
        return [i for i in self._nodes if self.is_edge(j, i)]

    def parents_of(self, j):
        return [i for i in self._nodes if self.is_edge(i, j)]

    def get_nodes(self):
        return self._nodes

    def get_adj(self):
        adj = np.array([np.zeros(len(self._nodes)) for _ in range(len(self._nodes))])
        for i in self._nodes:
            for j in self._nodes:
                if self.is_edge(i, j):
                    adj[i][j] = 1
        return adj

    def get_links(self):  # returns instantaneous links
        links = {}
        for j in self._nodes:
            links[j] = []
            for i in self._nodes:
                if self.is_edge(i, j):
                    links[j].append(((i, 0), 1, None))
        return links

    def get_graph_mdlcost(self):
        return self.eval_other_dag(self.get_adj(), rev=False)

    def is_edge(self, i, j):
        return i in self.node_pa[j]

    def exists_anticausal_edge(self, parent, node):
        if self.is_edge(node, parent):
            return True
        return False

    def add_edge(self, i, j: int, score: int, gain):
        """
        Add Xi -> Xj

        :param i: parent
        :param j: target
        :param score: score(parents(Xj) + {Xi}, Xj).  _, score, _, _ = T.eval_edge(edge)
        :return:
        """
        self.node_mdl[j] = score
        self.node_pa[j].add(i)
        self.node_ch[i].add(j)

        if self.verbosity > 0:
            self.logger.info(
                f"\tAdd edge {i} -> {j}: s={np.round(gain[0][0], 2)} pa={self.parents_of(j)} \t{self.is_true_edge(i)(j)}"
            )

    def remove_edge(self, i, j):
        """
        Remove Xi -> Xj. This will also update the score for Xj to be that of Xpa(j)\Xi -> Xj.

        :param i: parent
        :param j: child
        :return:
        """
        assert i in self.node_pa[j]
        self.node_pa[j].remove(i)
        self.node_ch[i].remove(j)
        pa_up = self.parents_of(j)
        self.node_mdl[j] = self.eval_edge(j, pa_up)

        if self.verbosity > 0:
            self.logger.info(
                f"\tRemove edge {i} -> {j}: s={np.round(self.node_mdl[j][0][0], 2)} pa={pa_up}\t{self.is_true_edge(i)(j)}"
            )

    def info_adj(self):
        pass

    def remove_all_edges(self):
        for i in self._nodes:
            for j in self._nodes:
                if not (i in self.node_pa[j]):
                    continue
                self.remove_edge(i, j)

    def eval_edge_addition(self, j, i):  # j is target, i is parent
        pa_cur = self.parents_of(j)
        pa_up = pa_cur.copy()
        pa_up.append(i)

        score_cur = self.eval_edge(j, pa_cur)
        score_up = self.eval_edge(j, pa_up)

        gain = score_cur - score_up
        return gain, score_up, pa_up, score_cur, pa_cur

    def eval_edge_flip(self, j, ch):
        """current edge j ->ch, Evaluates {j} <- {ch},pa_j against {j} u pa_ch -> pa_j"""
        pa_j_cur = self.parents_of(j)
        pa_ch_cur = self.parents_of(ch)
        assert j in pa_ch_cur

        pa_j_up = (
            pa_j_cur.copy()
        )  # [p for p in self._nodes if (p in pa_j_cur or p == ch)]
        pa_j_up.append(ch)

        pa_ch_up = (
            pa_ch_cur.copy()
        )  # [p for p in self._nodes if (p in pa_ch_cur and p != j)]
        pa_ch_up.remove(j)

        score_cur = self.eval_edge(j, pa_j_cur) + self.eval_edge(ch, pa_ch_cur)
        score_up = self.eval_edge(j, pa_j_up) + self.eval_edge(ch, pa_ch_up)

        gain = score_cur - score_up
        return gain

    def eval_edge(self, j, pa) -> (int, int, int, list, list):
        """
        Evaluates a causal relationship pa(Xj)->Xj.

        :param j: Xj
        :param pa: pa(Xj)
        :return: score_up=score(Xpa->Xj)
        """
        hash_key = f"j_{str(j)}_pa_{str(pa)}"

        if self.mdl_cache.__contains__(hash_key):
            score_up = self.mdl_cache[hash_key]
            return score_up

        info = ""
        for i in pa:
            info += f"{self.is_true_edge(i)(j)} & "
        score_up = score_edge(
            self.data_C,
            self.gp_hyperparams,
            parents=pa,
            target=j,
            logger=self.logger,
            verbosity=self.verbosity - 1,
            edge_info=info,
        )

        self.mdl_cache[hash_key] = score_up
        return score_up

    def initial_edges(self, q: UPQ, skip_insignificant=False) -> UPQ:

        for j in self._nodes:
            pa = []
            score = self.eval_edge(j, pa)
            others = [i for i in self._nodes if not (i == j)]
            for i in others:
                score_ij = self.eval_edge(j, [i])

                edge_ij = q_entry(i, j, score_ij, score)
                # gain = score_ij - score
                # if smaller_preferred:
                # prio = -gain * 100 # negating gain for prioritization in q
                # else:
                #   prio = gain * 100

                gain = score - score_ij
                prio = gain * 100
                if (not skip_insignificant) or (not is_insignificant(gain)):
                    q.add_task(task=edge_ij, priority=prio)
                self.pair_edges[i][j] = edge_ij
        return q

    def eval_edges(self, j, new_parents):
        """
        Evaluate gain of replacing causal parents of Xj
        Args:
            j: target Xj
            new_parents: new parent set
        Returns: gain

        """
        old_score = self.eval_edge(j, self.parents_of(j))
        new_score = self.eval_edge(j, new_parents)
        gain = old_score - new_score
        return gain

    def update_edges(self, j, new_parents):
        """Update graph around Xj
        :param j: target Xj
        :param new_parents: new parent set
        :return:
        """
        old_parents = self.parents_of(j)
        for i in old_parents:
            self.remove_edge(i, j)

        for i in new_parents:
            gain, score, _, _, _ = self.eval_edge_addition(j, i)
            self.add_edge(i, j, score, gain)

    # Evaluate any DAG, looking up hashed scores for seen edges and computing them for new ones
    def eval_other_dag(self, adj, rev=False):
        mdl = 0
        for j in self._nodes:
            if rev:  # whether adj[j][i]==1 means i->j or vice versa
                pa = [i for i in self._nodes if adj[j][i] != 0]
            else:
                pa = [i for i in self._nodes if adj[i][j] != 0]
            score_j = self.eval_edge(j, pa)
            mdl = mdl + score_j
        return mdl

    def has_cycle(
        self, i, j
    ):  # from https://www.geeksforgeeks.org/detect-cycle-in-a-graph/
        visited = [False] * (self.N + 1)
        recStack = [False] * (self.N + 1)
        for node in range(self.N):
            if visited[node] == False:
                if self._has_cycle_util(node, visited, recStack, i, j) == True:
                    return True
        return False

    def _has_cycle_util(self, v, visited, recStack, i, j):
        visited[v] = True
        recStack[v] = True
        neighbors = []
        if v in self.node_ch:
            neighbors = self.node_ch[v]
        if v == i:
            neighbors = [n for n in range(self.N) if n in neighbors or n == j]

        for neighbour in neighbors:
            if visited[neighbour] == False:
                if self._has_cycle_util(neighbour, visited, recStack, i, j) == True:
                    return True
            elif recStack[neighbour] == True:
                return True

        recStack[v] = False
        return False


class q_entry:
    def __init__(self, i, j, score_ij, score_0):
        self.i = i
        self.j = j
        self.pa = i

        # Score of edge i->j in the empty graph
        self.score_ij = score_ij
        # Score of []->j
        self.score_0 = score_0

    def __hash__(self):
        return hash((self.i, self.j))

    def __eq__(self, other):
        return self.i == other.i & self.j == other.j

    def __str__(self):
        return f"j_{str(self.i)}_i_{str(self.j)}"
