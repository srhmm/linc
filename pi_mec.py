
import numpy as np

from pi_search_clustering import pi_search_clustering
from pi_search_mdl import pi_mdl, pi_mdl_conf, pi_mdl_best
from pi_tree import is_insignificant
from pi import PI
from pi_search_ilp import pi_search_ILP
from utils_pi import pi_convertfrom_pair, pi_enum
from vsn import Vsn


class PiMEC:
    def __init__(self, C_n, X_n, XYc, vsn):
        self.C_n = C_n
        self.X_n = X_n
        self.XYc = XYc
        self.vsn = vsn
        self.nodes = set([i for i in range(X_n)])

        self.score_cache = {}
        self.mdl_cache = {}
        self.pi_cache = {}

    def eval_adj(self, adj, gain=True):
        mdl = 0
        mdls = [0 for _ in self.nodes]
        for j in self.nodes:
            pa = [i for i in self.nodes if adj[j][i]==1] #adj[i][j]==1]
            score_j, mdl_j, _ = self.eval_edge(j, pa)
            if gain:
                mdl = mdl + score_j
                mdls[j] = score_j
            else:
                mdl = mdl + mdl_j
                mdls[j] = mdl_j
        return mdl, mdls

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

        score_up, mdl_up, pi_up = edge_score_target(self.XYc, covariates=pa, target=j, vsn=self.vsn, C_n=self.C_n)

        # Weird case of inf scores - use worst-case (no parent) score to avoid inf scores over the whole causal model
        if score_up == np.inf:
            hash_key = f'j_{str(j)}_pa_{str([])}'
            if self.score_cache.__contains__(hash_key):
                #print("(LINC) setting score to", self.score_cache[hash_key])
                return self.score_cache[hash_key], self.mdl_cache[hash_key],  self.pi_cache[hash_key]

            score_up, mdl_up, pi_up = edge_score_target(self.XYc, covariates=[], target=j, vsn=self.vsn, C_n=self.C_n)
            #print("(LINC) setting score to", round(score_up,2))

        self.score_cache[hash_key] = score_up
        self.mdl_cache[hash_key] = mdl_up
        self.pi_cache[hash_key] = pi_up
        return score_up, mdl_up, pi_up

    def conf_mat(self, adj, gain=True):
        sig_mat = np.zeros((len(adj), len(adj)))
        gain_mat = np.zeros((len(adj), len(adj)))
        for i in range(len(adj)):
            for j in range(len(adj)):
                if adj[i][j] == 1:
                    adj_rev = adj.copy()
                    adj_rev[i][j] = 0
                    adj_rev[j][i] = 1

                    mdl_causal, _ = self.eval_adj(adj, gain)
                    mdl_rev, _ = self.eval_adj(adj_rev, gain)
                    gain = mdl_rev - mdl_causal
                    gain_mat[i][j] = gain
                    if is_insignificant(gain, False, False, mdl_gain=False, alpha=0.05):
                        sig_mat[i][j] = 0
                    else:
                        sig_mat[i][j] = gain
        return gain_mat, sig_mat

# SCORES

def edge_score_target(XYc, covariates, target:int,
                     vsn: Vsn, C_n :int,
                      speedup = True)-> (int, int, list):

    if vsn.subsample_size is None: sub_size = min([len(XYc[c_i]) for c_i in range(C_n)])
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

    score, mdl_score, pi = edge_score(Xc, Yc, C_n, sub_size, vsn, (speedup and len(covariates) == 0))

    return score, mdl_score, pi

def edge_score(Xc, Yc, C_n, subsample_size, vsn, speedup_no_parents):
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
        if C_n == 1:
            score, _, _, _  = gpr.GP_c[0].mdl_score_ytrain()
            print("One context")
            return score, score, [[0]]

        if speedup_no_parents:
            pi = [[c_i for c_i in range(C_n)]]
            mdl_score, _, _, _, _ = pi_mdl(gpr, pi, vsn.regression_per_group, subsample_size)
        else:
            if vsn.ilp:
                if vsn.ilp_partitions:

                    gpr.cmp_distances()

                    if vsn.clustering:
                        pi, mdl_score, _, _, _  = pi_search_clustering(gpr.pair_mdl_gains,C_n, gpr, vsn.regression_per_group, subsample_size)
                    else:
                        pair_indicator_vars, _, pair_contexts = pi_search_ILP(gpr.pair_mdl_gains, C_n, wasserstein=False)
                        pi = pi_convertfrom_pair(pair_indicator_vars, pair_contexts, C_n)
                        mdl_score, _, _, _, _ = pi_mdl(gpr, pi, vsn.regression_per_group, subsample_size)
                else:
                    pi = [[c_i for c_i in range(C_n)]]
                    mdl_score, _, _, _, _ = pi_mdl(gpr, pi, vsn.regression_per_group, subsample_size)
            else: pi, mdl_score, _, _, _, _ = pi_mdl_best(gpr, vsn)

        if vsn.mdl_gain:
            gain_over_0, _, _ = pi_mdl_conf(gpr, pi, C_n, vsn.regression_per_group,
                                            subsample_size) #simply does MDL(model, pi0) - MDL(model, pi)
            score = -gain_over_0 #to make this consistent version above: smaller is better

        else:
            score = mdl_score

    return score, mdl_score, pi

