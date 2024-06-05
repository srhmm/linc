import itertools
import numpy as np
import causaldag as cd

from .util_dag import dag_n_edges


##For data generation: Convert true DAG and info about interventions or regimes to list of instantaneous and lagged causal links
def gen_regimes_from_dag(dag, self_weights, intervened_weights, N, random_state, tau_max, funs,
                         interventions, nodes_per_intervention, regimes, time_per_regime, nodes_per_regime):
    raise NotImplementedError


def gen_interventions_from_dag(dag, self_weights, intervened_weights, N, random_state, tau_max, funs,
                               interventions, nodes_per_intervention):
    """Generates links for multiple contexts with interventions (causal mechanism shifts)"""

    intervened_links = [None for _ in range(interventions)]
    intervened_nodes = [None for _ in range(interventions)]
    for case in range(interventions):
        targets = random_state.choice(range(N), nodes_per_intervention, replace=False)
        links = dict()
        cnode = N
        snode = cnode + 1
        min_lag = 0

        for i in dag.nodes:
            links[i] = []

            fun_pa = random_state.choice(funs, size=len(dag.parents_of(i)))

            # Add causal parents in dag
            for index_j, j in enumerate(dag.parents_of(i)):
                if min_lag == tau_max:
                    lag = tau_max
                else:
                    lag = int(-random_state.choice(range(min_lag, tau_max), 1)[0])
                # print('choose function:',fun_pa[index_j][0])

                w = dag.weight_mat[j][i]
                if i in targets:
                    w = intervened_weights.weight_mat[j][i]
                assert (w != 0)
                links[i].append(((j, lag), w, fun_pa[index_j])) # links[i].append(((j, lag), w, fun_pa[index_j][1]))

            # Add self links - lag -1?
            fun = random_state.choice(funs, size=1)[0][1]
            lag = -1
            w = self_weights.weight_mat[i][N + i]
            assert (w != 0)
            links[i].append(((i, lag), w, fun))

        links[cnode] = []
        links[snode] = []
        intervened_links[case] = links
        intervened_nodes[case] = targets
    return intervened_links, intervened_nodes

def gen_links_from_lagged_dag(dag, N, random_state, funs):
    """
    Converts DAG with lagged effects to link list (as used in tigramite data generation)

    :param dag: true DAG including self transitions
    :param N: n nodes
    :param random_state: rand
    :param funs: list of possible functional forms for each causal relationship
    """
    # TODO: each variable should have only one list (which is a problem in case of DAG ><><>< like (Jilles example))
    links = dict()
    cnode = dag.nodes[-2]
    snode = dag.nodes[-1]

    for i in range(N):
        links[i] = []
        fun_pa = random_state.choice(funs, size=len(dag.parents_of(i)))

        # Add causal parents in dag
        for index_j, j in enumerate(dag.parents_of(i)):
            if j != cnode and j != snode: # Skip context & spatial links
                lag = int(i//N-j//N)
                w = dag.weight_mat[j][i]
                assert (w != 0)
                j_t = j+lag*N
            else: # context & spatial links
                lag = 0
                w = dag.weight_mat[j][i]
                assert (w != 0)
                j_t = N+(j%N)-1
            links[i].append(((j_t, lag), w, fun_pa[index_j][1]))

    links[N] = [] # context node
    links[N+1] = [] # spatial node
    return links

def gen_links_from_dag(dag: cd.DAG,
                       self_weights,
                       N,
                       random_state,
                       tau_max,
                       funs):
    """ Converts DAG to link list (as used in tigramite data generation)

    :param dag: true DAG
    :param self_weights: artificial DAG with weights for self transitions
    :param N: n nodes
    :param random_state: rand
    :param tau_max: max transition lag
    :param funs: list of possible functional forms for each causal relationship

    Returns: list of links

    """
    """Generates links for an observational context"""
    links = dict()
    cnode = N
    snode = cnode + 1
    min_lag = 0

    for i in dag.nodes:
        links[i] = []
        fun_pa = random_state.choice(funs, size=len(dag.parents_of(i)))

        # Add causal parents in dag
        for index_j, j in enumerate(dag.parents_of(i)):
            if min_lag==tau_max:
                lag = tau_max
            else:
                lag = int(-random_state.choice(range(min_lag, tau_max), 1)[0])
            w = dag.weight_mat[j][i]
            assert (w != 0)
            links[i].append(((j, lag), w, fun_pa[index_j][1]))

        # Add self links - lag -1?
        fun = random_state.choice(funs, size=1)[0][1]
        lag = -1
        w = self_weights.weight_mat[i][N + i]
        assert (w != 0)
        links[i].append(((i, lag), w, fun))

        # Skip context links
        lag = -1
        w = self_weights.weight_mat[i][(2 * N) + 1]
        assert (w != 0)
        # links[i].append(((cnode, lag), w, fun))

        # Skip space links
        lag = 0
        w = 0.6  # self_weights.weight_mat[i][(2*N)+2]
        assert (w != 0)
        # links[i].append(((snode, lag), w, fun))

    links[cnode] = []
    links[snode] = []
    return links


##For scoring: Convert test DAG and max time lag tau_max to list of possible instantaneous and lagged causal links
def gen_all_links_from_adj(adj, tau_max, all_instantaneous):
    """Generates links for all combinations of time lags for each parent-effect relationship"""
    n_edges = dag_n_edges(adj)

    #todo combos should also be different for each parent, not only for each edge
    combos = itertools.combinations_with_replacement(range(tau_max + 1), n_edges)
    if all_instantaneous:
        combos = itertools.combinations_with_replacement(range(1), n_edges)
    all_links = []
    for combo in combos:

        links = dict()
        cnode = len(adj)
        snode = cnode + 1
        fun = lambda x: x
        n_edge = 0
        for i in range(len(adj)):
            links[i] = []

            # Add causal parents in dag
            for j in np.where(adj.T[i] != 0)[0]:
                lag = -combo[n_edge]
                n_edge = n_edge + 1
                links[i].append(((j, lag), 1, fun))

            # Add self links
            links[i].append(((i, -1), 1, fun))

        links[cnode] = []
        links[snode] = []
        all_links.append((links, combo))
    return all_links


def gen_all_target_links(covariates, i, tau_max):
    combos = itertools.combinations_with_replacement(range(tau_max + 1), len(covariates))
    """Generates the links for a single effect, for now, all instantaneous"""
    all_links = []

    for combo in combos:
        links = []
        # Add causal parents in dag
        for index_j, j in enumerate(covariates):
            lag = -combo[index_j]
            links.append(((j, lag), 1, None))
        # Add self links
        links.append(((i, -1), 1, None))
        all_links.append((links, combo))
    return all_links

def gen_lagged_target_links(covariates, i):
    """Generates the links for a single effect, with given time lags"""
    links = []
    # Add causal parents
    for j, lag in covariates:
        links.append(((j, -lag), 1, None))
    # Add self links
    links.append(((i, -1), 1, None))
    return links

def gen_instantaneous_target_links(covariates, i, fixed_lag = 0):
    """Generates the links for a single effect, here, all instantaneous"""
    links = []
    # Add causal parents
    for j in covariates:
        links.append(((j, fixed_lag), 1, None))
    # Add self links
    links.append(((i, -1), 1, None))
    return links


def gen_superset_target_links(covariates, i, max_lag):
    """Generates the links for a single effect, with all possible time lags up to tau_max"""
    links = []
    # Add causal parents
    for j in covariates:
        for lag in range(max_lag+1):
            links.append(((j, -lag), 1, None))
    # Add self link
    links.append(((i, -1), 1, None))
    return links


def gen_links_from_adj(adj, tau_max):
    links = dict()
    cnode = len(adj)
    snode = cnode + 1
    min_lag = 0
    fun = lambda x: x
    for i in range(len(adj)):
        links[i] = []

        # Add causal parents in dag
        for j in np.where(adj.T[i] != 0)[0]:  # dag.parents_of(i):
            lag = int(-np.random.choice(range(min_lag, tau_max), 1)[0])
            links[i].append(((j, lag), 1, fun))

        # Add self links
        links[i].append(((i, -1), 1, fun))
    links[cnode] = []
    links[snode] = []
    return links


def  gen_linkdag_from_links(links, N, tau_max):
    adj = np.zeros((N*tau_max, N))
    for target in links:
        for ((i, lag), w, _) in links[target]:
            index = N*-lag + i
            adj[index][target] = w
    return adj
def is_lagged_edge(i, lag_i, j, adj):
    N = len(adj[0])
    index = N * lag_i + i
    return adj[index][j]!=0
def info_edge(adj, max_lag) :
    def info_fun(parent, j):
        i, lag_i = parent
        N = len(adj[0])
        index = N * lag_i + i
        if adj[index][j]!=0:
            return "causal"
        for lag in range(max_lag):
            index = N * lag + i
            if adj[index][j]!=0:
                return "causal_wrong_lag"

        for lag in range(max_lag):
            index = N * lag + j
            if adj[index][i]!=0:
                return "anticausal"
        return "spurious"
    return info_fun