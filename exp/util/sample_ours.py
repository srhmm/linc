from itertools import product

from exp.exptypes import NoiseType, FunType, sample_random_multivariate_taylor_series


def gen_context_data(options, params, seeds, _depth=99, given_dag=None):
    """Generates continuous data in multiple contexts"""

    # Hyperparams
    random_state = np.random.default_rng(seeds[_depth])
    nb_edges = random_state.integers(1, params["N"])
    params["R"] = 1
    params["D"] = 1

    # Generate a dag with different edge weights in each context
    weights = _gen_context_dag(options, params, nb_edges, params["I"], random_state)
    if given_dag is not None:
        weights = dict.fromkeys(
            set(product(set(range(params["R"])), set(range(params["C"]))))
        )
        weights = {k: given_dag for k in weights}
    # Generate D datasets from C contexts
    data_D = dict()
    data_C_D = dict.fromkeys(
        set(product(set(range(params["C"])), set(range(params["D"]))))
    )

    for cpt, (c, d) in enumerate(product(range(params["C"]), range(params["D"]))):
        dag = weights[(0, c)]  # regime index ignored
        X = _gen_data_in_context(
            dag,
            params["S"],
            random_state,
            options.noise_dist,
            params["F"],
        )

        assert X.shape == (params["S"], params["N"])
        data_C_D[(c, d)] = X
        data_D[cpt] = X

    # if invalid_data and _depth > 0:
    #    return gen_context_data(options, params, seeds, _depth - 1)
    invalid_data = False

    # data_summary = SimpleNamespace(datasets=data_D, data_C_D=data_C_D)

    gauss_dag = weights[(0, 0)]
    dag = gauss_dag.weight_mat
    is_true_edge = (
        lambda i: lambda j: (
            "causal "
            if dag[i][j] != 0
            else "anticausal" if dag[j][i] != 0 else "spurious"
        ),
    )

    return data_D, dag, gauss_dag, weights, is_true_edge


def _gen_data_in_context(
    dag,
    n_samples,
    random_state,
    noise_dist: NoiseType,
    function_type: FunType,
    noise_std=0.2,
):
    """Generate data from a given graph and edge weights."""
    noise_fun = lambda sz, std: (
        (random_state.normal(size=sz, scale=std))
        if noise_dist == NoiseType.GAUSS
        else (random_state.uniform(size=sz, scale=std) * 2 - 1)
    )
    fun_type, fun = function_type

    order = dag.topological_sort()
    n_nodes = len(order)
    X = np.zeros((n_samples, n_nodes))
    for node in order:
        parents = list(dag.parents_of(node))
        if len(parents) == 0:
            X[:, node] = noise_fun(n_samples, noise_std)
            continue
        elif fun_type == FunType.TAYLOR:
            center = np.zeros(len(parents))
            radius = 1
            # todo rstate
            f = sample_random_multivariate_taylor_series(
                10, len(parents), center, radius
            )
            # make list of parent vectors
            in_data = [X[:, parents[i]] for i in range(len(parents))]
            intermediate_pred = f(*in_data)
            # standardize
            intermediate_pred = (
                intermediate_pred - np.mean(intermediate_pred)
            ) / np.std(intermediate_pred)

            X[:, node] = intermediate_pred + noise_fun(n_samples, noise_std)
            # normalize to -1 and 1
            X[:, node] = (X[:, node] - np.min(X[:, node])) / (
                np.max(X[:, node]) - np.min(X[:, node])
            ) * 2 - 1

        elif fun_type == FunType.LIN:
            coefficients = dag.weight_mat[:, node]
            intermediate_pred = np.dot(X, coefficients)
            intermediate_pred = (
                intermediate_pred - np.mean(intermediate_pred)
            ) / np.std(intermediate_pred)

            X[:, node] = intermediate_pred + noise_fun(n_samples, noise_std)
            X[:, node] = (X[:, node] - np.min(X[:, node])) / (
                np.max(X[:, node]) - np.min(X[:, node])
            ) * 2 - 1
        else:
            in_data = [X[:, parents[i]] for i in range(len(parents))]
            intermediate_pred = fun(*in_data)
            # standardize
            intermediate_pred = (
                intermediate_pred - np.mean(intermediate_pred)
            ) / np.std(intermediate_pred)

            X[:, node] = intermediate_pred + noise_fun(n_samples, noise_std)
            # normalize to -1 and 1
            X[:, node] = (X[:, node] - np.min(X[:, node])) / (
                np.max(X[:, node]) - np.min(X[:, node])
            ) * 2 - 1
    return X


def _gen_context_dag(options, params, nb_edges, intervention_nb, random_state):
    params["R"] = 1
    ## Define DAG structure
    import causaldag as cd
    from itertools import combinations
    from graphical_models.rand import unif_away_zero

    arcs = cd.rand.directed_erdos((params["N"]), 0)

    pairs = random_state.choice(
        list(combinations(range(params["N"]), 2)), nb_edges, replace=False
    )
    pairs = [(j, i) if random_state.random() > 0.5 else (i, j) for (i, j) in pairs]
    # lags = random_state.choice(range(options.true_tau_max + 1), nb_edges)
    for i in range(nb_edges):
        arcs = add_edge(pairs[i][0], pairs[i][1], 0, params["N"], arcs)

    ## For each context define intervention targets
    intervention_targets = dict.fromkeys(set(range(params["C"])))
    for c in intervention_targets.keys():
        # interventions shouldn't be on edges from spatial or context variable
        intervention_targets[c] = random_state.choice(
            list(arcs.arcs), size=intervention_nb, replace=False
        )  # TODO: assert different from a context to another?
        intervention_targets[c] = list(tuple(l) for l in intervention_targets[c])

    # todo not needed:
    ## For each regime define intervention targets
    intervention_targets_regimes = dict.fromkeys(set(range(params["R"])))
    for r in intervention_targets_regimes.keys():
        nb = params["I"]  # random_state.integers(1, params["N"] + 1)
        # interventions shouldn't be on edges from spatial or context variable
        intervention_targets_regimes[r] = random_state.choice(
            list(arcs.arcs), size=nb, replace=False
        )  # TODO: assert different from a context to another?
        intervention_targets_regimes[r] = list(
            tuple(l) for l in intervention_targets_regimes[r]
        )

    ## For each regime, define general weights and special weights for intervention tagets in contexts
    weights = dict.fromkeys(
        set(product(set(range(params["R"])), set(range(params["C"]))))
    )  # key = (regime, context)

    initial_weights = cd.rand.rand_weights(arcs)

    c_weights = {
        c: {t: unif_away_zero()[0] for t in intervention_targets[c]}
        for c in intervention_targets.keys()
    }
    for r in range(params["R"]):
        r_weights = dict()
        for t in intervention_targets_regimes[r]:
            w = unif_away_zero()[0]
            while abs(w - initial_weights.arc_weights[t]) < 0.1:
                w = unif_away_zero()[0]
            r_weights[t] = w
        # r_weights = {t: unif_away_zero()[0] for t in intervention_targets_regimes[r]}
        for c in range(params["C"]):
            weights[(r, c)] = cd.rand.rand_weights(arcs)
            for arc in initial_weights.arcs:
                if (
                    arc in intervention_targets_regimes[r]
                    and arc in intervention_targets[c]
                ):
                    w = unif_away_zero()[0]
                    while (
                        abs(w - initial_weights.arc_weights[arc]) < 0.1
                        or abs(w - r_weights[arc]) < 0.1
                    ):
                        w = unif_away_zero()[0]
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], w)
                elif (
                    arc not in intervention_targets_regimes[r]
                    and arc in intervention_targets[c]
                ):
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], c_weights[c][arc])
                elif (
                    arc not in intervention_targets_regimes[r]
                    and arc not in intervention_targets[c]
                ):
                    weights[(r, c)].set_arc_weight(
                        arc[0], arc[1], initial_weights.arc_weights[arc]
                    )
                elif (
                    arc in intervention_targets_regimes[r]
                    and arc not in intervention_targets[c]
                ):
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], r_weights[arc])
                else:
                    raise ValueError("wrong case")

    return weights


import numpy as np


def add_edge(node_i_t, node_j_t, lag, N, arcs):
    """
    Add arc to the DAG from a parent to the child at time t

    :param node_i_t: parent node name at time t
    :param node_j_t: child node name at time t
    :param lag: lag (negative value)
    :param N: number of variables
    :param arcs: DAG
    :return: DAG
    """
    arcs.add_arc(node_i_t + (-lag * N), node_j_t)
    return arcs
