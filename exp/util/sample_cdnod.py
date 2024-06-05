from types import SimpleNamespace

from gpcd.util.causallearn_utils import cpdag2dags
from gpcd.util.causallearn_utils import dag2cpdag


def gen_cdnod_dag(N, C, S, p, seed):
    true_dag = _sample_dag("er", N, p, seed=seed)
    true_cpdag = dag2cpdag(true_dag)

    mec_size = len(cpdag2dags(true_cpdag))
    total_edges = np.sum(true_dag)
    unoriented_edges = np.sum((true_cpdag + true_cpdag.T) == 2) // 2

    # Get interventions
    sampled_targets = _sample_interventions(N, C, S, seed=seed)
    is_true_edge = (
        lambda i: lambda j: (
            "causal "
            if true_dag[i][j] != 0
            else "anticausal" if true_dag[j][i] != 0 else "spurious"
        ),
    )
    truths = SimpleNamespace(
        gauss_dag=None,
        dag=true_dag,
        targets=sampled_targets,
        is_true_edge=is_true_edge,
    )
    return truths


def _graph2dag(graph):
    """Converts nx.Graph to an directed, acyclic form. Returns the adjancency matrix"""
    adj = nx.adjacency_matrix(graph).todense()
    adj = adj + adj.T
    adj = (adj != 0).astype(int)
    adj = np.tril(adj)

    assert nx.is_directed_acyclic_graph(
        nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
    )

    return adj


def erdos_renyi_dag(n, p, seed=None):
    """
    Simulates an Erdos Renyi random DAG on n vertices
    with expected degree p. Each node has the same expected
    degree.

    If p is an integer, it is the expected
    number of connected edges. Else, it is the expected degree
    fraction relative to n.
    """
    if p > 1 or isinstance(p, int):
        p = p / (n - 1)
    G = nx.erdos_renyi_graph(n, p, seed, directed=False)
    return _graph2dag(G)


def connected_erdos_renyi_dag(n, p, seed=None):
    """
    Simulates an Erdos Renyi random DAG on n vertices
    with expected degree p. Each node has the same expected
    degree and the graph is gauranteed connected, with
    a deterministic number of edges.

    If p is an integer, it is the expected
    number of connected edges. Else, it is the expected degree
    fraction relative to n.
    """
    if p <= 1 and isinstance(p, float):
        p = p * n
        if int(p) != p:
            import warnings

            warnings.warn(f"Number of neighbors {p:.1f} will be rounded")

    G = nx.connected_watts_strogatz_graph(n, k=round(p), p=1 - 1 / n, seed=seed)
    return _graph2dag(G)


def barabasi_albert_dag(n, p, seed=None):
    """
    Simulates an Barabasi Albert DAG on n vertices
    with expected degree p. The degree distribution follows
    a power law, and the graph is guaranteed to be connected.

    If p is an integer, it is the expected
    number of connected edges. Else, it is the expected degree
    fraction relative to n. Important, p must be <= 0.5
    or the integer equivalent to be guaranteed to succeed on all graphs.
    """
    if p > 1 or isinstance(p, int):
        p = p / (n - 1)

    # BA model input m leads to K=(1+...+m) + m*(n-m) total edges
    # p = K
    m = 0.5 * (2 * n - 1 - np.sqrt(4 * n**2 - 4 * n + 1 - 4 * p * n**2 + 4 * p * n))
    if int(m) != m:
        import warnings

        warnings.warn(f"Number of neighbors {m:.1f} will be rounded")

    G = nx.barabasi_albert_graph(n, round(m), seed)
    return _graph2dag(G)


def complete_dag(n, p=None, seed=None):
    """
    Returns a complete DAG over n variables
    """
    G = np.ones((n, n)) - np.eye(n)
    return np.tril(G)


def _sample_dag(dag_simulator, n_variables, dag_density, seed=None, inc=0):
    """
    Samples a DAG from a specified distribution
    """
    if dag_simulator == "er":
        dag = erdos_renyi_dag(n_variables, dag_density, seed=seed)
    elif dag_simulator == "ba":
        dag = barabasi_albert_dag(n_variables, dag_density, seed=seed)
    elif dag_simulator == "complete":
        dag = complete_dag(n_variables)
    else:
        raise ValueError(f"DAG simulator {dag_simulator} not valid optoion")

    count = 0
    if len(cpdag2dags(dag2cpdag(dag))) == 1:
        new_seed = int(10 * np.random.uniform()) + inc
        # Don't sample already solved MECs
        np.random.seed(new_seed)
        dag = _sample_dag(
            dag_simulator, n_variables, dag_density, new_seed, inc=inc + 1
        )  # inc makes sure the seeds don't cycle during recursion
        count += 1
        if count > 100:
            raise ValueError(
                f"Cannot sample a DAG in these settings with nontrivial MEC ({[dag_simulator, n_variables, dag_density]})"
            )

    return dag


def _sample_interventions(n_variables, n_total_environments, sparsity, seed=None):
    np.random.seed(seed)
    if isinstance(sparsity, float):
        sparsity = np.round(n_variables * sparsity).astype(int)
    sampled_targets = [
        np.random.choice(n_variables, sparsity, replace=False)
        for _ in range(n_total_environments)
    ]
    print(len(sampled_targets[0]))
    return sampled_targets


"""Simulated causal datasets"""

import numpy as np
import networkx as nx
from functools import partial


def sample_topological(n, equations, dag, noise, random_state=None):
    """
    Samples from a Structural Causal Model (SCM) in topological order

    Parameters
    ----------
    n : int
        Number of observations to sample

    equations : list of callables
        List of SCM equations, each a function accepting two parameters
        - All variables, of course only parents will be used.
        - An exogenous noise variable.

    noise : callable or list of callables
        Exogenous noise for each structural equation. If a single callable,
        then the same function will be used for all equations.

    random_state : int, optional
        Seed for reproducible randomness.

    Returns
    -------
    np.ndarray, shape (n, len(equations))
        Sampled observational data
    """
    np.random.seed(random_state)
    n_vars = len(equations)
    X = np.zeros((n_vars, n))

    topological_order = list(nx.topological_sort(nx.DiGraph(dag)))

    if not callable(noise):
        assert len(equations) == len(
            noise
        ), f"Must provide the same number of structural \
            equations as noise variables. Provided {len(equations)} and \
                {len(noise)}"

    for i in topological_order:
        f = equations[i]
        if not callable(noise):
            u = np.asarray([noise[i]() for _ in range(n)])
        else:
            u = np.asarray([noise() for _ in range(n)])
        X[i] = f(X, u)

    return X.T


def _icp_base_func(
    X,
    u,
    parents,
    function,
    f_join,
    intervened,
    intervention_func,
    pre_intervention,
):
    """Helper function for icp simulations"""
    # X shape (m_features, n_samples)
    X = X * parents[:, np.newaxis]
    X = X[parents != 0]
    X = function(X)
    if intervened:
        if pre_intervention:
            return f_join(intervention_func(X), axis=0) + u
        else:
            return intervention_func(f_join(X, axis=0) + u)
    else:
        return f_join(X, axis=0) + u


def sample_nonlinear_icp_sim(
    dag,
    n_samples,
    nonlinearity="id",
    noise_df=2,
    combination="additive",
    intervention_targets=None,
    intervention="soft",
    intervention_shift=0,
    intervention_scale=1,
    intervention_pct=None,
    random_state=None,
    pre_intervention=False,
    lambda_noise=None,
):
    """
    Simulates data from a given dag according to the simulation design
    in Heinz-Deml et al. 2018

    Parameters
    ----------
    dag : numpy.ndarray, shape (m, m)
        Weighted adjacency matrix.
        dag[i, j] != 0 if there is an edge from Xi -> Xj. The edge weight
        dag[i, j] will weight Xi in the computation of Xj,

    n_samples : int
        Number of training samples

    nonlinearity : {'id', 'relu', 'sqrt', 'sin', 'cubic'} or callable
        Nonlinear function of parent value.

    noise_df : int, nonnegative, default=100
        The degrees of freedom of the t-distribution from which
        the noise variable is sampled. Larger values are more
        similar to a Gaussian distribution.

    combination : {'additive', 'multiplicative'}
        How the functions of the variable's parents are combined.

    intervention_targets : list of features, optional
        Variables to intervene on.

    intervention : {'soft', 'hard'}
        Type of intervention. A 'soft' intervention adds noise.
        A 'hard' intervention is only the noise. The noise is a
        t-distribution with `noise_df` degrees of freedom, and
        shifted and scaled as specified.

    intervention_shift : float, default=0
        Shifted mean applied to noise data.

    intervention_scale : float, default=1
        Scale applied to noise data, pre shift

    intervention_pct : float or int, optional
        If `float`, the likelihood any given variable is intervened on.
        If `int`, the number of targets to intervene on.

    random_state : int, optional
        Allows reproducibility of randomness.

    Returns
    -------
    numpy.ndarray : shape (n_samples, dag.shape[0])
        Simulated data

    Notes
    -----
    Heinz-Deml et al. 2018 considers the following settings:

    n_samples : {100, 200, 500, 2000, 5000}
    noise_df : {2, 3, 5, 10, 20, 50, 100}
    intervention_shift : {0, 0.1, 0.2, 0.5, 1, 2, 5, 10}
    intervention_scale : {0, 0.1, 0.2, 0.5, 1, 2, 5, 10}

    """
    m = dag.shape[0]
    np.random.seed(random_state)

    if combination == "additive":
        f_join = np.sum
    elif combination == "multiplicative":
        f_join = np.prod

    # Choose nonlinearity
    if nonlinearity == "id":
        nonlinearity_func = lambda X: X
    elif nonlinearity == "relu":
        nonlinearity_func = lambda X: np.maximum(X, 0)
    elif nonlinearity == "sqrt":
        nonlinearity_func = lambda X: np.sin(X) * np.sqrt(np.abs(X))
    elif nonlinearity == "sin":
        nonlinearity_func = lambda X: np.sin(2 * np.pi * X)
    elif nonlinearity == "cubic":
        nonlinearity_func = lambda X: X**3
    elif callable(nonlinearity):
        nonlinearity_func = nonlinearity
    else:
        raise ValueError(f"Nonlinearity invalid: {nonlinearity}")

    # Choose intervention targets
    if intervention_targets is None:
        if isinstance(intervention_pct, float):
            intervention_targets = [
                i for i in range(m) if np.random.uniform() < intervention_pct
            ]
        elif isinstance(intervention_pct, int):
            intervention_targets = np.random.choice(
                m, size=(intervention_pct), replace=False
            )
        else:
            intervention_targets = []
    elif isinstance(intervention_targets, int):
        intervention_targets = [intervention_targets]

    # Intervention function
    if intervention == "hard":
        intervention_func = (
            lambda X: intervention_scale
            * (np.random.standard_t(df=noise_df, size=X.shape))
            + intervention_shift
        )
    elif intervention == "soft":
        intervention_func = (
            lambda X: X
            + intervention_scale * (np.random.standard_t(df=noise_df, size=X.shape))
            + intervention_shift
        )
    elif callable(intervention):
        intervention_func = intervention
    else:
        raise ValueError(f"Invalid intervention: {intervention}")

    if lambda_noise is None:
        noise = lambda: np.random.standard_t(df=noise_df)
    else:
        noise = lambda: lambda_noise(np.random.standard_t(df=noise_df))

    equations = [
        partial(
            _icp_base_func,
            parents=parents,
            function=nonlinearity_func,
            f_join=f_join,
            intervened=(i in intervention_targets),
            intervention_func=intervention_func,
            pre_intervention=pre_intervention,
        )
        for i, parents in enumerate(dag.T)
    ]

    X = sample_topological(n_samples, equations, dag, noise, random_state)

    return X


def _cdnod_base_func(
    X, u, parents, coefs, functions, noise_scale, noise_shift, additive
):
    """Helper function for icp simulations"""
    # X shape (m_features, n_samples)
    # X = X * parents[:, np.newaxis]
    n_samples = X.shape[1]
    X = X[parents != 0]
    # X shape (m_parents, n_samples)
    X = np.asarray([b * f(x) for b, f, x in zip(coefs, functions, X)])
    if additive:
        return np.sum(X, axis=0) + noise_scale * (u + noise_shift)
    else:
        if sum(parents) == 0:
            return np.random.normal(0, 1, (n_samples,))
        return np.sum(X, axis=0) * np.abs(noise_scale * (u + noise_shift))


def sample_cdnod_sim(
    dag,
    n_samples,
    functions=[
        np.tanh,
        np.sinc,
        lambda x: x**2,
        lambda x: x**3,
    ],
    intervention_targets=None,
    intervention_pct=None,
    base_random_state=None,
    domain_random_state=None,
    iv_hard=False,
    iv_scale=False,
    iv_coef=False,
    iv_fun=False,
    iv_shift=False,
    noise_uniform=True,
    base_coef=False,
):
    """
    Simulates data from a given dag according to the simulation design
    in Huang et al. 2020

    Parameters
    ----------
    dag : numpy.ndarray, shape (m, m)
        Weighted adjacency matrix.
        dag[i, j] != 0 if there is an edge from Xi -> Xj. The edge weight
        dag[i, j] will weight Xi in the computation of Xj,

    n_samples : int
        Number of training samples

    functions : list of callables
        Possible functions of parent variables to be sampled and summed
        when simulating the SCM.

    intervention_targets : list of features, optional
        Variables to intervene on.

    intervention_pct : float or int, optional
        If `float`, the likelihood any given variable is intervened on.
        If `int`, the number of targets to intervene on.

    base_random_state : int, optional
        Allows reproducibility of randomness of underlying SCM

    domain_random_state : int, optional
        Allows reproducibility of randomness of intervention

    Returns
    -------
    numpy.ndarray : shape (n_samples, dag.shape[0])
        Simulated data

    Notes
    -----

    """
    m = dag.shape[0]
    dag = (dag != 0).astype(int)
    base_seed = np.random.RandomState(base_random_state)
    domain_seed = np.random.RandomState(domain_random_state)

    # Choose intervention targets
    if intervention_targets is None:
        if isinstance(intervention_pct, float):
            intervention_targets = [
                i for i in range(m) if domain_seed.uniform() < intervention_pct
            ]
        elif isinstance(intervention_pct, int):
            intervention_targets = domain_seed.choice(
                m, size=(intervention_pct), replace=False
            )
        else:
            intervention_targets = []
    elif isinstance(intervention_targets, int):
        intervention_targets = [intervention_targets]

    domain_functions = [
        domain_seed.choice(functions, size=(np.sum(parents)))
        for i, parents in enumerate(dag.T)
    ]
    # If intervention on variable, use domain specific seed. Otherwise shared base seed
    functions = [
        base_seed.choice(functions, size=(np.sum(parents)))
        for i, parents in enumerate(dag.T)
    ]

    additives = [
        False
        # True #base_seed.choice([True, False])
        for i, parents in enumerate(dag.T)
    ]

    def _coefs(parents):
        if base_coef:
            return base_seed.uniform(0.5, 2.5, size=(np.sum(parents)))
        else:
            return [1] * sum(parents)

    base_equations = [
        partial(
            _cdnod_base_func,
            parents=parents,
            coefs=_coefs(parents),  # [1]*sum(parents),
            # coefs=base_seed.uniform(0.5, 2.5, size=(np.sum(parents))),
            functions=functions[i],
            noise_scale=1,
            noise_shift=0,
            additive=additives[i],
        )
        for i, parents in enumerate(dag.T)
    ]

    def _coefs_iv(parents):
        if iv_hard:
            return [0] * sum(parents)
        elif iv_coef:
            return domain_seed.uniform(0.5, 2.5, size=(np.sum(parents)))
        else:
            return [1] * sum(parents)

    def _noise_scaling():
        if iv_scale:
            return domain_seed.uniform(2, 5)
        else:
            return 1

    def _noise_shift():
        if iv_scale:
            return domain_seed.uniform(2, 10)
        else:
            return 0

    def _functions_iv(parents, i):
        if iv_fun:
            return domain_functions[i]
        else:
            return functions[i]

    domain_equations = [
        partial(
            _cdnod_base_func,
            parents=parents,
            # coefs=[1]*sum(parents),
            coefs=_coefs_iv(
                parents
            ),  # domain_seed.uniform(0.5, 2.5, size=(np.sum(parents))),
            functions=_functions_iv(parents, i),
            # functions=domain_seed.choice(functions, size=(np.sum(parents))),
            # noise_scale=domain_seed.uniform(1, 3),
            noise_scale=_noise_scaling(),
            noise_shift=_noise_shift(),
            additive=additives[i],
        )
        for i, parents in enumerate(dag.T)
    ]

    if noise_uniform:
        base_noises = [
            base_seed.choice(
                [
                    # lambda: np.random.normal(0, 1),
                    # lambda: np.random.uniform(-0.5, 0.5),
                    lambda: 1 / np.random.uniform(1, 3),
                    lambda: np.random.uniform(1, 3),
                ]
            )
            for i in range(m)
        ]

        domain_noises = [
            domain_seed.choice(
                [
                    # lambda: np.random.normal(0, 1),
                    # lambda: np.random.uniform(-0.5, 0.5),
                    lambda: 1 / np.random.uniform(1, 3),
                    lambda: np.random.uniform(1, 3),
                ]
            )
            for i in range(m)
        ]
    else:
        base_noises = [lambda: np.random.normal(0, 1) for i in range(m)]

        domain_noises = [lambda: np.random.normal(0, 1) for i in range(m)]

    equations = [
        domain if i in intervention_targets else base
        for i, (base, domain) in enumerate(zip(base_equations, domain_equations))
    ]

    noises = [
        domain if i in intervention_targets else base
        for i, (base, domain) in enumerate(zip(base_noises, domain_noises))
    ]

    X = sample_topological(n_samples, equations, dag, noises, domain_random_state)

    return X
