from copy import deepcopy

import numpy as np
import numpy as onp
#import jax.numpy as jnp
#from jax import random, vmap, jit
from collections import namedtuple
#from bacadi.bacadi.utils.graph import graph_to_mat #, adjmat_to_str, make_all_dags, mat_to_graph

#from bacadi.bacadi.graph.graph import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, UniformDAGDistributionRejection

from sergio.sergio_sampler import sergio_clean

Target = namedtuple('Target', (
    'passed_key',               # jax.random key passed _into_ the function generating this object
    'graph_model',
    'n_vars',
    'n_observations',
    'n_ho_observations',
    'g',                        # [n_vars, n_vars]
    'theta',                    # PyTree
    'x',                        # [n_observation, n_vars]    data
    'x_ho',                     # [n_ho_observation, n_vars] held-out data
    'x_interv',                 # list of (interv dict, interventional data)
    'x_interv_data',            # (n_obs stacked, n_vars) interventional data (same data as above)
    'interv_targets',           # (n_env, n_vars)
    'envs',                     # (n_obs) giving env that sample i came from
    'x_ho_interv',              # a tuple (interv dict, held-out interventional data)
    'x_ho_interv_data',         # (n_obs stacked, n_vars) heldout interventional data (same data as above)
    'envs_ho',                  # (n_obs) giving env that sample i came from
    'interv_targets_ho',        # (n_env, n_vars)
    'gt_posterior_obs',         # ground-truth posterior with observational data
                                # (log distribution tuple as e.g. given by `particle_marginal_empirical`), or None
    'gt_posterior_interv'       # ground-truth posterior with interventional data
                                # (log distribution tuple as e.g. given by `particle_marginal_empirical`), or None
))


def make_sergio( dag,
                seed=1,
                sergio_hill=2,
                sergio_decay=0.8,
                sergio_noise_params=1.0,
                sergio_cell_types=10,
                sergio_k_lower_lim=1,
                sergio_k_upper_lim=5,
                graph_prior_edges_per_node=2,
                n_observations=500,
                n_ho_observations=500,
                n_intervention_sets=5,
                n_interv_obs=500):
    #key = random.PRNGKey(seed)
    #graph_model = make_graph_model(
    #    n_vars=n_vars,
    #    graph_prior_str='sf',
    #    edges_per_node=graph_prior_edges_per_node)

    #key, subk = random.split(key)
    #subk = seed
    #g_gt = graph_model.sample_G(subk)
    #g_gt_mat = None #TODO np.array(graph_to_mat(g_gt))

    #g_gt = erdos_renyi_dag(n_vars, dag_dens, seed=seed)
    #g_gt_mat = g_gt
    g_gt = dag
    g_gt_mat = dag
    n_vars = len(dag)

    n_ko_genes = n_intervention_sets  # TODO
    rng = onp.random.default_rng(seed)
    passed_rng = deepcopy(rng)

    def k_param(rng, shape):
        return rng.uniform(
            low=sergio_k_lower_lim,  # default 1
            high=sergio_k_upper_lim,  # default 5
            size=shape)

    # MLE from e.coli
    def k_sign_p(rng, shape):
        return rng.beta(a=0.2588, b=0.2499, size=shape)

    def b(rng, shape):
        return rng.uniform(low=1, high=3, size=shape)

    n_obs = n_observations
    n_obs_ho = n_ho_observations
    spec = {}
    # number of intv
    spec['n_observations_int'] = 2 * n_intervention_sets * n_interv_obs
    # double to have heldout dataset
    spec['n_observations_obs'] = 2 * n_observations + n_ho_observations

    data = sergio_clean(
        spec=spec,
        rng=rng,  # function?
        g=g_gt_mat,
        effect_sgn=None,  # TODO
        toporder=None,
        n_vars=n_vars,
        b=b,  # function
        k_param=k_param,  # function
        k_sign_p=k_sign_p,  # function
        hill=sergio_hill,  # default 2
        decays=sergio_decay,  # default 0.8
        noise_params=sergio_noise_params,  # default 1.0
        cell_types=sergio_cell_types,  # default 10
        n_ko_genes=n_ko_genes)

    # extract data
    # obs
    x = np.array(data['x_obs'][:n_obs])
    x_ho = np.array(data['x_obs'][n_obs:(n_obs + n_obs_ho)])
    # interv
    x_interv_obs = np.array(data['x_obs'][-n_obs:])
    x_interv, x_ho_interv = np.split(data['x_int'], 2)
    envs, envs_ho = np.split(data['int_envs'], 2)
    x_interv_data = np.concatenate([x_interv_obs, x_interv])
    envs = np.concatenate([np.zeros(x_interv_obs.shape[0]),
                            envs]).astype(int)

    # ho interv
    x_ho_interv = np.array(x_ho_interv)
    envs_ho = np.array(envs_ho, dtype=int)
    interv_targets = data['interv_targets']

    # standardize
    mean = x_interv_data.mean(0)
    std = x_interv_data.std(0)

    x = (x - mean) / np.where(std == 0.0, 1.0, std)
    x_ho = (x_ho - mean) / np.where(std == 0.0, 1.0, std)
    x_interv_data = (x_interv_data - mean) / np.where(std == 0.0, 1.0, std)
    x_ho_interv = (x_ho_interv - mean) / np.where(std == 0.0, 1.0, std)

    target = Target(
        passed_key=passed_rng,
        graph_model=None, #graph_model,
        n_vars=n_vars,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations,
        g=g_gt_mat,
        theta=None,
        x=x,
        x_ho=x_ho,
        x_interv=[{} for _ in interv_targets],  # dummy
        x_interv_data=x_interv_data,
        interv_targets=interv_targets,
        envs=envs,
        x_ho_interv=[{} for _ in interv_targets],  # dummy
        x_ho_interv_data=x_ho_interv,
        interv_targets_ho=interv_targets,
        envs_ho=envs_ho,
        gt_posterior_obs=None,
        gt_posterior_interv=None)
    return target