from types import SimpleNamespace

import numpy as np

from exp.exptypes import DataSimulator
from exp.util.sample_cdnod import (
    sample_cdnod_sim,
    gen_cdnod_dag,
)
from exp.util.sample_ours import gen_context_data
from exp.util.sample_sergio import make_sergio


def gen_data(
    options,
    params,
    seedseq,
    seed,
):
    if options.data_simulator == DataSimulator.OURS:
        data_c, dag, gauss_dag, weights, is_true_edge = gen_context_data(
            options, params, seedseq.spawn(100)
        )
        data = SimpleNamespace(data_c=data_c, sim=DataSimulator.OURS)
        truths = SimpleNamespace(
            dags=weights, gauss_dag=gauss_dag, dag=dag, is_true_edge=is_true_edge
        )

    elif options.data_simulator == DataSimulator.CDNOD:
        sparsity = None
        # raise NotImplementedError
        domain_seed = int(1000 * np.random.uniform())
        truths = gen_cdnod_dag(
            params["N"], params["C"], sparsity, options.dag_edge_p, seed
        )
        data_c = [
            sample_cdnod_sim(
                dag=truths.gauss_dag,
                n_samples=params["S"],
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i,
                iv_hard=None,
                iv_scale=None,
                iv_coef=None,
            )
            for i, targets in enumerate(truths.targets)
        ]
        data = SimpleNamespace(data_c=data_c, sim=DataSimulator.CDNOD)

    elif options.data_simulator == DataSimulator.SERGIO:
        sparsity = 1
        options.dag_edge_p = 0.3
        # raise NotImplementedError
        truths = gen_cdnod_dag(
            params["N"], params["C"], sparsity, options.dag_edge_p, seed
        )
        tgt = make_sergio(
            dag=truths.dag,
            seed=seed,
            n_observations=params["S"],
            n_ho_observations=params["S"],
            n_interv_obs=params["S"],
            n_intervention_sets=params["C"] - 1,
        )
        data_c = [None for _ in range(params["C"])]
        data_c[0] = tgt.x
        for c_i in range(params["C"] - 1):
            data_c[c_i + 1] = tgt.x_interv_data[tgt.envs == c_i]

        data = SimpleNamespace(data_c=data_c, tgt=tgt, sim=DataSimulator.SERGIO)
    else:
        raise ValueError(f"Invalid data simulator {options.data_simulator}")

    truths.is_true_edge = (
        lambda i: lambda j: (
            "causal "
            if truths.dag[i][j] != 0
            else "anticausal" if truths.dag[j][i] != 0 else "spurious"
        ),
    )
    return data, truths
