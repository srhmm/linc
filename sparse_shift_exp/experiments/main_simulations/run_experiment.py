import argparse
from pathlib import Path
import logging
import pickle
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from cdt.metrics import SID, SHD

from time import perf_counter

from experiments.main_simulations.make_gp import sample_data_gp, IvType
from sparse_shift.datasets import (
    sample_cdnod_sim,
    sample_topological,
    erdos_renyi_dag,
    connected_erdos_renyi_dag,
    barabasi_albert_dag,
    complete_dag, sample_nonlinear_icp_sim,
)
from sparse_shift.plotting import plot_dag
from sparse_shift.testing import test_mechanism_shifts, test_mechanism
from sparse_shift.methods import FullPC, PairwisePC, MinChangeOracle, MinChange, FullMinChanges
from sparse_shift.metrics import dag_true_orientations, dag_false_orientations, \
    dag_precision, dag_recall, average_precision_score
from exp_metrics import dag_fpr, dag_tpfpfnbi, \
    pr_scores_optimistic, pr_scores, soft_score_var
from sparse_shift.utils import dag2cpdag, cpdag2dags

from experiments.main_simulations.make_sergio import make_sergio
import os
import warnings

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


def _sample_dag(dag_simulator, n_variables, dag_density, seed=None, inc=0):
    """
    Samples a DAG from a specified distribution
    """
    if dag_simulator == "er":
        dag = erdos_renyi_dag(n_variables, dag_density, seed=seed)
    elif dag_simulator == "ba":
        dag = barabasi_albert_dag(n_variables, dag_density, seed=seed)
    elif dag_simulator == 'complete':
        dag = complete_dag(n_variables)
    else:
        raise ValueError(f"DAG simulator {dag_simulator} not valid optoion")

    count = 0
    if len(cpdag2dags(dag2cpdag(dag))) == 1:
        # Don't sample already solved MECs

        new_seed = seed - inc
        # Don't sample already solved MECs
        np.random.seed(new_seed)
        dag = _sample_dag(dag_simulator, n_variables, dag_density, new_seed, inc = inc+1) # inc makes sure the seeds don't cycle during recursion

        count += 1
        if count > 100:
            raise ValueError(f"Cannot sample a DAG in these settings with nontrivial MEC ({[dag_simulator, n_variables, dag_density]})")

    return dag


def _sample_interventions(n_variables, n_total_environments, sparsity, seed=None):
    np.random.seed(seed)
    if isinstance(sparsity, float):
        sparsity = np.round(n_variables * sparsity).astype(int)
    sampled_targets = [
        np.random.choice(n_variables, sparsity, replace=False)
        for _ in range(n_total_environments)
    ]
    return sampled_targets


def _sample_datasets(data_simulator, sample_size, dag, intervention_targets, n_envs, seed=None):
    """
    Samples multi-environment data from a specified distribution
    """

    np.random.seed(seed)
    domain_seed = int(1000 * np.random.uniform())
    if data_simulator == "cdnod":
        Xs = [
            sample_cdnod_sim(
                dag,
                sample_size,
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i
            )
            for i, targets in enumerate(intervention_targets)
        ]
    elif data_simulator == "hard-iv":
        Xs = [
            sample_cdnod_sim(
                dag,
                sample_size,
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i,
                iv_hard=True,
            ) for i, targets in enumerate(intervention_targets)
        ]
    elif data_simulator == "scale-iv":
        Xs = [
            sample_cdnod_sim(
                dag,
                sample_size,
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i,
                iv_scale=True,
            ) for i, targets in enumerate(intervention_targets)
        ]
    elif data_simulator == "cf-iv":
        Xs = [
            sample_cdnod_sim(
                dag,
                sample_size,
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i,
                iv_coef=True,
            ) for i, targets in enumerate(intervention_targets)
        ]
    elif data_simulator == "fun-iv":
        Xs = [
            sample_cdnod_sim(
                dag,
                sample_size,
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i,
                iv_fun=True,
            ) for i, targets in enumerate(intervention_targets)
        ]

    elif data_simulator == "shift-iv":
        Xs = [
            sample_cdnod_sim(
                dag,
                sample_size,
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i,
                iv_shift=True,
            ) for i, targets in enumerate(intervention_targets)
        ]
    elif data_simulator == "funcf-iv":
        Xs = [
            sample_cdnod_sim(
                dag,
                sample_size,
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i,
                iv_fun=True, iv_coef=True,
            ) for i, targets in enumerate(intervention_targets)
        ]
    elif data_simulator == "gp":
        '''
        Xs = [
            sample_cdnod_sim(
                dag,
                sample_size,
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i,
                iv_coef=False,
                functions=[lambda x: np.random.RandomState(seed).multivariate_normal(
                    mean=np.zeros(len(x)), cov=kernel_rbf(x,x))]
            ) for i, targets in enumerate(intervention_targets)
        ]
        '''
        Xs = sample_data_gp(dag, n_envs, sample_size, len(dag), IvType.PARAM_CHANGE, intervention_targets, seed)
    elif data_simulator == "ling":
        Xs = [
            sample_cdnod_sim(
                dag,
                sample_size,
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i,
                functions=[lambda x: x],
                noise_uniform=False, iv_coef=True #TODO!!!
            ) for i, targets in enumerate(intervention_targets)
        ]
    elif data_simulator == "sergio":
        true_dag = dag
        tgt = make_sergio(true_dag, seed=seed, n_observations=sample_size, n_ho_observations=sample_size,
                          n_interv_obs=sample_size,
                          n_intervention_sets=n_envs - 1)

        Xs = [None for _ in range(n_envs)]
        Xs[0] = tgt.x
        for c_i in range(n_envs - 1):
            Xs[c_i + 1] = tgt.x_interv_data[tgt.envs == c_i]

    else:
        raise ValueError(f"Data simulator {data_simulator} not valid optoion")

    return Xs


def main(args):
    # Determine experimental settings
    if args.quick:
        from exp_quick_settings import get_experiment_params, get_param_keys
    else:
        from exp_settings import get_experiment_params, get_param_keys

    # Initialize og details
    logging.basicConfig(
        filename="logging.log",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.INFO,
    )
    logging.info(f"NEW RUN:")
    logging.info(f"Args: {args}")
    logging.info(f"Experimental settings:")
    logging.info(get_experiment_params(args.experiment))

    # Create results csv header
    header = np.hstack(
        [
            ["params_index"],
            get_param_keys(args.experiment),
            ["Method", "Soft", "Number of environments", "Rep"],
            ["Number of possible DAGs", "MEC size", "MEC total edges", "MEC unoriented edges"],
            ["True orientation rate", "False orientation rate", "Precision", "Recall", 'Average precision',
             'AUROC_OPT', 'AUROC', 'SID', 'SHD', 'RT'],
        ]
    )
    # Create edge_decisions csv header
    header_edges = np.hstack(
        [
            ["params_index"],
            get_param_keys(args.experiment),
            ["Method", "Soft", "Number of environments", "Rep"],
            ["Correct", "Incorrect", "Bidirected",  "score"]
        ]
    )
    if not os.path.exists('results/'):
        os.makedirs('results/')

    from pathlib import Path
    #
    name = 'context'
    identifier = 2
    fnm = f"./results/{name}_{identifier}.csv"
    f = Path(fnm)
    while f.is_file():
        identifier = identifier + 1
        fnm = f"./results/{name}_{identifier}.csv"
        f = Path(fnm)

    write_file = open(f"./results/{name}_{identifier}.csv", "w+")
    write_file.write(", ".join(header) + "\n")
    write_file.flush()

    write_file_edges = open(f"./results/edge_{args.experiment}_{identifier}.csv", "w+")
    write_file_edges.write(", ".join(header_edges) + "\n")
    write_file_edges.flush()


    # Construct parameter grids
    param_dicts = get_experiment_params(args.experiment)
    prior_indices = 0
    logging.info(f'{len(param_dicts)} total parameter dictionaries')
    for params_dict in param_dicts:
        param_keys, param_values = zip(*params_dict.items())
        params_grid = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]

        # Iterate over
        logging.info(f'{len(params_grid)} total parameter combinations')

        for i, params in enumerate(params_grid):
            logging.info(f"Params {i} / {len(params_grid)}")
            run_experimental_setting(
                args=args,
                params_index=i + prior_indices,
                write_file=write_file,
                write_file_edges=write_file_edges,
                **params,
            )
        
        prior_indices += len(params_grid)
    logging.info(f'Complete')


def run_experimental_setting(
    args,
    params_index,
    write_file, write_file_edges, experiment,
    n_variables,
    n_total_environments,
    sparsity,
    intervention_targets,
    sample_size,
    dag_density,
    reps,
    data_simulator,
    dag_simulator,
):

    if data_simulator == "sergio":
        assert n_variables == -1 # s.t. experiment is not repeated unnecessarily if a list of values for n_variables was given in settings
        n_variables = n_total_environments - 1

    assert n_variables > 1

    skip_oracle=True
    # Determine experimental settings
    if args.quick:
        from exp_quick_settings import get_experiment_methods
    else:
        from exp_settings import get_experiment_methods

    name = args.experiment


    if sparsity is not None and sparsity > n_variables:
        logging.info(f"Skipping: sparsity {sparsity} greater than n_variables {n_variables}")
        return

    experimental_params = [
        params_index,
        n_variables,
        n_total_environments,
        sparsity,
        intervention_targets,
        sample_size,
        dag_density,
        reps,
        data_simulator,
        dag_simulator,
    ]
    experimental_params = [str(val).replace(", ", ";") for val in experimental_params]

    def _run_rep(rep, write):
        results = []
        rshift = 900
        # Get DAG
        true_dag = _sample_dag(dag_simulator, n_variables, dag_density, seed=rep + rshift)
        true_cpdag = dag2cpdag(true_dag)
        mec_size = len(cpdag2dags(true_cpdag))
        total_edges = np.sum(true_dag)
        unoriented_edges = np.sum((true_cpdag + true_cpdag.T) == 2) // 2

        # Get interventions
        if intervention_targets is None:
            sampled_targets = _sample_interventions(
                n_variables, n_total_environments, sparsity, seed=rep
            )
        else:
            sampled_targets = intervention_targets

        # Compute oracle results
        if not skip_oracle:
            fpc_oracle = FullPC(true_dag)
            mch_oracle = MinChangeOracle(true_dag)

            for n_env, intv_targets in enumerate(sampled_targets):
                n_env += 1
                fpc_oracle.add_environment(intv_targets)
                mch_oracle.add_environment(intv_targets)

                cpdag = fpc_oracle.get_mec_cpdag()

                true_orients = np.round(dag_true_orientations(true_dag, cpdag), 4)
                false_orients = np.round(dag_false_orientations(true_dag, cpdag), 4)
                precision = np.round(dag_precision(true_dag, cpdag), 4)
                recall = np.round(dag_recall(true_dag, cpdag), 4)
                ap = recall

                result = ", ".join(
                    map(
                        str,
                        experimental_params + [
                            "Full PC (oracle)",
                            False,
                            n_env,
                            rep,
                            len(fpc_oracle.get_mec_dags()),
                            mec_size,
                            total_edges,
                            unoriented_edges,
                            true_orients,
                            false_orients,
                            precision,
                            recall,
                            ap,
                        ],
                    )
                ) + "\n"
                if write:
                    write_file.write(result)
                    write_file.flush()
                else:
                    results.append(result)

                cpdag = mch_oracle.get_min_cpdag()

                true_orients = np.round(dag_true_orientations(true_dag, cpdag), 4)
                false_orients = np.round(dag_false_orientations(true_dag, cpdag), 4)
                precision = np.round(dag_precision(true_dag, cpdag), 4)
                recall = np.round(dag_recall(true_dag, cpdag), 4)
                ap = recall

                result = ", ".join(
                    map(
                        str,
                        experimental_params + [
                            "Min changes (oracle)",
                            False,
                            n_env,
                            rep,
                            len(mch_oracle.get_min_dags()),
                            mec_size,
                            total_edges,
                            unoriented_edges,
                            true_orients,
                            false_orients,
                            precision,
                            recall,
                            ap,
                        ],
                    )
                ) + "\n"
                if write:
                    write_file.write(result)
                    write_file.flush()
                else:
                    results.append(result)

            del fpc_oracle, mch_oracle

        # Sample dataset
        if data_simulator is None:
            return results

        Xs = _sample_datasets(
            data_simulator, sample_size, true_dag, sampled_targets, seed=rep+rshift, n_envs=n_total_environments
        )
	
        print("*** Rep:", rep+rshift, "#Contexts:", n_total_environments,  "#Vars:",  n_variables,
              "#Samples:", sample_size,  "Sparsity:", sparsity , "Density:",dag_density,  "Sim:", data_simulator, "Print: ",write_file, "***" )

       # if not os.path.exists(f'./data/{rep}/{name}/'):
       #     os.makedirs(f'./data/{rep}/{name}/')


        # Compute empirical results
        for save_name, method_name, mch, hyperparams in get_experiment_methods(
            args.experiment
        ):
            FAST=False
            #if n_total_environments > 25:
            #    FAST = True
            time_init = perf_counter()
            mch = mch(cpdag=true_cpdag, **hyperparams)
            time_allenv = [perf_counter()- time_init, perf_counter()- time_init]  # overall runtime for all environments, for soft in True, False

            for n_env, X in enumerate(Xs):
                n_env += 1
                time_st = perf_counter()
                mch.add_environment(X)


                if not FAST and n_env < n_total_environments:
                    continue

                time_allenv = [time_allenv[0] + perf_counter() - time_st, time_allenv[1] + perf_counter() - time_init]

                for soft in [True, False]:

                    time_st = perf_counter()
                    min_cpdag = mch.get_min_cpdag(soft)

                    runtime_ctx = round(perf_counter() -time_st, 5)
                    time_allenv[0 if soft else 1] = time_allenv[0 if soft else 1] + runtime_ctx
                    runtime_all = time_allenv[0 if soft else 1]


                    true_orients = np.round(dag_true_orientations(true_dag, min_cpdag), 4)
                    false_orients = np.round(dag_false_orientations(true_dag, min_cpdag), 4)
                    precision = np.round(dag_precision(true_dag, min_cpdag), 4)
                    recall = np.round(dag_recall(true_dag, min_cpdag), 4)
                    fpr = np.round(dag_fpr(true_dag, min_cpdag), 4)
                    tp, fp, fn, bi = np.round(dag_tpfpfnbi(true_dag, min_cpdag), 4)
                    sid = SID(true_dag, min_cpdag)
                    shd = SHD(true_dag, min_cpdag)

                    # Considers best case AP, AUC, AUROC (do not say very much, but we can use them if there is no conf./score associated to edges)
                    ap, _, auroc_opt = pr_scores_optimistic(true_dag, min_cpdag)

                    if hasattr(mch, 'pvalues_'):
                        # Considers mechanisms by pvalue
                        soft_scores = soft_score_var(mch.soft_scores_, mch.pvalues_)
                        ap, _, auroc = pr_scores(true_dag, min_cpdag, soft_scores)
                    else:
                        soft_scores= []
                        auroc = auroc_opt



                    result = ", ".join(
                        map(
                            str,
                            experimental_params + [
                                method_name,
                                soft,
                                n_env,
                                rep,
                                len(mch.get_min_dags(soft)),
                                mec_size,
                                total_edges,
                                unoriented_edges,
                                true_orients,
                                false_orients,
                                precision,
                                recall,
                                ap,
                                auroc_opt,
                                auroc,
                                sid,
                                shd,
                                runtime_all
                            ],
                        )
                    ) + "\n"
                    if write:
                        write_file.write(result)
                        write_file.flush()
                    else:
                        results.append(result)


                    # Edge decisions
                    for i in range(len(true_dag)):
                        for j in range(len(true_dag)):
                            tp, fp, bi = 0, 0, 0

                            if (len(soft_scores) == len(min_cpdag)):  #
                                score = soft_scores[j]
                            else:
                                score = None
                            if min_cpdag[i][j] == 1:
                                if true_dag[i][j] == 1:
                                    if min_cpdag[j][i] == 1:
                                        bi = 1
                                    else:
                                        tp = 1
                                else:
                                    fp = 1
                                edge_decision = ", ".join(
                                    map(
                                        str,
                                        experimental_params + [
                                            method_name,
                                            soft,
                                            n_env,
                                            rep,
                                            tp, fp, bi, score
                                        ],
                                    )
                                ) + "\n"
                                if write:
                                    write_file_edges.write(edge_decision)
                                    write_file_edges.flush()

                # Save pvalues
                if not os.path.exists(f'./results/pvalue_mats/{name}/'):
                    os.makedirs(f'./results/pvalue_mats/{name}/')
                if hasattr(mch, 'pvalues_'):
                    np.save(
                        f"./results/pvalue_mats/{name}/{name}_{save_name}_pvalues_params={params_index}_rep={rep}.npy",
                        mch.pvalues_,
                    )

                print(str(n_env)+"/"+str(n_total_environments), "-Time:", np.round(runtime_all), "\n    -e(tp,fp,fn,b)", str(total_edges)+"("+str(tp)+","+str(fp)+","+str(fn)+","+str(bi)+")" , "-p/r/fpr:", str(precision)+"/"+ str(recall)+"/"+str(fpr), "\n    -SID:", sid)
                if auroc is not None:
                    print("\t-AP:", ap,  "-AUC_opt:", auroc_opt,  "-AUC:", auroc)
        return results

    rep_shift = 0
    if args.jobs is not None:
        results = Parallel(
                n_jobs=args.jobs,
            )(
                delayed(_run_rep)(rep + rep_shift, False) for rep in range(reps)
            )
        for result in np.concatenate(results):
            write_file.write(result)
        write_file.flush()
    else:
        for rep in tqdm(range(reps)):
            _run_rep(rep + rep_shift, write=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        help="experiment parameters to run",
    )
    parser.add_argument(
        "--jobs",
        help="Number of jobs to run in parallel",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--quick",
        help="Enable to run a smaller, test version",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    main(args)
