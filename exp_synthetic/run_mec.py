# This script was adapted from sparse_shift (see LICENSE)

import argparse
import logging
import itertools
from time import perf_counter

import numpy as np
from cdt.metrics import SID, SHD
from pulp import PulpSolverError
from tqdm import tqdm
from joblib import Parallel, delayed

from exp_synthetic.methods_linc import pr_scores_, dag_fpr, dag_tpfpfnbi, pr_scores_optimistic
from gen_context_data import sample_data_gp
from intervention_types import IvType
from util_make_sergio import make_sergio
from sparse_shift import (
    sample_cdnod_sim,
    erdos_renyi_dag,
    barabasi_albert_dag,
    complete_dag,
)
from sparse_shift import dag_true_orientations, dag_false_orientations, \
    dag_precision, dag_recall
from sparse_shift import dag2cpdag, cpdag2dags

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
        new_seed = int(10*np.random.uniform()) + inc
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
    print(len(sampled_targets[0]))
    return sampled_targets


def _sample_datasets(data_simulator, sample_size, dag, intervention_targets, n_envs,
                     seed):
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
        ]'''
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
        tgt = make_sergio(true_dag, seed=seed, n_observations=sample_size, n_ho_observations=sample_size, n_interv_obs=sample_size,
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
    from settings import get_experiment_params, get_param_keys

    # Initialize og details
    logging.basicConfig(
        filename="../logging.log",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.INFO,
    )
    logging.info(f"NEW RUN:")
    logging.info(f"Args: {args}")
    logging.info(f"Experimental settings:")
    logging.info(get_experiment_params(args.experiment))

    # Create results csv header
    header_mec = np.hstack(
        [
            ["params_index"],
            get_param_keys(args.experiment),
            ["Method", "Soft", "Number of environments", "Rep"],
            ["Number of possible DAGs", "MEC size", "MEC total edges", "MEC unoriented edges"],
            ["True orientation rate", "False orientation rate", "Precision", "Recall", 'Average precision', 'AUROC_OPT', 'AUROC',  'SID', 'SHD', 'RT'],
        ]
    )

    # Create edge_decisions csv header
    header_edges = np.hstack(
        [
            ["params_index"],
            get_param_keys(args.experiment),
            ["Method", "Soft", "Number of environments", "Rep"],
            ["Correct", "Incorrect", "Bidirected",  "score_ij",  "score_ji", "gain_ij", "gain_ji"]
        ]
    )
    header_dag = np.hstack(
        [
            ["params_index"],
            get_param_keys(args.experiment),
            ["Method", "Soft", "Number of environments", "Rep"],
            ["TP","TN","FP","FN","shd","sid","runtime"]
        ]
    )

    # Additional identifier for this run (for debugging)

    if not os.path.exists('../results/'):
        os.makedirs('../results/')

    from pathlib import Path
    identifier= 1
    fnm = f"./results/{args.experiment}_{identifier}.csv"
    f = Path(fnm)
    id2 = ""
    while f.is_file():
        identifier= identifier + 1
        fnm = f"./results/{args.experiment}_{identifier}{id2}.csv"
        f = Path(fnm)
    # Results: Decisions for orienting edges in a MEC
    write_file = open(f"./results/{args.experiment}_{identifier}{id2}.csv", "w+")
    write_file.write(", ".join(header_mec) + "\n")
    write_file.flush()

    # Results: DAG search with unknown mec
    #write_file_dagsearch = open(f"../results/dag{identifier}{id2}.csv", "w+")
    #write_file_dagsearch .write(", ".join(header_dag) + "\n")
    #write_file_dagsearch .flush()

    # Debug: Decisions for each edge (only for LINC)
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
                write_file_dagsearch=None, #write_file_dagsearch,
                **params,
            )
        
        prior_indices += len(params_grid)
    logging.info(f'Complete')


def run_experimental_setting(
    args,
    params_index,
    write_file, write_file_edges, write_file_dagsearch, experiment,
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
        if n_total_environments<3:
            n_variables = 2

    assert n_variables > 1
    # Determine experimental settings
    from exp_synthetic.settings import get_experiment_methods

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
        rshift=200#50
        # Get DAG
        true_dag = _sample_dag(dag_simulator, n_variables, dag_density, seed=rep+rshift)
        true_cpdag = dag2cpdag(true_dag)

        mec_size = len(cpdag2dags(true_cpdag))
        total_edges = np.sum(true_dag)
        unoriented_edges = np.sum((true_cpdag + true_cpdag.T) == 2) // 2

        # Get interventions
        if intervention_targets is None:
            sampled_targets = _sample_interventions(
                n_variables, n_total_environments, sparsity, seed=rep+rshift
            )
        else:
            sampled_targets = intervention_targets

        # Skipping oracle experiments

        # Sample dataset
        if data_simulator is None:
            return results

        Xs = _sample_datasets(
            data_simulator, sample_size, true_dag, sampled_targets, n_envs = n_total_environments, seed=rep+rshift
        )
        # Note: if data_simulator=="sergio", interv_targets is ignored and we use diagonal exp design


        print("*** Rep: ", rep+rshift,"# Contexts: ", n_total_environments,  "# Vars:",  n_variables,
              "# Samples:", sample_size,  "Sparsity:", sparsity ,  "Density:",dag_density, "Sim:", data_simulator, "Print: ",write_file, "***" )
        #open('competitors_data.txt', 'w').close()
        # Compute empirical results
        for save_name, method_name, mch, hyperparams in get_experiment_methods(
            args.experiment
        ):
            time_st = perf_counter()

            ##np.save('competitors_data/DAG1.npy',true_cpdag)

            mch = mch(cpdag=true_cpdag, dag=true_dag,
                      **hyperparams)

            max_env = len(Xs)

            for n_env, X in enumerate(Xs):
                n_env += 1
                mch.add_environment(X)
                soft_todo = [True, False]

                #for LINC, only consider the full data with all environments, and no soft/hard score distinction
                if hasattr(mch, 'maxenv_only'):
                    if mch.maxenv_only and (n_env < max_env):
                        continue
                    soft_todo = [True]
                #for others, add one environment at a time to the data and discover a DAG over it (following the original implementation)
                for soft in soft_todo:
                    time_ctx = perf_counter()
                    # Discover best DAG in the MEC
                    min_cpdag = mch.get_min_dags(soft)


                    runtime_all = round(perf_counter() -time_st, 5)
                    runtime_ctx = round(perf_counter() -time_ctx, 5)

                    true_orients = np.round(dag_true_orientations(true_dag, min_cpdag), 4)
                    false_orients = np.round(dag_false_orientations(true_dag, min_cpdag), 4)
                    precision = np.round(dag_precision(true_dag, min_cpdag), 4)
                    recall = np.round(dag_recall(true_dag, min_cpdag), 4)

                    fpr = np.round(dag_fpr(true_dag, min_cpdag), 4)
                    tp, fp, fn, bi = np.round(dag_tpfpfnbi(true_dag, min_cpdag), 4)

                    sid = SID(true_dag, min_cpdag)
                    shd = SHD(true_dag, min_cpdag)

                    if hasattr(mch,'min_gains_'):
                        # Per Edge, MDL Gains
                        #edge_ap, edge_roc, edge_auc = pr_scores(true_dag, mch.min_gains_)
                        # Per Variable/Mechanism, MDL Scores
                        ap, _, auroc = pr_scores_(true_dag, min_cpdag, mch.min_gains_)#TODO mch.min_mdl_node_)
                        _, _, auroc2 = pr_scores_(true_dag, min_cpdag, mch.min_mdl_node_)
                        _, _, auroc_opt = pr_scores_optimistic(true_dag, min_cpdag)
                        ap, auroc_opt, auroc  = np.round(ap, 4), np.round(auroc_opt, 4),np.round(auroc, 4)

                    else:
                        ap, auroc_opt, auroc, auroc2  = None, None, None, None

                    TP, FP, FN, TN = 0,0,0,0
                    #Print each decision, i.e. for the DAG estimated for all (!) environments, evaluate each edge
                    if n_env == max_env and hasattr(mch, 'min_gains_'):

                        # Edge decisions
                        for i in range(len(true_dag)):
                            for j in range(len(true_dag)):
                                tp, fp, bi = 0, 0, 0


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
                                                tp, fp, bi, mch.min_mdl_node_[i][j], mch.min_mdl_node_[j][i],
                                                mch.min_gains_[i][j], mch.min_gains_[j][i]
                                            ],
                                        )
                                    ) + "\n"
                                    if write:
                                        write_file_edges.write(edge_decision)
                                        write_file_edges.flush()

                    #Print precision, recall for the DAG estimated up to environment n_env
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
                                runtime_all,
                                auroc2
                            ],
                        )
                    ) + "\n"
                    result_dag_search = ", ".join(
                        map(
                            str,
                            experimental_params + [
                                method_name,
                                soft,
                                n_env,
                                rep,
                                TP,
                                TN,
                                FP,
                                FN,
                                shd,
                                sid,
                                runtime_all
                            ],
                        )
                    ) + "\n"
                    if write:
                        write_file.write(result)
                        ##write_file_dagsearch.write(result_dag_search)
                        write_file.flush()
                        ##write_file_dagsearch.flush()
                    else:
                        results.append(result)
                        #results_dagsearch.append(result_dag_search)

                    print(str(n_env) + "/" + str(n_total_environments), "-Time:", np.round(runtime_all),
                                "\n    -e(tp,fp,fn,b)",
                                str(total_edges) + "(" + str(tp) + "," + str(fp) + "," + str(fn) + "," + str(bi) + ")",
                                "-p/r/fpr:", str(precision) + "/" + str(recall) + "/" + str(fpr), "\n    -SID:", sid,
                                "-AP:", ap, "-AUC_OPT:", auroc_opt, "-AUC:", auroc)

                    # Save pvalues
                if not os.path.exists(f'./results/pvalue_mats/{name}/'):
                    os.makedirs(f'./results/pvalue_mats/{name}/')
                if hasattr(mch, 'pvalues_'):
                    np.save(
                        f"./results/pvalue_mats/{name}/{name}_{save_name}_pvalues_params={params_index}_rep={rep}.npy",
                        mch.pvalues_,
                    )
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
            try:
                _run_rep(rep + rep_shift, write=True)
            except(PulpSolverError):
                continue

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
