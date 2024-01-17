import numpy as np
import pandas as pd
from sparse_shift.utils import dag2cpdag, cpdag2dags
from sparse_shift.methods import MinChange, AugmentedPC, FullMinChanges, ParamChanges
from sparse_shift.metrics import dag_true_orientations, dag_false_orientations, \
    dag_precision, dag_recall, average_precision_score
import argparse
import logging
import os

# CPDAG from the Sachs et al. paper
dag = np.zeros((11, 11))
dag[2, np.asarray([3, 4])] = 1
dag[4, 3] = 1
dag[8, np.asarray([10, 7, 0, 1, 9])] = 1
dag[7, np.asarray([0, 1, 5, 6, 9, 10])] = 1
dag[0, 1] = 1
dag[1, 5] = 1
dag[5, 6] = 1

true_dag = dag
true_cpdag = dag2cpdag(true_dag)
mec_size = len(cpdag2dags(true_cpdag))
total_edges = np.sum(true_dag)
unoriented_edges = np.sum((true_cpdag + true_cpdag.T) == 2) // 2

METHODS = [
    (
        'mc',
        'MC',
        ParamChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
        }
    ),
    '''
    
    (
        'full_pc_kci',
        'Full PC (KCI)',
        FullMinChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'kci',
            'test_kwargs': {},
        }
    ),
    (
        'mch_kci',
        'Min changes (KCI)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'kci',
            'test_kwargs': {
                "KernelX": "GaussianKernel",
                "KernelY": "GaussianKernel",
                "KernelZ": "GaussianKernel",
            },
        }
    ),'''
]


def main(args):
    soft = True
    # FULL PC KCI
    # Compute empirical results
    print("CELL CYTOMETRY DATA")
    for m in range(len(METHODS)):
        save_name, method_name, mch, hyperparams = METHODS[m]
        print("METHOD: ", method_name)
        mch = mch(cpdag=true_cpdag, **hyperparams)
        results = []

        Xs = [
            np.log(
                pd.read_csv(f'dataset_{i}.csv')
            ) for i in range(1, 10)
        ]

        if args.quick:
            # Just two environments
            Xs = [X[:100] for X in Xs[:3]]

        for n_env, X in enumerate(Xs):
            mch.add_environment(X)
            print("ENVIRONMENT:", n_env, "soft:", soft)
            min_cpdag = mch.get_min_cpdag(soft)
            true_orients = np.round(dag_true_orientations(true_dag, min_cpdag), 4)
            false_orients = np.round(dag_false_orientations(true_dag, min_cpdag), 4)
            precision = np.round(dag_precision(true_dag, min_cpdag), 4)
            recall = np.round(dag_recall(true_dag, min_cpdag), 4)

            results += [true_orients, false_orients, precision, recall]
            print(n_env, ': ', np.round(precision, 4), ', ', np.round(recall, 4))
            if not os.path.exists('results/'):
                os.makedirs('results/')
            np.save(f"./results/cytometry_{save_name}_pvalues_env={n_env+1}.npy", mch.pvalues_)
            print(n_env, ': ', np.round(precision, 4), ', ', np.round(recall, 4))
            if not os.path.exists('results/'):
                os.makedirs('results/')

            print("")
            vars = pd.read_csv(f'dataset_1.csv').columns
            print("Causal Edges")
            for i in range(len(min_cpdag)):
                for j in range(len(min_cpdag)):
                    if min_cpdag[i][j] == 1 and true_dag[i][j] == 1:
                        print("\t", vars[i], "->", vars[j])
            print("Wrongly Oriented Edges")
            for i in range(len(min_cpdag)):
                for j in range(len(min_cpdag)):
                    if min_cpdag[i][j] == 1 and true_dag[i][j] == 0:
                        print("\t", vars[i], "->", vars[j])

            print("Missing Edges")
            for i in range(len(min_cpdag)):
                for j in range(len(min_cpdag)):
                    if min_cpdag[i][j] == 0 and true_dag[i][j] == 1:
                        print("\t", vars[i], "->", vars[j])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        help="Enable to run a smaller, test version",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    main(args)
