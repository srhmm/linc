import numpy as np
import pandas as pd

from exp_synthetic.methods_linc import LINC
from sparse_shift import dag_true_orientations, dag_false_orientations
import argparse


METHODS = [
    (
        'linc_rff_nogain',
        'linc_rff_nogain',
        LINC,
        {
            'rff': False,
            'mdl_gain': False,
            'ILP': True,
            'clus': False,
            'known_mec': True,
            'pi_search': False
        },
    )
]


def main(args):
    # An unoriented DAG
    dag = np.zeros((2, 2))
    dag[0, 1] = 1
    dag[1, 0] = 1

    results = [0, 0, 0, 0]
    results_w = [0, 0, 0, 0]
    confs = []
    decisions = []
    pairmeta = pd.read_csv('../tuebingen/pairs/pairmeta.txt', header=None, delimiter="\s")
    indices = pairmeta[0]

    for i in range(0,41): #44, max(indices)):
        #if (i==42) or (i==43):
        #    continue
        pair_index = pairmeta[0][i]

        true_dag = np.zeros((2,2))
        if pairmeta[1][i] > 2 or pairmeta[3][i] > 2 \
            or pairmeta[1][i] != pairmeta[2][i] or pairmeta[3][i] != pairmeta[4][i]:
            continue
        #cand:
        #    pa = [i for i in self.nodes if cand[j][i]==1] #adj[i][j]==1]
        if pairmeta[1][i]==1:
            true_dag[0][1] = 1
            print("Pair:", pair_index, "X->Y")
        else:
            true_dag[1][0] = 1
            print("Pair:", pair_index, " <- ")
        if pair_index<10:
            fnm = f'../tuebingen/pairs/pair000{pair_index}.txt'
        else:
            fnm = f'../tuebingen/pairs/pair00{pair_index}.txt'
        X =  pd.read_csv(fnm, header = None, delimiter=r'[\s]+')

        Xs = [np.array(X[:min(len(X), 1000)])] # np.array(X[:int(np.floor(len(X)/2))], dtype=float),  np.array(X[int(np.floor(len(X)/2)):], dtype=float)]

        # Compute empirical results
        save_name, method_name, mch, hyperparams = METHODS[0]
        mch = mch(cpdag=dag, dag=true_dag, **hyperparams)

        for n_env, Xi in enumerate(Xs):
            mch.add_environment(Xi)

        min_cpdag = mch.get_min_dags(True)
        print("")

        true_orients = np.round(dag_true_orientations(true_dag, min_cpdag), 4)
        false_orients = np.round(dag_false_orientations(true_dag, min_cpdag), 4)

        w = pairmeta[5][i]
        results[0] += true_orients
        results_w[0] += true_orients * w
        results[1] += false_orients
        results_w[1] += false_orients * w
        decisions += [true_orients]
        confs += [np.round(mch.min_mdl_node_[np.nonzero(mch.min_mdl_node_)][0], 4)]
        print("Decision: ", true_orients, false_orients, w, np.round(mch.min_mdl_node_[np.nonzero(mch.min_mdl_node_)][0],2))
        print("Overall: ", results, "weighted:", results_w)

    print(results)
    print(decisions)
    print(confs)

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
