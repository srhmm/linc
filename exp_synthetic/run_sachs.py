import numpy as np
import pandas as pd

from exp_synthetic.methods_linc import LINC
from exp_synthetic.settings import METHODS_SACHS
from sparse_shift import dag2cpdag, cpdag2dags, ParamChanges, MinChange, FullMinChanges
from sparse_shift import dag_true_orientations, dag_false_orientations, \
    dag_precision, dag_recall
import argparse

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



def main(args):
    for  save_name, method_name, mch, hyperparams  in METHODS_SACHS:

        results = []

        Xs = [
            np.log(
                pd.read_csv(f'../cytometry/dataset_{i}.csv')
            ) for i in range(1, 10)
        ]

        if args.quick:
            # Just two environments
            Xs = [np.array(X[:100]) for X in Xs[:3]]
        else:
            # Make envs of the same length
            Xs = [np.array(X[:707]) for X in Xs]



        if method_name=='linc_rff_nogain':
            mch = mch(cpdag=true_cpdag, dag=true_cpdag,  **hyperparams)
            for n_env, X in enumerate(Xs):
                mch.add_environment(X)
            min_cpdag = mch.get_min_dags(True)
        else:
            mch = mch(cpdag=true_cpdag,  **hyperparams)
            for n_env, X in enumerate(Xs):
                mch.add_environment(X)
            min_cpdag = mch.get_min_cpdag(True)


        true_orients = np.round(dag_true_orientations(true_dag, min_cpdag), 4)
        false_orients = np.round(dag_false_orientations(true_dag, min_cpdag), 4)
        precision = np.round(dag_precision(true_dag, min_cpdag), 4)
        recall = np.round(dag_recall(true_dag, min_cpdag), 4)

        results += [true_orients, false_orients, precision, recall]
        print(  np.round(precision, 4), ', ', np.round(recall, 4))

        vars = pd.read_csv(f'../cytometry/dataset_1.csv').columns
        print("Causal Edges")
        for i in range(len(min_cpdag)):
            for j in range(len(min_cpdag)):
                if min_cpdag[i][j]==1 and true_dag[i][j]==1:
                        print("\t", vars[i], "->", vars[j])
        print("Wrongly Oriented Edges")
        for i in range(len(min_cpdag)):
            for j in range(len(min_cpdag)):
                if min_cpdag[i][j]==1 and true_dag[i][j]==0:
                        print("\t", vars[i], "->", vars[j])

        print("Missing Edges")
        for i in range(len(min_cpdag)):
            for j in range(len(min_cpdag)):
                if min_cpdag[i][j]==0 and true_dag[i][j]==1:
                    print("\t", vars[i], "->", vars[j])
        print("GROUND TRUTH:")
        print("Causal Edges")
        for i in range(len(min_cpdag)):
            for j in range(len(min_cpdag)):
                if true_dag[i][j]==1:
                        print("\t", vars[i], "->", vars[j])
        print("Anticausal Edges")
        for i in range(len(min_cpdag)):
            for j in range(len(min_cpdag)):
                if true_dag[j][i]==1:
                        print("\t", vars[i], "<-", vars[j])


    #np.save(f"./results/sachs_gp_{save_name}_scores_env={n_env+1}.npy", mch.min_mdl_node_)


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

'''RESULTS RFF
8 :  0.8235 ,  0.8235

	 cd3cd28.Raf -> cd3cd28.Mek
	 cd3cd28.Mek -> cd3cd28.Erk
	 cd3cd28.PIP3 -> cd3cd28.PIP2
	 cd3cd28.Erk -> cd3cd28.Akt
	 cd3cd28.PKA -> cd3cd28.Raf
	 cd3cd28.PKA -> cd3cd28.Mek
	 cd3cd28.PKA -> cd3cd28.Erk
	 cd3cd28.PKA -> cd3cd28.Akt
	 cd3cd28.PKA -> cd3cd28.Jnk
	 cd3cd28.PKC -> cd3cd28.Raf
	 cd3cd28.PKC -> cd3cd28.Mek
	 cd3cd28.PKC -> cd3cd28.PKA
	 cd3cd28.PKC -> cd3cd28.P38
	 cd3cd28.PKC -> cd3cd28.Jnk
Wrongly Oriented Edges
	 cd3cd28.PIP2 -> cd3cd28.Plcg
	 cd3cd28.PIP3 -> cd3cd28.Plcg
	 cd3cd28.P38 -> cd3cd28.PKA
Missing Edges
	 cd3cd28.Plcg -> cd3cd28.PIP2
	 cd3cd28.Plcg -> cd3cd28.PIP3
	 cd3cd28.PKA -> cd3cd28.P38
	 f1 = tp /(tp + 1/2* (fp + fn)) = 0.8235294117647058
	 f1(pooled pc):0.75
	 
'''