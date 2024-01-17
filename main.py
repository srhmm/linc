import time

import utils_pi
from data_read import read_sachs
from test_pi_search import test_partition_search
from tree_search import tree_search
from intervention_types import IvType
from function_types import FunctionType
from test_tree_search import test_tree_search

from vsn import Vsn

if __name__ == '__main__':

    reproduce_pi = False
    reproduce_dag = False
    reproduce_sachs = True

    if reproduce_pi:
        rff = True
        ilp = False
        run_repeats = 10
        initial_seed = 142
        D_n = 750
        k_max=2
        for fun_type in [FunctionType.GP_PRIOR]:
            #TODO next: rff no ilp for partition search; fix linear (all but param change); do sergio thing; maybe new 2022 cell dataset; rff w new likelihood for DAG search
            # TODO maybe pursue subset intersection idea again?
            for iv_type in [IvType.PARAM_CHANGE, IvType.SHIFT, IvType.SCALE]:
                for node_n in [5]:
                    for C_n in [5]:
                        partitions = utils_pi.pi_enum(C_n, k_max=2, k_min=2, permute=False)
                        for iv_per_node in [[1, 1]]:  # , [1,1], [1,2], [1,3]]:
                            vsn = Vsn(rff=rff, ilp_in_tree_search=True,
                                      regression_per_pair=False, #True,
                                      regression_per_group=False,
                                      mdl_gain=True)
                            st = time.perf_counter()
                            test_partition_search(vsn, partitions, len(partitions),  run_repeats, C_n, D_n, node_n, iv_type, iv_type,
                                                  iv_per_node,False, fun_type, initial_seed=initial_seed, only_ilp=ilp)
                            print('PI runtime: ', round(time.perf_counter() - st, 2) )
    if reproduce_dag:
        rff=True
        ilp=True

        mdl_gain=False
        run_id = 0
        run_repeats = 10 #50
        initial_seed = 42
        D_n = 750
        begin = time.perf_counter()
        for fun_type in [FunctionType.GP_PRIOR]:
            for iv_type in [IvType.PARAM_CHANGE, IvType.SCALE, IvType.CONST]:
                #for ilp in [True, False]: #for rff in [ True, False ]:
                for node_n in [5]: # [ 2,3,4,5,6,Ï7 ]:
                        for C_n in [5]: #[ 2,3,4,5 ]:
                            for iv_per_node in [[1,1]]:#, [2,2], [3,3]]:  #, [1,1], [1,2], [1,3]]:
                                vsn = Vsn(rff=rff, ilp_in_tree_search=ilp,
                                          mdl_gain=mdl_gain)
                                st = time.perf_counter()
                                test_tree_search(C_n=C_n, D_n=D_n, node_n=node_n,
                                                 fun_type=fun_type, iv_type=iv_type, iv_per_node=iv_per_node,
                                                 vsn=vsn, initial_seed=initial_seed,
                                                 run_id=run_id,iters=run_repeats)
                                print('TREE runtime: ', round(time.perf_counter() - st, 2), '\nTREE result:')

                                # Can read files with true and estimated DAGs and compute the SIDs with the following
                                #file_nm = out_naming.unique_name(C_n, D_n, node_n, fun_type, iv_type, vsn, iv_per_node)
                                #read_dag(file_nm, node_n, runs=run_repeats,
                                #        path="", offset=iv_per_node[1]-1)
        print('Overall runtime: ', round(time.perf_counter() - begin, 2))


    '''
    Sachs et al. dataset
    '''
    if reproduce_sachs:
        Dc, nms = read_sachs()
        T = tree_search(Dc, Vsn(rff=True), revisit_children=True, revisit_queue=True, revisit_parentsets=True, vb=False, tofile=False)
        print(T.adj)
        for (i,j) in T.adj:
            print("\t", nms[i], "->", nms[j])
        print("")
        # plcg -> PIP3
	    # PIP3 -> PIP2
	    # PKC -> pjnk
	    # p44/42 -> pmek
	    # p44/42 -> PKA
	    # PKC -> P38
	    # p44/42 -> praf
	    # p44/42 -> pakts473
	    # PKC -> plcg
	    # PKC -> p44/42
