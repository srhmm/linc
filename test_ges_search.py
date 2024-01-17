import time

import numpy as np
from cdt.metrics import SHD_CPDAG, SID_CPDAG

from function_types import FunctionType
from gen_context_data import gen_context_data
from unused.ges_mdl import GesMDL
from intervention_types import IvType
from out import Out
from vsn import Vsn


def test_ges(fun_type=FunctionType.LINEAR_GAUSS,
             iv_type=IvType.CONST,
             D_n=500, C_n=5,
             initial_seed=1,
             iters = 10
             ):
    shds, counts = 0,0
    node_n = 5
    vsn = Vsn(rff=True,
              mdl_gain=True,
              regression_per_group=False,
              regression_per_pair=False,
              ilp_in_tree_search=True,
              structure=False)

    file_nm = str(C_n) + '_' + str(D_n) + '_' + str(node_n) + '_' \
              + str(fun_type) + '_' + str(iv_type) + '_' + str(initial_seed) + '.txt'

    out = Out("ges/res_" + file_nm)
    for seed in range(iters):
        actual_seed = seed + initial_seed
        st = time.perf_counter()

        Dc, G, Gc, _, _, _, _, Pis, obs = gen_context_data(D_n=D_n, C_n=C_n, partition_search=False,
                                                           fun_type=fun_type,
                                                           iv_type_target=iv_type,  # doesnt matter here,
                                                           iv_type_covariates=iv_type,
                                                           iid_contexts=False, iv_in_groups=True, iv_per_node=[1,4],
                                                           node_n=node_n, seed=actual_seed)
        T = GesMDL(np.array(Dc), vsn)


        #TODO: include subset search for emp vario (vario emp + tree search),
        # full MDL score for vario with subset search (vario_MDL + MDL error term + tree_search)
        # and without subset search (linLinc)
        out.printto("DAG estimated")
        out.printto(T)
        out.printto("DAG truth")
        out.printto("\t"+str(G.arcs))

        trueG = np.array([[(0 if n == 0 else 1) for n in G.weight_mat[:, j]] for j in range(len(G.weight_mat))])
        #shd = SHD(trueG, T['G'].graph)
        #sid = SID(trueG, T['G'].graph)
        shd = SHD_CPDAG(trueG, T['G'].graph)
        sid = SID_CPDAG(trueG, T['G'].graph)
        shds, sids, counts = shds + shd, sids + sid, counts + 1
        out.printto("SHD:", shd)
        out.printto("SID:", shd)
        print("GES Time: " , time.perf_counter() - st)

    out.printto("*** Evaluation ***\nNodes N="+str(node_n) + "\nIvType:"+str(iv_type)+ ", FunType:"+str(fun_type)+
                "\nSHD: " +str(round(shds/counts, 2)) )
    out.close()

