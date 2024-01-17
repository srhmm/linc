import time

import numpy as np
from numpy.linalg import linalg
from sklearn.gaussian_process.kernels import RBF

from gpr_mdl import GaussianProcessRegularized
from pi import PI
from pi_search_ilp import pi_search_ILP
from pi_search_mdl import pi_mdl_conf, pi_mdl_best

from function_types import FunctionType
from gen_context_data import gen_context_data
from intervention_types import IvType
from regression import group_gaussianProcessRegression
from rff_mdl import GaussianProcessRFF
from rff_mdl_new import GaussianProcessRFFPi
from utils import data_groupby_pi
from utils_pi import pi_enum, pi_convertfrom_pair
from vsn import Vsn

def test_regression():
    Dc, dag, Gc, \
    target, parents, children, confounder, \
    partitions_X, observational_X = gen_context_data(5, 500, 5, 1, FunctionType.LINEAR_GAUSS, IvType.CONST,
                                                     iv_type_covariates=IvType.CONST, iid_contexts=False,
                                                     partition_search=False)

    Xc = np.array([Dc[c_i][:, [p for p in parents]] for c_i in range(5)])
    yc = np.array([Dc[c_i][:, target] for c_i in range(5)])
    partition = partitions_X[target]
    Xpi, ypi = data_groupby_pi(Xc, yc, partition)
    rff_reg = group_gaussianProcessRegression(Xpi, ypi, rand_fourier_features=True)# show_plt= True since for this seed, X is one-dim.
    exact_reg = group_gaussianProcessRegression(Xpi, ypi, rand_fourier_features=False, show_plt=True)

    rff = GaussianProcessRFF(n_components = 50)
    rff.fit(Xc[0], yc[0])
    rff.mdl_score_ytrain()

    gpr = GaussianProcessRegularized(kernel=1 * RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2)), alpha=1.5)
    gpr.fit(Xpi[0], ypi[0])


    linalg.norm(gpr.kernel_(gpr.X_train_) - rff.kernel_approx_)
    #for scale:
    linalg.norm(rff.kernel_approx_ - rff.kernel_approx_ + 0.4)
    linalg.norm(rff.kernel_approx_ - rff.kernel_approx_ + 0.5)

    rff2 = GaussianProcessRFF(n_components = 100)
    rff2.fit(Xc[0], yc[0])

    linalg.norm(gpr.kernel_(gpr.X_train_) - rff.kernel_approx_)
    linalg.norm(gpr.kernel_(gpr.X_train_) - gpr.kernel_(gpr.X_train_) + .3)


    #GP (592.3593409721158, 574.9210569865618, 6.129351527558802, 11.308932457995237)
    #LIN (1457.2510464139332, 1454.974488749066, 1.2484163685022713, 1.02814129636487)


    #partitions_X[3]

    Dc, dag, Gc, \
    target, parents, children, confounder, \
    partitions_X, observational_X = gen_context_data(5, 500, 5, 1, FunctionType.LINEAR_GAUSS, IvType.CONST,
                                                     iv_type_covariates=IvType.CONST, iid_contexts=False,
                                                     partition_search=False)
    Pis = pi_enum(5)
    scores = [0 for i in range(len(Pis))]
    Xc = np.array([Dc[c_i][:, [p for p in parents]] for c_i in range(5)])
    yc = np.array([Dc[c_i][:, target] for c_i in range(5)])
    true_partition = partitions_X[target]
    st = time.perf_counter()

    #TODO rerun these w full capac
    #mimics exhaustive version with rffs
    for i in range(len(Pis)):
        partition = Pis[i]
        Xpi, ypi = data_groupby_pi(Xc, yc, partition)
        rff_reg = group_gaussianProcessRegression(Xpi, ypi,
                                                  rand_fourier_features=True)  # show_plt= True since for this seed, X is one-dim.
        scores[i] = sum([s.mdl_score_ytrain()[0] for s in rff_reg]) #
        #scores[i] = sum([s.mdl_score_ytest(s.X_train_, s.y_train_)[0] for s in rff_reg]) #60 sec
    print ( "Time: " , time.perf_counter() - st)
    min(scores)

    # simulates exh version with rffs
    #TODO



    p = PI(Xc, yc, np.random.RandomState(1), rff=True)
    gain, _, _ = pi_mdl_conf(p, true_partition, 5, False, None)
    gain, _, _ = pi_mdl_conf(p, [[0,1],[2,3,4]], 5, False, None)

    st = time.perf_counter()
    gp_reg = group_gaussianProcessRegression(Xc, yc,
                                             rand_fourier_features=False)  # 20 sec

    # simulates exh version without rffs
    #TODO
    st = time.perf_counter()
    p = PI(Xc, yc, np.random.RandomState(1), rff=False)
    print("Intermed. Time: ", time.perf_counter() - st) #20sec
    pi_guess = pi_mdl_best(p, Vsn(False, False, False, False, False, False)) #98 sec
    print("Time: ", time.perf_counter() - st)





def gen_example_data():
    Dc, dag, Gc, \
    target, parents, children, confounder, \
    partitions_X, observational_X = gen_context_data(5, 500, 5, 1, FunctionType.LINEAR_GAUSS, IvType.CONST,
                                                     iv_type_covariates=IvType.CONST, iid_contexts=False,
                                                     partition_search=False)
    Xc = np.array([Dc[c_i][:, [p for p in parents]] for c_i in range(5)])
    yc = np.array([Dc[c_i][:, target] for c_i in range(5)])
    return Xc, yc


# EXHAUSTIVE VERSIONS -------


# def time_rff_allpartitions_regressionpergroup(Xc, yc):
#     st = time.perf_counter()
#     p = PI(Xc, yc, np.random.RandomState(1), rff=True)
#     print("Intermed. Time: ", time.perf_counter() - st)  # 0.8sec - regression fit for 5 contexts
#     pi_guess = pi_mdl_best(p, Vsn(True, False, True, False, False,
#                                   False))  # 24-38sec - fitting again for each group in each partition - interestingly, this is faster than per context!! although it is less exact
#     print("Time: ", time.perf_counter() - st)
#     for pi in p.pi_cache:
#         rfflist = p.pi_cache[pi]
#         print(pi, sum([rff.mdl_score_ytrain()[0] for rff in rfflist]))

    #pi:guess is one group - fix rffs


def time_rff_allpartitions_regressionpercontext(Xc, yc):
    Xc, yc = gen_example_data()
    st = time.perf_counter()
    p = PI(Xc, yc, np.random.RandomState(1), rff=True)
    print("Intermed. Time: ", time.perf_counter() - st) #0.9sec - regression fit for 5 contexts
    pi_guess = pi_mdl_best(p, Vsn(True, False, False, False, False, False)) #60sec - no fitting anymore but mdl estimates for each partition, as in regression
    print("Time: ", time.perf_counter() - st)

# def time_exactgp_allpartitions():
#     Xc, yc = gen_example_data()
#     st = time.perf_counter()
#     p = PI(Xc, yc, np.random.RandomState(1), rff=False)
#     print("Intermed. Time: ", time.perf_counter() - st) #20sec
#     pi_guess = pi_mdl_best(p, Vsn(False, False, False, False, False, False)) #70 sec
#     print("Time: ", time.perf_counter() - st)
    #pi_guess is correct, two groups



# ILP VERSIONS -------
# def time_rff_ilp_regressionperpair():
#     Xc, yc = gen_example_data()
#     C_n = 5
#     st = time.perf_counter()
#     p = PI(Xc, yc, np.random.RandomState(1), rff=True, skip_regression_pairs=False)
#     p.cmp_distances(from_G_ij=True)
#     print("Intermed. Time: ", time.perf_counter() - st) # 3 sec - pairwise fitting, slightly faster than pairwise pred but not of note
#     dists = p.pair_mdl_gains
#     vars_ij, _, c = pi_search_ILP(dists, C_n, wasserstein=False) # .1sec
#     # TODO
#     print("Time: ", time.perf_counter() - st)
#     guess_pi = pi_convertfrom_pair(vars_ij, c, C_n)

def time_rff_ilp_regressionpercontext():
    Xc, yc = gen_example_data()
    C_n = 5
    st = time.perf_counter()

    p = PI(Xc, yc, np.random.RandomState(1), rff=True, skip_regression_pairs=True)
    p.cmp_distances(from_G_ij=False)
    print("Intermed. Time: ", time.perf_counter() - st) #  5sec - pairwise predictions
    dists = p.pair_mdl_gains
    vars_ij, _, c = pi_search_ILP(dists, C_n, wasserstein=False) #.1 sec
    guess_pi = pi_convertfrom_pair(vars_ij, c, C_n)
    print("Time: ", time.perf_counter() - st)


def time_exactgp_ilp():
    Xc, yc = gen_example_data()
    C_n = 5
    st = time.perf_counter()
    p = PI(Xc, yc, np.random.RandomState(1), rff=False, skip_regression_pairs=True)
    print("Intermed. Time: ", time.perf_counter() - st) #20sec
    p.cmp_distances(from_G_ij=False)
    print("Intermed. Time: ", time.perf_counter() - st) #7sec
    dists = p.pair_mdl_gains
    vars_ij, _, c = pi_search_ILP(dists, C_n, wasserstein=False) #.1sec
    print("Time: ", time.perf_counter() - st)
    guess_pi = pi_convertfrom_pair(vars_ij, c, C_n)

