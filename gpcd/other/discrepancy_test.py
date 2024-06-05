import itertools
import math

import numpy as np
from enum import Enum

from .util import pval_to_map, data_scale, is_insignificant
from .util_tests import test_mechanism, test_mechanism_regimes

from itertools import product
import warnings

class DiscrepancyTestType(Enum):
    SKIP = 0  # e.g. if we only need oracle results
    KCD = 1  # kernel cond discrepancy test
    GP = 2  # gaussian process regression and MDL

    def __str__(self):
        if self.value == 0:
            return 'SKIP'
        elif self.value == 1:
            return 'KCD'
        elif self.value == 2:
            return 'GP'
        else:
            return 'None'

    def __eq__(self, other):
        return self.value == other.value


def discrepancy_test_all(gp_time_space, cs, ws, pa_i: list, discrepancy_test_type: DiscrepancyTestType, alpha=0.5, verbosity=0):
    """ Tests conditional distributions for equality for a pair of contexts
    :param cs: list of contexts
    :param ws: list of windows
    :param gp_time_space:
    :param pa_i: causal parents of target
    :param discrepancy_test_type: test type
    :return:
    """
    # Dc = None
    Dc_list = list()
    Ds = dict.fromkeys(product(cs, ws))
    for (c, w) in Ds.keys():
        _, _, _, data, D = gp_time_space[c][w]
        Dc_list.append(np.hstack((D, data.reshape(-1, 1))))
        # if Dc is None:
        #     Dc = np.expand_dims(np.array(np.hstack((D, data.reshape(-1, 1)))), 0)
        # else:
        #     tmp = np.expand_dims(np.hstack((D, data.reshape(-1, 1))), 0)
        #     Dc = np.concatenate((Dc, tmp), axis=0)

    n_c = len(Dc_list)

    n_pairs = len([i for i in itertools.combinations(range(n_c), 2)])
    pval_mat = np.ones((n_c, n_c))

    if discrepancy_test_type == DiscrepancyTestType.SKIP:
        map = [0 for _ in range(n_c)]

    elif discrepancy_test_type == DiscrepancyTestType.KCD:
        # parents = [1 if node_i in pa_i else 0 for node_i in range(Dc.shape[2])]

        parents = [1 if node_i in pa_i else 0 for node_i in range(Dc_list[0].shape[1])]
        # pval_mat = test_mechanism_regimes(Dc, Dc[0].shape[1]-1, parents, ws, 'kci', {}) # H0: c ⊥ i | pa_i i.e. same contexts
        pval_mat = test_mechanism_regimes(Dc_list, Dc_list[0].shape[1]-1, parents, ws, 'kci', {}) # H0: c ⊥ i | pa_i i.e. same contexts
        map = pval_to_map(pval_mat, alpha=alpha) # high p-value = same context (proba of the obs under H0)

    elif discrepancy_test_type == DiscrepancyTestType.GP:
        for i, (c_i, w_i) in enumerate(product(cs, ws)):
            for j, (c_j, w_j) in enumerate(product(cs, ws)):
                if (c_i, w_i) == (c_j, w_j): continue
                _, gp_i, data_pa_i, data_i, D1 = gp_time_space[c_i][w_i]
                _, gp_j, data_pa_j, data_j, D2 = gp_time_space[c_j][w_j]

                data_pa_ij = np.concatenate([data_pa_i, data_pa_j])
                data_ij = np.concatenate([data_i, data_j])

                mdl_ij, loglik_ij, _, _ = gp_i.mdl_score_ytest(data_pa_j, data_j)
                mdl_ji, loglik_ji, _, _ = gp_j.mdl_score_ytest(data_pa_i, data_i)
                mdl_ii, loglik_ii, _, _ = gp_i.mdl_score_ytest(data_pa_i, data_i)
                mdl_jj, loglik_jj, _, _ = gp_j.mdl_score_ytest(data_pa_j, data_j)

                assert(loglik_ij > 0 and loglik_ji > 0
                       and loglik_ii > 0 and loglik_jj > 0)
                eq_i = is_insignificant(abs(loglik_ii - loglik_ji))
                eq_j = is_insignificant(abs(loglik_jj - loglik_ij)) # loglik_jj - loglik_ij
                pval_mat[i, j] = int(eq_i and eq_j) # TODO: or or and?
        map = pval_to_map(pval_mat, alpha=alpha)

    else:
        raise ValueError()

    return map, pval_mat


def discrepancy_test(gp_time_space, c_i, w_i, c_j, w_j,
                     node_i: int, pa_i: list, discrepancy_test_type: DiscrepancyTestType, alpha=0.5, verbosity=0):
    """ Tests conditional distributions for equality for a pair of contexts
    :param gp_time_space:
    :param node_i: target node
    :param pa_i: causal parents of target
    :param discrepancy_test_type: test type
    :return:
    """
    max_lag = 100

    _, gp_i, data_pa_i, data_i, D1 = gp_time_space[c_i][w_i]
    _, gp_j, data_pa_j, data_j, D2 = gp_time_space[c_j][w_j]

    Dc = np.array([np.hstack((D1, data_i.reshape(-1, 1))),
                   np.hstack((D2, data_j.reshape(-1, 1)))]) # target node values at time t in last position
    n_c = Dc.shape[0]
    n_vars = D1.shape[1]
    assert (n_vars == D2.shape[1])

    n_samples = min(Dc[0].shape[0], Dc[1].shape[0])

    n_pairs = len([i for i in itertools.combinations(range(n_c), 2)])
    pval_mat = np.ones((n_pairs, n_pairs))

    if discrepancy_test_type == DiscrepancyTestType.SKIP:
        map = [0 for _ in range(n_c)]

    elif discrepancy_test_type == DiscrepancyTestType.KCD:
        parents = [1 if node_i in pa_i
            or( True in [(node_i, t) in pa_i for t in range(max_lag) ])
                   else 0 for node_i in range(n_vars)]
        try:
            pval_mat = test_mechanism(Dc, n_vars, parents, 'kci', {}) # H0: c ⊥ i | pa_i i.e. same contexts
            map = pval_to_map(pval_mat, alpha=alpha)  # high p-value = same context (proba of the obs under H0)

        except ValueError:
            map = [0, 0]
            pval_mat = None
            print("val err") #zero variance error

    elif discrepancy_test_type == DiscrepancyTestType.GP:
        data_pa_ij = np.concatenate([data_pa_i, data_pa_j])
        data_ij = np.concatenate([data_i, data_j])

        mdl_ij, loglik_ij, _, _ = gp_i.mdl_score_ytest(data_pa_j, data_j)
        mdl_ji, loglik_ji, _, _ = gp_j.mdl_score_ytest(data_pa_i, data_i)
        mdl_ii, loglik_ii, _, _ = gp_i.mdl_score_ytest(data_pa_i, data_i)
        mdl_jj, loglik_jj, _, _ = gp_j.mdl_score_ytest(data_pa_j, data_j)

        assert(loglik_ij > 0 and loglik_ji > 0
               and loglik_ii > 0 and loglik_jj > 0)

        eq_i = is_insignificant(np.abs(loglik_ii - loglik_ji))
        eq_j = is_insignificant(np.abs(loglik_jj - loglik_ij))

        parents = [1 if node_i in pa_i else 0 for node_i in range(Dc.shape[2]-1)]
        pval_mat = test_mechanism(Dc, D1.shape[1], parents, 'kci', {}) # H0: c ⊥ i | pa_i i.e. same contexts
        map = pval_to_map(pval_mat, alpha=alpha) # high p-value = same context (proba of the obs under H0)

        map = [0, 0] if (eq_i or eq_j) else [0, 1]
        if verbosity > 0:
            print(f'\tCase {w_i}x{c_i}x{w_j}x{c_j}:\tmap={map}\teq_i: {eq_i}, eq_j: {eq_j}')  # , \teq_joint:{eq_joint[0][0]} ')

        #debug
        parents = [1 if node_i in pa_i else 0 for node_i in range(Dc.shape[2]-1)]
        pval_mat = test_mechanism(Dc, D1.shape[1], parents, 'kci', {})  # H0: c ⊥ i | pa_i i.e. same contexts
        map = pval_to_map(pval_mat, alpha=alpha)  # high p-value = same context (proba of the obs under H0)
        print(f"\tCase {w_i}x{c_i}x{w_j}x{c_j}:\tmap={map}\teq_i: {eq_i}, eq_j: {eq_j}, eq_kci: {pval_to_map(test_mechanism(Dc, D1.shape[1], parents, 'kci', {}), alpha=alpha)}")

    else:
        raise ValueError()



    return map, pval_mat


def _augment(Dc):
    n = Dc.shape[2] + 1
    D = np.random.normal(size=(Dc.shape[0], Dc.shape[1], n))
    D[:, :, range(n - 1)] = Dc
    return D, [n]