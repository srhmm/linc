import time

from pi import PI
from utils import data_groupby_pi
import numpy as np
from utils_pi import pi_enum, pi_enum_split

"""Help functions for exhaustive search for a partition Pi of contexts into groups"""


def pi_mdl(context_pi: PI, partition: list,
           regression_per_group,
           subsample_size=None,
           normalize=False, # if regr per group
           min_max=False #if not regression per group: whether to choose the best (min) partition s.t. the worst (max) context model encodes each group (otherwise min min)
           ):
    """ MDL score for a partition

    :param context_pi: pi class
    :param partition: current partition
    :param regression_per_group: whether regression within each group, otherwise regression only in each single context
    :return: MDL score sum, MDL score per group, MDL model score, MDL data score, MDL penalty
    """

    # Pool data per group ------------
    Xc, yc = context_pi.Xc, context_pi.yc
    C_n = context_pi.C_n
    if subsample_size is not None:
        Xc = np.array([Xc[c_i][: subsample_size] for c_i in range(C_n)])
        yc = np.array([yc[c_i][: subsample_size] for c_i in range(C_n)])
    Xpi, ypi = data_groupby_pi(Xc, yc, partition)
    mdl_data = [0 for _ in partition]
    mdl_model = [0 for _ in partition]
    mdl_pen = [0 for _ in partition]
    mdl = [0 for _ in partition]

    # GP models per group ------------
    GP_k = None
    if regression_per_group:
        GP_k = context_pi.pi_lookup(partition)

    # MDL score per group ------------
    for g_k in range(len(partition)):
        group_k = partition[g_k]
        mdl_data_k, mdl_model_k, mdl_pen_k = 0, 0, 0

        # Use the group GP to encode all contexts ------------
        if regression_per_group:
            D_min = np.inf
            for c_i in group_k: #TODO: does min over contexts make sense?
                D_n = context_pi.Xc[c_i].shape[0]
                if D_n < D_min:
                    D_min = D_n
                #before:
                #    mdl_i, mdl_data_i, mdl_model_i, mdl_pen_i = GP_k[g_k].mdl_score_ytrain_normalized(D_n)
                #    mdl_k, mdl_data_k, mdl_model_k, mdl_pen_k = mdl_k + mdl_i, mdl_data_k + mdl_data_i, mdl_model_k + mdl_model_i, mdl_pen_k + mdl_pen_i
            if normalize:
                mdl_k, mdl_data_k, mdl_model_k, mdl_pen_k = GP_k[g_k].mdl_score_ytrain_normalized(D_min)
            else:
                #doesnt work: mdl_k, mdl_data_k, mdl_model_k, mdl_pen_k = GP_k[g_k].mdl_score_ytrain_scaled(D_min)
                mdl_k, mdl_data_k, mdl_model_k, mdl_pen_k = GP_k[g_k].mdl_score_ytrain()



        # Use the context-specific GP that can encode all contexts in the group with fewest bits ------------
        if not regression_per_group:
            mdl_k = np.inf
            if min_max:
                mdl_k = -np.inf
            assert context_pi.GP_c is not None
            # Find "barycenter"
            for c_i in group_k:
                Xk, yk = Xpi[g_k], ypi[g_k]  # The pooled data
                mdl_i, mdl_data_i, mdl_model_i, mdl_pen_i = context_pi.GP_c[c_i].mdl_score_ytest(Xk, yk)

                # A. "barycenter": if mdl_i < mdl_k:
                # B. "min-max":
                if (min_max and mdl_i > mdl_k) or (not min_max and mdl_i < mdl_k):
                    mdl_k = mdl_i
                    mdl_data_k = mdl_data_i
                    mdl_model_k = mdl_model_i
                    mdl_pen_k = mdl_pen_i

        mdl[g_k] = mdl_k
        mdl_data[g_k] = mdl_data_k
        mdl_model[g_k] = mdl_model_k
        mdl_pen[g_k] = mdl_pen_k

    pass
    return sum(mdl), mdl, sum(mdl_model), sum(mdl_data), sum(mdl_pen)

#
# def pi_mdl_adapt(context_pi: PI,
#            subsample_size=None):
#     """ Pairwise adaptation cost v1
#
#     :param context_pi: pi class
#     :return: MDL score sum, MDL score per context, MDL model score, MDL data score, MDL penalty
#     """
#
#     Xc, yc = context_pi.Xc, context_pi.yc
#     C_n = context_pi.C_n
#     if subsample_size is not None:
#         Xc = np.array([Xc[c_i][: subsample_size] for c_i in range(C_n)])
#         yc = np.array([yc[c_i][: subsample_size] for c_i in range(C_n)])
#
#     mdl_adapt = [0 for _ in range(C_n)]
#     mdl_self = [(0,0,0) for _ in range(C_n)]
#
#     for c_i in range(C_n):
#         mdl_i, mdldata_i, mdlmodel_i, _ = context_pi.GP_c[c_i].mdl_score_ytrain()
#         mdl_self[c_i] = (mdl_i, mdldata_i, mdlmodel_i)
#
#         # MDL score per context ------------
#     for c_i in range(C_n):
#         mdl_adapt_i = 0
#         mdl_i, mdldata_i, mdlmodel_i = mdl_self[c_i]
#         # Sum adaptation costs ------------
#         for c_j in range(C_n):
#             if c_j <= c_i:
#                 continue
#             mdl_j, mdldata_j, mdlmodel_j = mdl_self[c_j]
#             Xpi, ypi = data_groupby_pi(Xc, yc, Pi=[[c_i, c_j]])
#             Xij, yij = Xpi[0], ypi[0]
#             Xi, yi = Xc[c_i], yc[c_i]
#             Xj, yj = Xc[c_j], yc[c_j]
#             v1 = False
#             if v1:
#                 mdl_i_joint, mdldata_i_joint, mdlmodel_i_joint, _ = context_pi.GP_c[c_i].mdl_score_ytest(Xij, yij)
#                 mdl_j_joint, mdldata_j_joint, mdlmodel_j_joint, _ = context_pi.GP_c[c_j].mdl_score_ytest(Xij, yij)
#                 mdl_adapt_i = mdl_adapt_i + min(mdl_i + mdl_j - mdl_i_joint,  mdl_i + mdl_j - mdl_j_joint)
#             else:
#                 print("v2")
#                 # mdl_ij, _, _, _ = context_pi.GP_c[c_i].mdl_score_ytest(Xj, yj)
#                 #mdl_ji, _, _, _ = context_pi.GP_c[c_j].mdl_score_ytest(Xi, yi)
#                 mdl_i_joint, mdldata_i_joint, mdlmodel_i_joint, _ = context_pi.GP_c[c_i].mdl_score_ytest(Xij, yij)
#                 mdl_j_joint, mdldata_j_joint, mdlmodel_j_joint, _ = context_pi.GP_c[c_j].mdl_score_ytest(Xij, yij)
#                 mdl_adapt_i = mdl_adapt_i + min(min(mdl_i + mdl_j, mdl_i_joint),  min(mdl_i + mdl_j, mdl_j_joint))
#         mdl_adapt[c_i] = mdl_adapt_i
#
#     return sum(mdl_adapt), mdl_adapt
#



def pi_wasserstein(context_pi: PI, partition: list,
           #regression_per_group
           ):
    """ Wasserstein score for a partition

    :param context_pi: pi class
    :param partition: current partition
    :return: Wasserstein distances
    """
    # Pool data per group ------------
    Xpi, ypi = data_groupby_pi(context_pi.Xc, context_pi.yc, partition)
    wasserstein = [0 for _ in partition]

    # Wasserstein score per group ------------
    for g_k in range(len(partition)):
        group_k = partition[g_k]
        Xk, yk = Xpi[g_k], ypi[g_k]
        score_k = np.inf

        # Use the context-specific GP that has lowest distance to all other contexts in the group ------------
        assert context_pi.GP_c is not None

        for c_i in group_k:
            # Find Wasserstein barycenter
            Xk, yk = Xpi[g_k], ypi[g_k]
            score_i = None # TODO context_pi.GP_c[c_i].mdl_score_ytest(Xk, yk)

            if score_i < score_k:
                score_k = score_i

        wasserstein[g_k] =score_k

    return sum(wasserstein), wasserstein


"""Exhaustive Search over all Partitions """
#
# def pi_structure_best(context_pi, regression_per_group, mdl_gain, subsample_size=None):
#     #Compute MDL score for all partitions and among the best ones return the one with least groups k
#     c_n = context_pi.C_n
#     Pizero = [[c_i] for c_i in range(c_n)]
#     min_atk = [np.inf for _ in range(c_n)]
#     min_allk = np.inf
#     argmin_atk = [Pizero for _ in range(c_n)]
#     argmin_allk = Pizero
#     mdl_dict = {}
#
#     partitions = pi_enum(c_n, permute=True)
#
#     for partition in partitions:
#         k = len(partition)-1
#         mdl, _, mdl_model, mdl_data, mdl_pen = pi_mdl(context_pi, partition, regression_per_group,
#                                                       subsample_size=subsample_size)
#         cur = mdl/(10**11)
#         if mdl_gain:
#             gain_over_0, _, _ = pi_mdl_conf(context_pi, partition, c_n, regression_per_group=regression_per_group,
#                                             subsample_size=subsample_size)
#             cur = -gain_over_0/(10**9)
#
#         mdl_dict[str(partition)] = (mdl, partition)
#         if cur < min_atk[k]:
#             min_atk[k] = cur
#             argmin_atk[k] = partition
#         if cur < min_allk:
#             min_allk = cur
#             argmin_allk = partition
#     alpha = 0.05
#     min_chosen = min_allk
#     argmin_chosen = argmin_allk
#     found_better_k = False
#     for k in range(c_n):
#         diff =min_atk[k] - min_allk
#         if (2 ** (-diff) > alpha):
#             print("\tinsig. difference ", min_allk, min_atk[k], argmin_allk, argmin_atk[k])
#             min_chosen = min_atk[k]
#             argmin_chosen = argmin_atk[k]
#             if not np.isclose(diff, 0):
#                 found_better_k = True
#             break
#
#     print(found_better_k)
#     if min_chosen==np.Inf:
#         print("No min found")
#
#     mdl, _, mdl_model, mdl_data, mdl_pen = pi_mdl(context_pi, argmin_chosen, regression_per_group,
#                                                       subsample_size=subsample_size)
#     mdl_dict_sorted = sorted(mdl_dict.items(), key=lambda item: item[1][0], reverse=False)
#
#     correct_for_pizero=True
#     if correct_for_pizero:
#         pi0, (score0, _) = mdl_dict_sorted[0]
#         pi1, (score1, _) = mdl_dict_sorted[1]
#         is_insig = (2 ** (-(score0 - score1)) > 0.05)
#
#         if is_insig and (len(pi0) == 1):
#             # replace pi0 by pi1
#             argmin_chosen = pi1
#             mdl, _, mdl_model, mdl_data, mdl_pen = pi_mdl(context_pi, pi1, regression_per_group,
#                                                           subsample_size=subsample_size)
#             print("REPLACEMENT", pi0, score0, "by" , pi1, score1)
#         else:
#             print("No Replacement")
#     return argmin_chosen, mdl, mdl_model, mdl_data, mdl_pen, mdl_dict_sorted


def pi_mdl_best(context_pi, vsn):
    return pi_best_at_k(context_pi, vsn.regression_per_group, vsn.mdl_gain, vsn.subsample_size)


def pi_best_at_k(context_pi, regression_per_group, mdl_gain, subsample_size=None):

    """Compute MDL score for most partitions and return the best one

    :param context_pi:  class PI object with GP regression per context  #
    :param regression_per_group: whether regression within each group, otherwise regression only in each single context
    :return: partition, mdl, mdl model, mdl data, mdl penalty, sorted dict(pi, mdl)
    """
    c_n = context_pi.C_n
    Pizero = [[c_i] for c_i in range(c_n)]

    Piall = [[c_i for c_i in range(c_n)]]

    mdl_dict = {}
    min_mdl = np.inf
    min_cur = np.inf
    min_at_k = Piall
    argmin = Pizero
    min_data = 0
    min_pen = 0
    min_model = 0


    for k in range(2,c_n):
        #Keep track of the best partition with a given number of groups
        min_at_nextk = Piall
        mdl_at_nextk = np.inf

        #Generate fewer candidates for the partitions considered at next k (by splitting one group in the previous partition)
        partitions = pi_enum_split(min_at_k)
        for partition in partitions:
            mdl, _, mdl_model, mdl_data, mdl_pen = pi_mdl(context_pi, partition, regression_per_group,
                                                          subsample_size=subsample_size)
            cur = mdl
            if mdl_gain:
                gain_over_0, _, _ = pi_mdl_conf(context_pi, partition, c_n, regression_per_group=regression_per_group,
                                                subsample_size=subsample_size)
                cur = -gain_over_0

            mdl_dict[str(partition)] = (mdl, partition)
            if cur < min_cur:
                min_cur = cur
                min_mdl = mdl
                min_model = mdl_model
                min_data = mdl_data
                min_pen = mdl_pen
                argmin = partition

            if cur < mdl_at_nextk:
                min_at_nextk = partition
                mdl_at_nextk = cur
        min_at_k = min_at_nextk

    if min_mdl==np.Inf:
        print("(LINC pi_search_mdl) No min found")#
        arg_min = [[c_i] for c_i in range(c_n)]
        min_mdl, _, min_model, min_data, min_pen = pi_mdl(context_pi, arg_min, regression_per_group,
                                                      subsample_size=subsample_size)
        pass

    mdl_dict_sorted = sorted(mdl_dict.items(), key=lambda item: item[1][0], reverse=False)
    correct_for_pizero=True
    if correct_for_pizero:
        _, (score0, pi0) = mdl_dict_sorted[0]
        _, (score1, pi1) = mdl_dict_sorted[1]
        is_insig = (2 ** (-(score0 - score1)) > 0.05)

        if is_insig and (len(pi0) == 1):
            # replace pi0 by pi1
            argmin = pi1
            mdl, _, mdl_model, mdl_data, mdl_pen = pi_mdl(context_pi, pi1, regression_per_group,
                                                          subsample_size=subsample_size)
            #print("REPLACEMENT", pi0, round(score0,2), "by" , pi1, round(score1, 2))
        #else:
            #print("No Replacement, 1.", pi0, round(score0, 2),  "2.", pi1, round(score1,2), is_insig, len(pi0))
    return argmin, min_mdl, min_model, min_data, min_pen, mdl_dict_sorted


def pi_overall_best(context_pi, regression_per_group, mdl_gain, subsample_size=None):

    """Compute MDL score for all partitions and return the best one

    :param context_pi:  class PI object with GP regression per context  #
    :param regression_per_group: whether regression within each group, otherwise regression only in each single context
    :return: partition, mdl, mdl model, mdl data, mdl penalty, sorted dict(pi, mdl)
    """
    c_n = context_pi.C_n
    Pizero = [[c_i] for c_i in range(c_n)]

    mdl_dict = {}
    min_mdl = np.inf
    min_cur = np.inf
    argmin = Pizero
    min_data = 0
    min_pen = 0
    min_model = 0

    partitions = pi_enum(c_n, permute=True)

    for partition in partitions:
        mdl, _, mdl_model, mdl_data, mdl_pen = pi_mdl(context_pi, partition, regression_per_group,
                                                      subsample_size=subsample_size)
        cur = mdl
        if mdl_gain:
            gain_over_0, _, _ = pi_mdl_conf(context_pi, partition, c_n, regression_per_group=regression_per_group,
                                            subsample_size=subsample_size)
            cur = -gain_over_0

        mdl_dict[str(partition)] = (mdl, partition)
        if cur < min_cur:
            min_cur = cur
            min_mdl = mdl
            min_model = mdl_model
            min_data = mdl_data
            min_pen = mdl_pen
            argmin = partition

    if min_mdl==np.Inf:
        print("(LINC pi_search_mdl) No min found")#
        arg_min = [[c_i] for c_i in range(c_n)]
        min_mdl, _, min_model, min_data, min_pen = pi_mdl(context_pi, arg_min, regression_per_group,
                                                      subsample_size=subsample_size)
        pass

    mdl_dict_sorted = sorted(mdl_dict.items(), key=lambda item: item[1][0], reverse=False)
    correct_for_pizero=True
    if correct_for_pizero:
        _, (score0, pi0) = mdl_dict_sorted[0]
        _, (score1, pi1) = mdl_dict_sorted[1]
        is_insig = (2 ** (-(score0 - score1)) > 0.05)

        if is_insig and (len(pi0) == 1):
            # replace pi0 by pi1
            argmin = pi1
            mdl, _, mdl_model, mdl_data, mdl_pen = pi_mdl(context_pi, pi1, regression_per_group,
                                                          subsample_size=subsample_size)
            #print("REPLACEMENT", pi0, round(score0,2), "by" , pi1, round(score1, 2))
        #else:
            #print("No Replacement, 1.", pi0, round(score0, 2),  "2.", pi1, round(score1,2), is_insig, len(pi0))
    return argmin, min_mdl, min_model, min_data, min_pen, mdl_dict_sorted


"""Confidence and Significance"""


def pi_mdl_conf(context_pi, pi_guess, C_n, regression_per_group, subsample_size):
    pi_0 = [[c_i] for c_i in range(C_n)]
    mdl_0, _, model_0, data_0, _ = pi_mdl(context_pi, pi_0, regression_per_group, subsample_size)
    mdl_guess, _, model_guess, data_guess, _ = pi_mdl(context_pi, pi_guess, regression_per_group, subsample_size)

    # pi_1 = [[c_i for c_i in range(C_n)]]
    # mdl_1, _, _, _, _= pi_mdl(context_pi, pi_1, regression_per_group)

    gain = mdl_0 - mdl_guess
    gain_model = model_0 - model_guess
    gain_data = data_0 - data_guess
    return gain, gain_model, gain_data


def pi_is_insignificant(context_pi, pi_guess, C_n,regression_per_group, alpha=0.05):
    #if len(pi_guess) == 1 or len(pi_guess) == C_n:
    #    return True
    gain, _, _ = pi_mdl_conf(context_pi, pi_guess, C_n, regression_per_group, subsample_size=None) #TODO
    return (gain < 0 or 2 ** (-gain) > alpha)


"""Normalized Gain """

def pi_normalized_gain(pi_good, pi_bad,
                       context_pi_good, context_pi_bad,
                       regression_per_group,
                       C_n, subsample_size):
    # whether pi_good has a gain over pi_bad (result is positive if yes)

    gain1, model1, data1 = pi_mdl_conf(context_pi_good, pi_good, C_n, regression_per_group, subsample_size)
    gain2, model2, data2 = pi_mdl_conf(context_pi_bad, pi_bad, C_n, regression_per_group, subsample_size)
    # Larger is better above

    # Larger below means pi_good is better than pi_bad
    # Better meaning it more significantly improves over the null model
    return gain1 - gain2, model1 - model2, data1 - data2

