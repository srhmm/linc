import itertools

import numpy as np

from intervention_types import IvType
from function_types import FunctionType
from out import Out
from pi import PI, ContextRegression
from gen_context_data import gen_context_data
from pi_search_ilp import pi_search_ILP
from pi_search_mdl import pi_mdl_best, pi_mdl
from utils import printo, DecisionType, AccuracyType
from utils_pi import pi_matchto_pi, pi_enum, pi_convertfrom_pair, pi_matchto_pi_exact, pi_group_map
import time
import sys
import os

from vsn import Vsn


def test_partition_search(vsn : Vsn, partitions: list, iters_pi, iters_sub,
                       C_n, D_n, node_n,
                       iv_type_target: IvType, iv_type_covariates: IvType, iv_per_node, iid_contexts,
                       fun_type: FunctionType,
                       initial_seed=1,
                       only_vario=False,
                       only_ilp=False
                       ):
    """
    :param vsn: Vsn
    :param partitions: Partition list, from pi_enum(C_n=5), for the relevant target
    :param iters_pi: how many partitions, 1 to len(partitions)
    :param iters_sub: how many DAGs per partition
    :param C_n: number of contexts
    :param node_n: number of DAGs
    :param D_n: number samples/context
    :param iv_type_target: iv type, for the relevant target
    :param iv_type_covariates: iv type, for other variables
    :param iv_per_node: see gen_context_data
    :param iid_contexts: see gen_context_data
    :param fun_type: function type, linear Gauss or nonlinear
    :param initial_seed: random seed to count up from
    :param only_vario: only vario version
    :param only_ilp: this skips the mdl exhaust. search
    :param skip_regression_group: only regression per context, skip the exact solution
    :return: results
    """

    it = -1
    seed = initial_seed
    rst = np.random.RandomState(seed)
    initial_time = time.perf_counter()
    assert 1 <= iters_pi <= len(partitions)

    # Methods tested
    keys = ['Vario', 'Wasserstein_ILP', 'Wasserstein_ILP(exact)', 'MDL_linear', #baselines
            'MDL_GP', 'MDL_GP(exact)', #MDL scores based on regression per context or regression per partition
            'MDL_ILP_gain', 'MDL_ILP_dist', #ILP approximations, each with a pairwise distance
            'guess_together', 'guess_separate'#random: always put all contexts  into the same(a different) group
            ]
    # Dictionary with each decision of each method
    results = init_results(keys)

    # debug
    dict_pi, dict_param, dict_G, dict_Gc = dict(), dict(), dict(), dict()
    dict_scores, dict_model, dict_data, dict_pen = dict(), dict(), dict(), dict()
    dict_pi['truth'] = [0 for _ in range(iters_pi * iters_sub)]
    for key in keys:
        for dic in [dict_pi, dict_data, dict_pen, dict_model, dict_scores]:
            dic[key] = [0 for _ in range(iters_pi * iters_sub)]

    # printing
    file_nm = str(C_n)+'_' + str(D_n)+'_' +str(node_n)+'_' \
                  +str(fun_type)+'_' +str(iv_type_target) +'_gain.' +str(vsn.mdl_gain) +'_rff.' +str(vsn.rff) +  '_pair.' +str(vsn.regression_per_pair)+ '_NumIv.' + str(iv_per_node[0])  + str(iv_per_node[1]) +  '.txt' #different seeds will print to the same file
    log_file_nm = 'partitions/log_' + file_nm
    res_file_nm = 'partitions/res_' + file_nm

    out_log = Out(log_file_nm, vb=True, tofile=True) #can make verbose
    out_res = Out(res_file_nm)

    # Each Partition -----------------
    for p in range(iters_pi):
        pi_star = partitions[p]
        for i in range(iters_sub):
            it = it + 1
            seed = seed + 1
            dict_pi['truth'][it] = pi_star
            out_log.printto('\nIteration', str(it + 1), '/', str(iters_pi * iters_sub))
            print('\nIteration', str(it + 1), '/', str(iters_pi * iters_sub))

            # random DAG and data, random target Y in the DAG has partition Pistar -----------------
            Dc, gdag, gdag_c, target, parents, children, cfd, _, _ = \
                gen_context_data(C_n=C_n, D_n=D_n, node_n=node_n,
                                 seed=seed, fun_type=fun_type,iv_type_target=iv_type_target,
                                 iv_type_covariates= iv_type_covariates, iid_contexts=iid_contexts,
                                 partition_search=True, partition_Y=pi_star, iv_per_node=iv_per_node)

            # Here, ensure the target Y has causal parents pa(Y) in the DAG, as we do partition search for Y | pa(Y)
            # and if confounding, ensure that a confounder exists in the DAG
            # otherwise, skip this DAG
            skip_dag = (len(parents) == 0 or (iv_type_target is IvType.CONFOUNDING and cfd is None))
            trials = 0
            while skip_dag and trials < 1000:
                seed = seed + 1
                trials = trials + 1
                Dc, gdag, gdag_c, target, parents, children, cfd, _, _ = \
                    gen_context_data(C_n=C_n, D_n=D_n, node_n=node_n, seed=seed,
                                     fun_type=fun_type, iv_type_target=iv_type_target,
                                     iv_type_covariates= iv_type_covariates, iid_contexts=iid_contexts,
                                     partition_search=True, partition_Y=pi_star, iv_per_node=iv_per_node)
                skip_dag = (len(parents) == 0 or (iv_type_target is IvType.CONFOUNDING and cfd is None))

            if skip_dag: #prbly never happens, if it does choose another seed
                raise Exception('no parents / confounders, try another seed')

            dict_G[it] = gdag
            dict_Gc[it] = gdag_c


            # Linear regression -----------------
            st = time.perf_counter()
            PAc = np.array([Dc[c_i][:, [p for p in parents]] for c_i in range(C_n)])
            yc = np.array([Dc[c_i][:, target] for c_i in range(C_n)])
            regression_pi_linear = PI(PAc, yc, rst,
                                      skip_regression=True, skip_regression_pairs=True,  # skips nonlinear regression
                                      info_regression=ContextRegression(None, None, None),pi_search=True)

            dict_param[str(it)] = regression_pi_linear.coef_c_noscaling

            # guessing of partitions
            name = 'guess_together'
            pi_guess = [[c_i for c_i in range(C_n)]]
            results = update_results(results, pi_guess, pi_star, name, confidence=1, C_n=C_n)

            name = 'guess_separate'
            pi_guess = [[c_i] for c_i in range(C_n)]
            results = update_results(results, pi_guess, pi_star, name, confidence=1, C_n=C_n)

            # Scoring every partition
            Pis = pi_enum(C_n, True)
            vario_min = -np.inf
            vario_argmin = None
            lin_min = np.inf
            lin_argmin = None

            for p_cur in range(len(Pis)):
                pi_cur = Pis[p_cur]
                vario_cur = regression_pi_linear.cmp_distances_linear(pi_test=pi_cur, emp=True)
                lin_cur = regression_pi_linear.cmp_distances_linear(pi_test=pi_cur, emp=False, linc=True)
                if vario_cur > vario_min:
                    vario_argmin = pi_cur
                if lin_cur < lin_min:
                    lin_argmin = pi_cur
                vario_min = max(vario_min, vario_cur)
                lin_min = min(lin_min, lin_cur)

            # VARIO -----------------
            name = 'Vario'
            pi_guess = vario_argmin
            confidence = vario_min
            results = update_results(results, pi_guess, pi_star, name, confidence, C_n)
            dict_pi[name][it] = pi_guess
            dict_scores[name][it] = vario_min

            print("Vario time:", round(time.perf_counter()-st,5), "sec")

            # LINC linear -----------------
            name = 'MDL_linear'
            pi_guess = lin_argmin
            confidence = lin_min #TODO
            results = update_results(results, pi_guess, pi_star, name, confidence, C_n)
            dict_pi[name][it] = pi_guess
            dict_scores[name][it] = lin_min

            out_log.printto('\t', str(name)+':', pi_guess)
            if only_vario:
                continue

            # Nonlinear Regression -----------------
            start_time = time.perf_counter()
            regression_pi = PI(PAc, yc, rst, skip_regression=False,
                               skip_regression_pairs=not vsn.regression_per_pair, rff=vsn.rff,pi_search=True)
            print("GP fit time:", round(time.perf_counter() - start_time, 5), "sec")
            start_time = time.perf_counter()
            regression_pi.cmp_distances(from_G_ij=vsn.regression_per_pair)

            comb_i = 0
            combs = itertools.combinations(range(C_n), 2)
            map_star = pi_group_map(pi_star, C_n)
            print ("Partition", pi_star)
            for c_i, c_j in combs:
                div = 1 #10 **10
                if map_star[c_i]==map_star[c_j]:
                    print("\t+", c_i, ",", c_j,": \tGain:",
                          round( regression_pi.pair_mdl_gains[comb_i] / div, 3),
                          "\tModel:", [round(x / div, 1) for x in regression_pi.pair_model_gains[comb_i]],
                          "\tData:", [round(x / div, 1) for x in regression_pi.pair_data_gains[comb_i]])
                else:
                    print("\t-", c_i, ",", c_j,": \tGain=",  round(regression_pi.pair_mdl_gains[comb_i]/ div, 3),
                    "\tModel:", [round(x/ div, 1) for x in regression_pi.pair_model_gains[comb_i]],
                    "\tData:",  [round(x/ div, 1) for x in regression_pi.pair_data_gains[comb_i]])
                comb_i = comb_i + 1
            print("GP mdl time:", round(time.perf_counter() - start_time, 5), "sec")

            # MDL -----------------
            if not only_ilp:
                prnt = 'GP (each pi) time:'
                name = 'MDL_GP'
                if vsn.regression_per_group:
                    name = 'MDL_GP(exact)'
                    prnt = 'GP (each group, each pi) time:'

                start_time = time.perf_counter()
                guess_pi, guess_mdl, guess_model, guess_data, guess_pen, debug_dict = \
                    pi_mdl_best(regression_pi, vsn)


                print(prnt, round(time.perf_counter() - start_time, 5), 'sec')
                out_log.printto('\t', str(name)+':', guess_pi)
                confidence = guess_mdl #TODO
                results = update_results(results, guess_pi, pi_star, name, confidence, C_n)
                dict_pi[name][it] = guess_pi
                dict_scores[name][it] = guess_mdl
                dict_model[name][it], dict_data[name][it], dict_pen[name][it] = guess_model, guess_data, guess_pen

            # MDL & ILP -----------------
            name = 'MDL_ILP_gain'

            start_time = time.perf_counter()
            dists = regression_pi.pair_mdl_gains
            #TODO remove
            if vsn.rff:
                shift = .1
            else:
                shift=0
            vars_ij, _, c = pi_search_ILP(dists, C_n, shift=shift, wasserstein=False)

            guess_pi = pi_convertfrom_pair(vars_ij, c, C_n)
            print("ILP & gain time:", round(time.perf_counter() - start_time, 5), "sec")

            out_log.printto('\t', str(name)+':', guess_pi)

            # compute the scores for this guess:
            guess_mdl, _, guess_model, guess_data, guess_pen = \
                pi_mdl(regression_pi, guess_pi, regression_per_group=vsn.regression_per_group)

            confidence = guess_mdl #TODO
            results = update_results(results, guess_pi, pi_star, name, confidence, C_n)

            dict_scores[name][it] = guess_mdl
            dict_model[name][it], dict_data[name][it], dict_pen[name][it] = guess_model, guess_data, guess_pen


            # MDL & ILP 2-----------------
            name = 'MDL_ILP_dist'

            start_time = time.perf_counter()
            #blockPrint()
            dists = regression_pi.pair_mdl_dists
            vars_ij, _, c = pi_search_ILP(dists, C_n, #shift=.2,
                                          wasserstein=True)
            guess_pi = pi_convertfrom_pair(vars_ij, c, C_n)
            #enablePrint()
            print("ILP & dist time:", round(time.perf_counter() - start_time, 5), "sec")
            out_log.printto('\t', str(name) + ':', guess_pi)

            guess_mdl, _, guess_model, guess_data, guess_pen = \
                pi_mdl(regression_pi, guess_pi, regression_per_group=vsn.regression_per_group)
            confidence = guess_mdl #TODO
            results = update_results(results, guess_pi, pi_star, name, confidence, C_n)

            dict_scores[name][it] = guess_mdl
            dict_model[name][it], dict_data[name][it], dict_pen[name][it] = guess_model, guess_data, guess_pen

            # ILP & Wasserstein distance -----------------
            if vsn.regression_per_pair:
                name = 'Wasserstein_ILP(exact)'
                pair_distances_plain = regression_pi.pair_wass_jointGP
            else:
                name = 'Wasserstein_ILP'
                pair_distances_plain = regression_pi.pair_wass_GP

            #blockPrint()
            start_time = time.perf_counter()
            vars_ij, _, c = pi_search_ILP(pair_distances_plain, C_n, shift=.14, wasserstein=True)
            guess_pi = pi_convertfrom_pair(vars_ij, c, C_n)
            confidence = 1
            results = update_results(results, guess_pi, pi_star, name, confidence, C_n)
            dict_pi[name][it] = guess_pi
            #enablePrint()
            print("ILP & Wasserstein time:", round(time.perf_counter() - start_time, 5), "sec")
            out_log.printto('\t', str(name)+':', guess_pi)

        # Evaluation -----------------
        out_res.printto('\n----------\n[NUM] Contexts:', C_n,'Samples:', D_n, 'Nodes:', node_n, 'Seed:', initial_seed, "Iters:", iters_pi ,
                        '\n[VSN] Regr/Group (GP):', (vsn.regression_per_group),
                        '\n[VSN] Regr/Pair (ILP):', (vsn.regression_per_pair),
                        #'\n[VSN] MDL conf over M0 (GP):', vsn.mdl_gain,
                        '\n[TYPES] FunType:', fun_type, '\n[TYPES] IntervType:', iv_type_target, 'Intervs:', iv_per_node[0], "up to", iv_per_node[1],
               '\n----------\nAssignment context pairs -> groups:')

    for key in results['decisions_pair']:
        if key == 'Wasserstein_ILP(exact)':
            continue
        out_log.printto(key)
        tp, tn, fp, fn = 0,0,0,0
        dic = results['decisions_pair'][key]

        if key == 'Vario':
            dic = sorted(dic.items(), key=lambda item: item[1][0], reverse=True)
        else:
            dic = sorted(dic.items(), key=lambda item: item[1][0])
        for k in dic:
            if k[1][1] is DecisionType.TP:
                tp = tp + 1
            else:
                if k[1][1] is DecisionType.TN:
                    tn = tn + 1
                else:
                    if k[1][1] is DecisionType.FP:
                        fp = fp + 1
                    else:
                        fn = fn + 1

        out_res.printto("\t", key, ":",  "F1: ", round(f1(tp, fp, fn),2),
                        "(", tp, ",", fp,  ",", tn, ",", fn, ")")

    out_res.printto('\nAssignment context -> obs/interv group:')
    for key in results['decisions_each']:

        #if key == 'Wasserstein_ILP(exact)':
        #    continue

        out_log.printto(key)
        acc, nacc = 0,0
        dic = results['decisions_each'][key]
        if key == 'Vario':
            dic = sorted(dic.items(), key=lambda item: item[1][0], reverse=True)
        else:
            dic = sorted(dic.items(), key=lambda item: item[1][0])
        for k in dic:
            if k[1][1] is AccuracyType.T:
                acc = acc + 1
            else:
                nacc = nacc + 1
            out_log.printto('\t', str(k[0]) + '.', k[1][1], "\t", round(k[1][0],2), "\tC",k[1][4],"in", k[1][3],
                   "vs. true" , k[1][2])
        out_res.printto("\t", key, ":", acc, "/", nacc+acc)
    out_res.printto('\nExact partition matching: ')
    for key in results['decisions_exact']:
        t = 0
        f = 0
        lens = [0 for _ in range(C_n)]
        for k in results['decisions_exact'][key]:
            if results['decisions_exact'][key][k][1] :
                t = t + 1
            else:
                f = f + 1
            cur_len = results['decisions_exact'][key][k][2]
            lens[cur_len-1] = lens[cur_len-1] +1
        out_res.printto("\t", str(key) + ":", t , "/", t+f , "\tpreds:", lens )



    out_res.printto("\nTime:", round(time.perf_counter() - initial_time, 5), "seconds")
    out_res.printto("\nAt:", time.asctime())
    out_log.close()
    out_res.close()
    return results

'''
    def print_res(resx, name, iters):
        printo(name, '\nTP: %s TN: %s \nFP: %s FN: %s' % (resx[0], resx[3], resx[1], resx[2]), f_name=result_file_nm)
        printo('F1:', #round(resx[4], 2))  #
               round(f1(resx[0], resx[1], resx[2]), 2), f_name=result_file_nm)
        printo('ARI:',  round(resx[4] /iters , 2), 'AMI:', round(resx[5]/iters,  2), f_name=result_file_nm)


    for key in results['counts_exact']:
        if key == 'Wasserstein_ILP(exact)' or key == 'MDL_GP(exact)':
            continue
        printo("\n", f_name=result_file_nm)
        printo(key, '\nMatch/Miss: %s / %s' % (results['counts_exact'][key][0], results['counts_exact'][key][1]), '\n', f_name=result_file_nm)
        print_res( results['counts_pair'][key], key, it)

    # {'pi' : dict_pi, 'L': dict_scores, 'L_model' : dict_model, 'L_data': dict_data, 'L_pen': dict_pen,
    #    'G' : dict_G,'Gc' : dict_Gc, 'linear_param':  dict_param}
'''
import contextlib
import io
import sys

# @contextlib.contextmanager
# def nostdout():
#     save_stdout = sys.stdout
#     save_stderr = sys.stderr
#     #sys.stdout = io.BytesIO()
#     sys.stderr = io.BytesIO()
#     yield
#     #sys.stdout = save_stdout
#     sys.stderr = save_stderr
#
# class DummyFile(object):
#     def write(self, x): pass
#
# @contextlib.contextmanager
# def nostdout():
#     save_stdout = sys.stdout
#     sys.stdout = DummyFile()
#     yield
#     sys.stdout = save_stdout
#
#
# # Disable
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')
#
#
# # Restore
# def enablePrint():
#     sys.stdout = sys.__stdout__


def f1(tp, fp, fn):
    if (tp + 0.5 * (fp + fn)) == 0:
        return 0
    return tp / (tp + 0.5 * (fp + fn))


def init_results(keys):
    # Each decision of each method (key), where a decision is either
    # a context pair assigned correctly to the same/to a different group, or
    # a single context assigned correctly to the observational/the interventional group.
    results = dict()
    results['decisions_each'] = dict()
    results['decisions_pair'] = dict()
    results['decisions_exact'] = dict()
    for key in keys:
        results['decisions_each'][key] = dict()
        results['decisions_pair'][key] = dict()
        results['decisions_exact'][key] = dict()
    return (results)


def update_results(results, pi_guess, pi_star, name, confidence, C_n):

    #pairw = pi_matchto_pi(pi_star, pi_guess, C_n)  # returns tp, fp, fn, tn, ari, ami
    #exact = pi_matchto_pi_exact(pi_star, pi_guess)  # returns match, nomatch

    map_star = pi_group_map(pi_star, C_n)
    map_guess = pi_group_map(pi_guess, C_n)
    assert (len(map_guess) == len(map_star))
    assert (C_n == len(map_guess))

    #TODO cleaner way??

    groups = range(len(pi_guess))
    which_contexts_overlap_0 =[[i for i in range(C_n) if map_guess[i] == g and map_star[i]==0] for g in groups]
    which_contexts_overlap_1 =[[i for i in range(C_n) if map_guess[i] == g and map_star[i]==1] for g in groups]

    obs_group = max(range(len(which_contexts_overlap_0)), key=which_contexts_overlap_0.__getitem__)
    remaining = [(ind,w) for (ind, w) in enumerate(which_contexts_overlap_1) if not (ind==obs_group)]
    interv_group = -1
    if len(remaining) > 0:
        interv_index = max(range(len(remaining)), key=lambda x:remaining.__getitem__(x)[1])
        interv_group = remaining[interv_index][0]
    if obs_group == interv_group:
        interv_group = -1
    overlap = len(which_contexts_overlap_0[obs_group]) + len(which_contexts_overlap_1[interv_group])

    #Special case: the interv group would have been the better nth group but obs group has been chosen because  it was considered first
    interv_group_alt = obs_group
    remaining = [(ind, w) for (ind, w) in enumerate(which_contexts_overlap_0) if not (ind == obs_group)]
    obs_group_alt = -1

    if len(remaining) > 0:
        index = max(range(len(remaining)), key=lambda x: remaining.__getitem__(x)[1])
        obs_group_alt = remaining[index][0]
    if obs_group_alt == interv_group_alt:
        obs_group_alt = -1
    overlap_alt = len(which_contexts_overlap_0[obs_group_alt]) + len(which_contexts_overlap_1[interv_group_alt])
    if overlap_alt > overlap:
        obs_group = obs_group_alt
        interv_group = interv_group_alt

    # obs_01 = obs_group
    # interv_01 = interv_group
    #
    #
    # groups = range(len(pi_guess))
    # which_contexts_overlap_0 = [[i for i in range(C_n) if map_guess[i] == g and map_star[i] == 1] for g in groups]
    # which_contexts_overlap_1 = [[i for i in range(C_n) if map_guess[i] == g and map_star[i] == 0] for g in groups]
    #
    # obs_group = max(range(len(which_contexts_overlap_0)), key=which_contexts_overlap_0.__getitem__)
    # remaining = [(ind, w) for (ind, w) in enumerate(which_contexts_overlap_1) if not (ind == obs_group)]
    # interv_group = -1
    # if len(remaining) > 0:
    #     interv_index = max(range(len(remaining)), key=lambda x: remaining.__getitem__(x)[1])
    #     interv_group = remaining[interv_index][0]
    # if obs_group == interv_group:
    #     interv_group = -1
    #
    # obs_10 = obs_group
    # interv_10 = interv_group
    # overlap_10 = len(which_contexts_overlap_0[obs_10]) + len(which_contexts_overlap_1[interv_10])
    #
    # if overlap_01 > overlap_10:
    #     obs_group = obs_01
    #     interv_group = interv_01
    match = True
    for ci in range(C_n):
        if not (map_guess[ci] == map_star[ci]):
            match=False
    index = 0
    if len(results['decisions_exact'][name]):
        index = max(results['decisions_exact'][name]) + 1
    results['decisions_exact'][name][index] =  (confidence, match, len(pi_guess), pi_star, pi_guess)

    for ci in range(C_n):
        index = 0
        if len(results['decisions_each'][name]) :
            index = max(results['decisions_each'][name]) + 1
        if (map_guess[ci] == 0 and map_star[ci] == obs_group) \
            or (map_guess[ci] == 1 and map_star[ci] == interv_group):
            results['decisions_each'][name][index] = (confidence, AccuracyType.T, pi_star, pi_guess, ci)
        else:
            results['decisions_each'][name][index] = (confidence, AccuracyType.F, pi_star, pi_guess, ci)

    for i, j in itertools.combinations(range(C_n), 2):
        shared_star = map_star[i] == map_star[j]
        shared_guess = map_guess[i] == map_guess[j]

        index = 0
        if len(results['decisions_pair'][name]) :
            index = max(results['decisions_pair'][name]) + 1
        if shared_guess:
            if shared_star:
                results['decisions_pair'][name][index] = (confidence, DecisionType.TP, pi_star, pi_guess, i, j)
            else:
                results['decisions_pair'][name][index] = (confidence, DecisionType.FP, pi_star, pi_guess, i, j)
        else:
            if shared_star:
                results['decisions_pair'][name][index] = (confidence, DecisionType.FN, pi_star, pi_guess, i, j)
            else:
                results['decisions_pair'][name][index] = (confidence, DecisionType.TN, pi_star, pi_guess, i, j)

    return results





def init_results_old(keys, C_n):
    results = dict()
    results['decisions_each'] = dict()
    results['decisions_pair'] = dict()
    # for debug
    #results['counts_pair'], results['counts_exact'], results['predicted_k'] = dict(), dict(), dict()
    for key in keys:
        results['decisions_each'][key] = dict() # key:instance, entry: ( confidence, decision (whether TP,FP,TN,FN))
        results['decisions_pair'][key] = dict()
        #results['predicted_k'][key] = dict()
        #for k in range(C_n + 1):
        #    results['predicted_k'][key][str(k)] = 0
        #results['predicted_k'][key]["g*"] = 0
        #results['counts_exact'][key] = [0, 0]  # matches, mismatches
        #results['counts_pair'][key] = [0, 0, 0, 0, 0, 0, 0]  # tp fp fn tn ari ami count
    return (results)

def update_results_old(results, pi_guess, pi_star, name, C_n):
    pairw = pi_matchto_pi(pi_star, pi_guess, C_n)  # returns tp, fp, fn, tn, ari, ami
    exact = pi_matchto_pi_exact(pi_star, pi_guess)  # returns match, nomatch
    for j in range(len(pairw)):
        results['counts_pair'][name][j] = results['counts_pair'][name][j] + pairw[j]
    results['counts_pair'][name][6] = results['counts_pair'][name][6] + 1
    for j in range(len(exact)):
        results['counts_exact'][name][j] = results['counts_exact'][name][j] + exact[j]
    if exact[0] == 1:
        results['predicted_k'][name]["g*"] = results['predicted_k'][name]["g*"] + 1
    else:
        results['predicted_k'][name][str(len(pi_guess))] = results['predicted_k'][name][str(len(pi_guess))] + 1
    return results
