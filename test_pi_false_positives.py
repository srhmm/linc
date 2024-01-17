import numpy as np

from function_types import FunctionType
from intervention_types import IvType
from out import Out
from pi import PI, ContextRegression
from gen_context_data import gen_context_data
from pi_search_ilp import pi_search_ILP
from pi_search_mdl import pi_mdl_best, pi_is_insignificant, pi_mdl, pi_mdl_conf, pi_normalized_gain
from test_pi_search import f1
from utils import printo
from statistics import mean, stdev
from utils_pi import pi_matchto_pair, pi_matchto_pi, pi_enum, pi_group_map, pi_convertfrom_pair, \
    pi_matchto_pi_exact, pi_valid
import time

from vsn import Vsn


def test_partition_fps(vsn: Vsn, iters=10,
                       anticausal=True,
                       only_vario=False,
                       initial_seed=1,
                       C_n=5, node_n=10,
                       D_n=500,
                       fun_type = FunctionType.LINEAR_GAUSS,
                       iv_type_covariates = IvType.PARAM_CHANGE,
                       iv_type_target = IvType.PARAM_CHANGE,
                       Pistar = [[0,1],[2,3,4]],
                       iv_per_node=[4,4],
                       iid_contexts=False,
                       extensive_result=False,
                       only_ilp=False,
                       n_parent_child=2
                       ):
    """
            Counts partition false positives -- how often a spurious partition is reported for a node Y given NONcausal variables.
            Randomly generates a DAG G (iters many) and picks Y,
            generates data for Y from its parents X in G, does GP regression for X->Y.
            The returned partition for (X, Y), X not Xpa, is matched with the ground truth,
            which is the partition without groups.
            VERSIONS: 1. Wasserstein Distances (of GPs in context pairs) + ILP,
            2. MDL gains (of GPs in context pairs) + ILP, 3. exhaustive search with MDL score,
            4. Vario with exhaustive search and linear models.

            Parameters
            ----------
            iters:
                how many runs
            initial_seed:
                random seed to count up from
            C_n:
                number of contexts
            node_n:
                number of DAGs
            iv_per_node:
                and other iv... arguments: interventions, as in gen_dag_nonlinear
            joint:
                if yes, GP regression for each context pair, on the pooled data
                and we compare this group model to the individual models;
                if no, we only compare the individual models (more efficient)
            """
    #if subsample_size is None:
    subsample_size = D_n
    if iters < 2:
        raise Exception("at least two iterations")
    it = -1
    seed = initial_seed
    initial_time = time.perf_counter()
    rst = np.random.RandomState(seed)

    file_nm = str(C_n)+'_' + str(D_n)+'_' +str(node_n)+'_' \
                  +str(fun_type)+'_' +str(iv_type_target) + \
              '_gain.' +str(vsn.mdl_gain) +'_rff.' +str(vsn.rff) +  '_pair.' +str(vsn.regression_per_pair)+\
              '_NumIv.' + str(iv_per_node[0])  + str(iv_per_node[1]) + '.txt' #different seeds will print to the same file
    log_file_nm = 'asymmetry/log_' + file_nm
    res_file_nm = 'asymmetry/res_' + file_nm
    out_log = Out(log_file_nm) #, vb=False, tofile=False) #can make verbose
    out_res = Out(res_file_nm)

    res, res1, res0, res_finegr, res_coarse, res_eq = dict(), dict(), dict(), dict(), dict(), dict()
    keys = ['Vario', 'Wasserstein-ILP', 'MDL-linear', # Baselines
           'MDL-nonlinear', 'MDL-nonlinear-ILP' ]  #ours

    counts = 0
    for key in keys:
        res[key] = [0, 0, 0, 0, 0, 0]
        res1[key], res0[key], res_finegr[key], res_coarse[key], res_eq[key] = 0,0,0,0,0
        #all decisions; one group; no group; more finegrained; same as in the causal dir.


    #scores
    scores_pa_z, scores_pa_ch, scores_model, scores_data, \
    scores_conf_ch_pa,scores_conf_model,scores_conf_data, scores_conf_z_pa, =\
        dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
    #test
    scoco, scoco_model, scoco_data, scoc, scoc_model, scoc_data = dict(), dict(), dict(), dict(), dict(), dict()
    #scores if imposing the one grp/no grp model
    scoo, scoo_model, scoo_data, scon, scon_model, scon_data = dict(), dict(), dict(), dict(), dict(), dict()

    dictionaries = [scores_pa_z, scores_pa_ch, scores_model, scores_data,
                    scores_conf_ch_pa,scores_conf_model,scores_conf_data, scores_conf_z_pa,
                    scoco, scoco_model, scoco_data, scoc, scoc_model, scoc_data,
                    scoo, scoo_model, scoo_data, scon, scon_model, scon_data
                    ]
    for key in ['Vario', 'MDL-nonlinear', 'MDL-nonlinear-ILP']:
        for dic in dictionaries:
            dic[key] = np.zeros((iters, 3))

    # Each Partition -----------------
    out_res.printto('\n----------\n[NUM] Contexts:', C_n,'Samples:', D_n, 'Nodes:', node_n, 'Seed:', initial_seed, "Iters:", iters ,   '|Pa|=|Ch|=', n_parent_child,
                        '\n[VSN] RFF:', vsn.rff,
                        '\n[VSN] MDL conf over M0 (GP):', vsn.mdl_gain,
                        '\n[TYPES] FunType:', fun_type, '\n[TYPES] IntervType (Cov):', iv_type_covariates, 'IntervType (Tgt):', iv_type_target,
                    'Intervs (Cov):', iv_per_node[0], "up to", iv_per_node[1],  'Intervs (Tgt):', len(Pistar)-1)


    for i in range(iters):
        Pizero = [[c_i] for c_i in range(C_n)]
        Pione = [[c_i for c_i in range(C_n)]]

        for _ in range(1): #indent, can remove
            it = it + 1
            seed = seed + 1
            print("\nIteration", str(it + 1) + "/" + str(iters))

            # Data for a DAG, random target has partition Pistar -----------------
            st = time.perf_counter()
            Dc, gdag, _, target, parents, children, _, _, _ = \
                gen_context_data(C_n=C_n, D_n=D_n, node_n=node_n,
                                 partition_search=True, partition_Y=Pistar,
                                 iv_type_target=iv_type_target,
                                 iv_type_covariates=iv_type_covariates,
                                 fun_type=fun_type,
                                 scale=True, seed=seed,
                                 iid_contexts=iid_contexts, iv_per_node=iv_per_node)

            noncausal = [n for n in gdag.nodes if (not(n==target)) and n not in parents and n not in children]
            # Here, ensure that children/noncausal variables exist and have the desired sizes
            trials = 0
            while (anticausal and len(children) == 0) \
                    or (not anticausal and len(noncausal) == 0) \
                    or (anticausal and ((len(parents) < n_parent_child) or (len(children) < n_parent_child))) \
                    or (not anticausal and ((len(parents) < n_parent_child) or (len(noncausal) < n_parent_child))) \
                    and trials < 10000: #termination
                seed = seed + 1
                trials = trials + 1
                Dc, gdag, _, target, parents, children, _, _, _ = \
                    gen_context_data(C_n=C_n, D_n=D_n, node_n=node_n,
                                     partition_search=True, partition_Y=Pistar,
                                     iv_type_target=iv_type_target,
                                     iv_type_covariates=iv_type_covariates, fun_type=fun_type,
                                     scale=True, seed=seed,
                                     iid_contexts=iid_contexts, iv_per_node=iv_per_node)
                noncausal = [n for n in gdag.nodes if (not(n==target)) and n not in parents and n not in children]

            print("Data gen time: ", time.perf_counter() - st)
            if (anticausal and len(children) == 0) or (not anticausal and len(noncausal) == 0) \
                    or (len(parents) < n_parent_child) or (len(noncausal) < n_parent_child) :
                raise Exception('no children/noncausal vars for this seed')
            if anticausal:
                variableset = children
            else:
                variableset = noncausal
            if len(variableset)>len(parents):
                variableset = [i for i in variableset]
                variableset = variableset[0:len(parents)]

            yc = np.array([Dc[c_i][:, target] for c_i in range(C_n)])

            parent = [p for p in parents][0]
            child = [c for c in variableset][0]

            if n_parent_child is not None: # to exclude the possibility/test whether sizes of parent and child set matter / make scores incomparable
                parents = [p for p in parents][:n_parent_child]
                children = [p for p in variableset][:n_parent_child]
            baseset = [p for p in parents if not p == parent]
            Xc = np.array([Dc[c_i][:, [p for p in gdag.nodes if p in baseset or p==child]] for c_i in range(C_n)])
            PAc = np.array([Dc[c_i][:, [p for p in parents]] for c_i in range(C_n)])
            #Bc = np.array([Dc[c_i][:, [p for p in baseset]] for c_i in range(C_n)])
            CHc = np.array([Dc[c_i][:, [p for p in children]] for c_i in range(C_n)])

            counts = counts + 1
            # VARIO -----------------
            vario_min_set, vario_min_pa, vario_min_ch, vario_min_base = -np.inf,-np.inf, -np.inf, -np.inf
            vario_argmin_pa, vario_argmin_set, vario_argmin_ch, vario_argmin_base = None, None, None, None

            regression_set = PI(Xc, yc, rst, skip_regression=True, # this skips the nonlinear regression only
                                      skip_regression_pairs=not vsn.regression_per_pair,
                                      info_regression=ContextRegression(None, None, None))

            regression_pa = PI(PAc, yc, rst, skip_regression=True,
                                      skip_regression_pairs=not vsn.regression_per_pair,
                                      info_regression=ContextRegression(None, None, None))

            regression_ch = PI(CHc, yc, rst, skip_regression=True,
                                      skip_regression_pairs=not vsn.regression_per_pair,
                                      info_regression=ContextRegression(None, None, None))

            Pis = pi_enum(C_n, True)
            for p_cand in range(len(Pis)):
                vario_pa = regression_pa.cmp_distances_linear(pi_test=Pis[p_cand], emp=True)
                if vario_pa > vario_min_pa:
                    vario_argmin_pa = Pis[p_cand]
                vario_min_pa = max(vario_min_pa, vario_pa)

                vario_set = regression_set.cmp_distances_linear(pi_test=Pis[p_cand], emp=True)
                if vario_set > vario_min_set:
                    vario_argmin_set = Pis[p_cand]
                vario_min_set = max(vario_min_set, vario_set)

                vario_ch = regression_ch.cmp_distances_linear(pi_test=Pis[p_cand], emp=True)
                if vario_ch > vario_min_ch:
                    vario_argmin_ch = Pis[p_cand]
                vario_min_ch = max(vario_min_ch, vario_ch)

            out_log.printto('\tVARIABLES:\tY=', target,
                   "\tpa(Y)=", [p for p in parents],#"=" , [p for p in gdag.nodes if p in baseset or p==parent],
                   "\tch(Y)=", [p for p in children],
                   "\tZ=", [p for p in gdag.nodes if p in baseset or p==child], "\n")

            out_log.printto('\tVARIO:\tZ->Y:', round(vario_min_set,2), vario_argmin_set,
                   "\n\t\tpa(Y)->Y:", round(vario_min_pa, 2), vario_argmin_pa,
                  # "\n\t\tB(Y)->Y:", round(vario_min_base, 2), vario_argmin_base,
                   "\n\t\tch(Y)->Y:", round(vario_min_ch, 2), vario_argmin_ch,
                   "\n\t\tG(pa,Z):", round(vario_min_pa - vario_min_set,2),
                   "\n\t\tG(pa,ch):", round(vario_min_pa - vario_min_ch,2))
                 #  "\n\t\tG(pa-B, Z-B):", round((vario_min_pa-vario_min_base) - (vario_min_set-vario_min_base) ,2),

            scores_pa_z['Vario'][i][0], scores_pa_z['Vario'][i][1], scores_pa_z['Vario'][i][2],  = \
                vario_min_pa, vario_min_set, vario_min_pa - vario_min_set
            scores_pa_ch['Vario'][i][0], scores_pa_ch['Vario'][i][1], scores_pa_ch['Vario'][i][2],  = \
                vario_min_pa, vario_min_ch, vario_min_pa - vario_min_ch

            # Eval the partition found for Children(Y) -> Y
            res, res1, res0, res_finegr, res_coarse, res_eq = eval_partition('Vario', Piguess = vario_argmin_ch,
                                                                     Piparents= Pistar, #TODO vario_argmin_pa,
                                                                     C_n=C_n, res=res,
                                                                     res1=res1, res0=res0, res_finegr=res_finegr,
                                                                     res_coarse=res_coarse, res_eq=res_eq)

            if only_vario:
                continue

            # Nonlinear Regression -----------------

            regression_pa = PI(PAc, yc, rst, skip_regression=False, skip_regression_pairs=not vsn.regression_per_pair, rff=vsn.rff)
            regression_pa.cmp_distances(from_G_ij=vsn.regression_per_pair )

            regression_set = PI(Xc, yc, rst, skip_regression=False, skip_regression_pairs=not vsn.regression_per_pair, rff=vsn.rff)
            regression_set.cmp_distances(from_G_ij=vsn.regression_per_pair )

            regression_ch = PI(CHc, yc, rst, skip_regression=False, skip_regression_pairs=not vsn.regression_per_pair, rff=vsn.rff)
            regression_ch.cmp_distances(from_G_ij=vsn.regression_per_pair )

            # MDL -----------------
            if not only_ilp:
                st = time.perf_counter()
                guess_pi, guess_mdl, guess_model, guess_data, guess_pen, _ = pi_mdl_best(regression_set, vsn)
                guess_pa, guess_mdl_pa, guess_model_pa,guess_data_pa, _, _ = pi_mdl_best(regression_pa, vsn)
                guess_ch, guess_mdl_ch, guess_model_ch, guess_data_ch, _, _ = pi_mdl_best(regression_ch, vsn)

                print("GP time:" , round(time.perf_counter() - st, 5), "sec")
                st = time.perf_counter()

                if (extensive_result):
                    #Evaluating the MDL scores under different partitions
                    scoo_ch, _,  scoo_model_ch, scoo_data_ch, _ = pi_mdl(regression_ch, Pione,  False, subsample_size)
                    scoo_pa, _, scoo_model_pa, scoo_data_pa, _ = pi_mdl(regression_pa, Pione, False, subsample_size)
                    scon_ch, _, scon_model_ch, scon_data_ch, _ = pi_mdl(regression_ch, Pizero,  False, subsample_size)
                    scon_pa, _, scon_model_pa, scon_data_pa, _ = pi_mdl(regression_pa, Pizero, False, subsample_size)

                    #repurposed scoc
                   # scoc_ch, _, scoc_model_ch, scoc_data_ch, _ = pi_mdl(regression_ch, guess_pa, False, subsample_size)
                   # scoc_pa, _, scoc_model_pa, scoc_data_pa, _ = pi_mdl(regression_pa, guess_pa, False, subsample_size)

                    # Correcting model scores for one-group model
                    #TODO revisit these as check_one_group had bug
                    #scoco_ch, scoco_model_ch, scoco_data_ch = guess_ch, guess_model_ch, guess_data_ch
                    #scoco_pa, scoco_model_pa, scoco_data_pa = guess_pa, guess_model_pa, guess_data_pa


                    conf_pa, como_pa, coda_pa  = pi_mdl_conf(regression_pa, guess_pa, C_n,  False, subsample_size)
                    conf_ch, como_ch, coda_ch = pi_mdl_conf(regression_ch, guess_ch, C_n,  False, subsample_size)
                    normalized_gain, gain_model, gain_data = pi_normalized_gain(guess_pa, guess_ch, regression_pa, regression_ch,False, C_n, subsample_size)
                    # means normalized_gain= conf_pa - conf_ch
                    #conf_z, _, _ = pi_mdl_conf(regression_set, guess_pi, C_n, False)

                    # Correcting conf. scores for one-group model
                    scoco_ch, scoco_model_ch, scoco_data_ch = conf_ch, como_ch, coda_ch
                    scoco_pa, scoco_model_pa, scoco_data_pa = conf_pa, como_pa, coda_pa

                    scoc_ch, scoc_model_ch, scoc_data_ch  = guess_mdl_ch, guess_model_ch, guess_data_ch
                    scoc_pa, scoc_model_pa, scoc_data_pa = guess_mdl_pa, guess_model_pa, guess_data_pa

                    if is_one_group(guess_ch, C_n):
                        #This will result in zero gain
                        scoco_ch, scoco_model_ch, scoco_data_ch = pi_mdl_conf(regression_ch, Pizero, C_n,  False, subsample_size)
                        scoc_ch, _,  scoc_model_ch, scoc_data_ch, _ = pi_mdl(regression_ch, Pizero, False, subsample_size)
                    if is_one_group(guess_pa, C_n):
                        scoco_pa, scoco_model_pa, scoco_data_pa = pi_mdl_conf(regression_pa, Pizero, C_n,  False, subsample_size)
                        scoc_pa, _,  scoc_model_pa, scoc_data_pa , _ = pi_mdl(regression_pa, Pizero, False, subsample_size)


                if pi_is_insignificant(regression_ch, guess_ch, C_n, regression_per_group=False):
                   ins = "(insig)"
                else:
                    ins  = ""
                out_log.printto('\tMDL \tZ->Y:', round(guess_mdl, 2), "\t",   guess_pi,"\tModel: ", round(guess_model,2),
                       "\n\t\tpa(Y)->Y:",  round(guess_mdl_pa,2),"\t", guess_pa, "\tModel: ", round(guess_model_pa,2),
                       #"\n\t\tB(Y)->Y:",  round(guess_mdl_base,2),  "\t", guess_base,
                       "\n\t\tch(Y)->Y:",  round(guess_mdl_ch,2), ins, "\t", guess_ch,"\tModel: ", round(guess_model_ch,2),
                       guess_ch)

                out_log.printto('\t\tG(pa,Z):', round(guess_mdl-guess_mdl_pa, 2)) #gain parents over Z
                       #'\n\t\tG(pa_conf, Z_conf):', round(conf_pa - conf_z, 2),#conf Z over pa
                if extensive_result:
                    out_log.printto('\n\t\tG(pa_conf,ch_conf) / model / data:', round(normalized_gain, 2), "/", round(gain_model, 2), "/", round(gain_data, 2),
                       '\n\t\tGconf(corrected):', round(scoco_pa - scoco_ch, 2),
                       '\n\t\tG / GMO / GDA (pa,ch):', round(guess_mdl_ch - guess_mdl_pa, 2), "/",  round(guess_model_ch - guess_model_pa, 2), "/",  round(guess_data_ch - guess_data_pa, 2),
                       '\n\t\tG(Pin):', round(scon_ch - scon_pa, 2),
                       '\n\t\tG(Pione):', round(scoo_ch - scoo_pa, 2),
                       #'\n\t\tG(Picausal):', round(scoc_ch - scoc_pa, 2),
                       #'\n\t\tGMO(corrected):', round(scoco_model_ch - scoco_model_pa, 2),
                       '\n\t\tGMO(Pin):', round(scon_model_ch - scon_model_pa, 2),
                       '\n\t\tGMO(Pione):', round(scoo_model_ch - scoo_model_pa, 2),
                       '\n\t\tGMO(Picausal):', round(scoc_model_ch - scoc_model_pa, 2),
                       #'\n\t\tGDA(corrected):', round(scoco_data_ch - scoco_data_pa, 2),
                       '\n\t\tGDA(Pin):', round(scon_data_ch - scon_data_pa, 2),
                       '\n\t\tGDA(Pione):', round(scoo_data_ch - scoo_data_pa, 2),
                       '\n\t\tGDA(Picausal):', round(scoc_data_ch - scoc_data_pa, 2))
                       #'\n\t\tG(pa-B,Z-B):', round((guess_mdl_base - guess_mdl_pa) - (guess_mdl_base - guess_mdl), 2),

                # ordinary scores
                scores_pa_z['MDL-nonlinear'][i][0], scores_pa_z['MDL-nonlinear'][i][1], scores_pa_z['MDL-nonlinear'][i][2] \
                    = guess_mdl_pa, guess_mdl, guess_mdl - guess_mdl_pa
                scores_pa_ch['MDL-nonlinear'][i][0], scores_pa_ch['MDL-nonlinear'][i][1], scores_pa_ch['MDL-nonlinear'][i][2]\
                    = guess_mdl_pa, guess_mdl_ch, guess_mdl_ch - guess_mdl_pa
                scores_model['MDL-nonlinear'][i][0], scores_model['MDL-nonlinear'][i][1], scores_model['MDL-nonlinear'][i][2]\
                    = guess_model_pa, guess_model_ch, guess_model_ch - guess_model_pa
                scores_data['MDL-nonlinear'][i][0], scores_data['MDL-nonlinear'][i][1], scores_data['MDL-nonlinear'][i][2]\
                    = guess_data_pa, guess_data_ch, guess_data_ch - guess_data_pa

                #scores_pa_z, scores_pa_ch, scores_conf_ch_pa, scores_conf_z_pa, scores_model, scores_data = dict(), dict(), dict(), dict(), dict(), dict()
                #scoco, scoco_model, scoco_data, scoc, scoc_model, scoc_data = dict(), dict(), dict(), dict(), dict(), dict()
                #scoo, scoo_model, scoo_data, scon, scon_model, scon_data = dict(), dict(), dict(), dict(), dict(), dict()
                # rest
                if extensive_result:
                    scoco['MDL-nonlinear'][i][0], scoco['MDL-nonlinear'][i][1], scoco['MDL-nonlinear'][i][2]\
                        = scoco_pa, scoco_ch, scoco_pa - scoco_ch
                    scoco_model['MDL-nonlinear'][i][0], scoco_model['MDL-nonlinear'][i][1], scoco_model['MDL-nonlinear'][i][2]\
                        = scoco_model_pa, scoco_model_ch, scoco_model_pa - scoco_model_ch
                    scoco_data['MDL-nonlinear'][i][0], scoco_data['MDL-nonlinear'][i][1], scoco_data['MDL-nonlinear'][i][2]\
                        = scoco_data_pa, scoco_data_ch, scoco_data_pa - scoco_data_ch

                    scoc['MDL-nonlinear'][i][0], scoc['MDL-nonlinear'][i][1], scoc['MDL-nonlinear'][i][2]\
                        = scoc_pa, scoc_ch, scoc_ch - scoc_pa
                    scoc_model['MDL-nonlinear'][i][0], scoc_model['MDL-nonlinear'][i][1], scoc_model['MDL-nonlinear'][i][2]\
                        = scoc_model_pa, scoc_model_ch, scoc_model_ch - scoc_model_pa
                    scoc_data['MDL-nonlinear'][i][0], scoc_data['MDL-nonlinear'][i][1], scoc_data['MDL-nonlinear'][i][2]\
                        = scoc_data_pa, scoc_data_ch, scoc_data_ch - scoc_data_pa


                    scoo['MDL-nonlinear'][i][0], scoo['MDL-nonlinear'][i][1], scoo['MDL-nonlinear'][i][2]\
                        = scoo_pa, scoo_ch, scoo_ch - scoo_pa
                    scoo_model['MDL-nonlinear'][i][0], scoo_model['MDL-nonlinear'][i][1], scoo_model['MDL-nonlinear'][i][2]\
                        = scoo_model_pa, scoo_model_ch, scoo_model_ch - scoo_model_pa
                    scoo_data['MDL-nonlinear'][i][0], scoo_data['MDL-nonlinear'][i][1], scoo_data['MDL-nonlinear'][i][2]\
                        = scoo_data_pa, scoo_data_ch, scoo_data_ch - scoo_data_pa

                    scon['MDL-nonlinear'][i][0], scon['MDL-nonlinear'][i][1], scon['MDL-nonlinear'][i][2]\
                        = scon_pa, scon_ch, scon_ch - scon_pa
                    scon_model['MDL-nonlinear'][i][0], scon_model['MDL-nonlinear'][i][1], scon_model['MDL-nonlinear'][i][2]\
                        = scon_model_pa, scon_model_ch, scon_model_ch - scon_model_pa
                    scon_data['MDL-nonlinear'][i][0], scon_data['MDL-nonlinear'][i][1], scon_data['MDL-nonlinear'][i][2]\
                        = scon_data_pa, scon_data_ch, scon_data_ch - scon_data_pa

                    #Other
                    scores_conf_ch_pa['MDL-nonlinear'][i][0], scores_conf_ch_pa['MDL-nonlinear'][i][1],\
                    scores_conf_ch_pa['MDL-nonlinear'][i][2], = conf_pa, conf_ch, normalized_gain#conf_pa - conf_ch #order flipped intentionally

                    scores_conf_ch_pa['MDL-nonlinear'][i][0], scores_conf_ch_pa['MDL-nonlinear'][i][1],\
                    scores_conf_ch_pa['MDL-nonlinear'][i][2], = conf_pa, conf_ch, normalized_gain

                    scores_conf_model['MDL-nonlinear'][i][0], scores_conf_model['MDL-nonlinear'][i][1],\
                    scores_conf_model['MDL-nonlinear'][i][2], = como_pa, como_ch, gain_model

                    scores_conf_data['MDL-nonlinear'][i][0], scores_conf_data['MDL-nonlinear'][i][1],\
                    scores_conf_data['MDL-nonlinear'][i][2], = coda_pa, coda_ch, gain_data
                    #scores_conf_z_pa['MDL-nonlinear'][i][0], scores_conf_z_pa['MDL-nonlinear'][i][1],\
                    #scores_conf_z_pa['MDL-nonlinear'][i][2], = conf_pa, conf_z, conf_pa - conf_z #order flipped intentionally


                    print("MDL time:" , round(time.perf_counter() - st, 5), "sec")
                    # Eval
                    res, res1, res0, res_finegr, res_coarse, res_eq = eval_partition('MDL-nonlinear', Piguess = guess_ch,
                                                                             Piparents= Pistar, #TODO guess_pa,
                                                                             C_n=C_n, res=res,
                                                                             res1=res1, res0=res0, res_finegr=res_finegr,
                                                                             res_coarse=res_coarse, res_eq=res_eq)

            # MDL + ILP -----------------

            st = time.perf_counter()
            dists = regression_set.pair_mdl_gains
            if vsn.rff:
                vars_ij, _, c = pi_search_ILP(regression_set.pair_mdl_dists, C_n, wasserstein=True)
            else:
                vars_ij, _, c = pi_search_ILP(dists, C_n, wasserstein=False)

            guess_pi = pi_convertfrom_pair(vars_ij, c, C_n)

            dists = regression_pa.pair_mdl_gains
            if vsn.rff:
                vars_ij, _, c = pi_search_ILP(regression_set.pair_mdl_dists, C_n, wasserstein=True)
            else:
                vars_ij, _, c = pi_search_ILP(dists, C_n, wasserstein=False)

            guess_pa = pi_convertfrom_pair(vars_ij, c, C_n)

            dists = regression_ch.pair_mdl_gains
            if vsn.rff:
                vars_ij, _, c = pi_search_ILP(regression_set.pair_mdl_dists, C_n, wasserstein=True)
            else:
                vars_ij, _, c = pi_search_ILP(dists, C_n, wasserstein=False)

            guess_ch = pi_convertfrom_pair(vars_ij, c, C_n)


            print("ILP time:" , round(time.perf_counter() - st, 5), "sec")
            #dists = regression_base.pair_mdl_gains
            #vars_ij, _, c = pi_search_ILP(dists, C_n, wasserstein=False)
            #guess_base = pi_convertfrom_pair(vars_ij, c, C_n)

            mdl_pi, _, model_pi, _, _ = pi_mdl(regression_set, guess_pi, False, subsample_size)
            mdl_pa, _, model_pa, _, _ = pi_mdl(regression_pa, guess_pa,  False, subsample_size)
            mdl_ch, _, model_ch, _, _ = pi_mdl(regression_ch, guess_ch,  False, subsample_size)
            #mdl_base, _, _, _, _ = pi_mdl(regression_base, guess_base, False)

            one_guess_ch, one_guess_pa = guess_ch, guess_pa
            if is_one_group(guess_ch, C_n):
                one_guess_ch = Pizero
            if is_one_group(guess_pa, C_n):
                one_guess_pa = Pizero
            _, _, one_model_ch, _, _ = pi_mdl(regression_ch, one_guess_ch,  False, subsample_size)
            _, _, one_model_pa, _, _ = pi_mdl(regression_ch, one_guess_pa,  False, subsample_size)


            conf_pa, como_pa, coda_pa = pi_mdl_conf(regression_pa, guess_pa, C_n,  False, subsample_size)
            conf_ch, como_ch, coda_ch = pi_mdl_conf(regression_ch, guess_ch, C_n, False, subsample_size)
            conf_z, _, _ = pi_mdl_conf(regression_set, guess_pi, C_n, False, subsample_size)
            #conf_base = pi_mdl_conf(regression_base, guess_base, C_n, False)

            print("MDL time:" , round(time.perf_counter() - st, 5), "sec")
            if pi_is_insignificant(regression_ch, guess_ch, C_n, regression_per_group=False):
                ins = "(insig)"
            else:
                ins = ""

            out_log.printto('\tMDL_ILP_gain:\tZ->Y', round(mdl_pi,2), "\t", guess_pi, "\tModel: ", round(model_pi,2),
                   '\n\t\tpa(Y)->Y:', round(mdl_pa,2), "\t", guess_pa, "\tModel: ", round(model_pa,2),
                   '\n\t\tch(Y)->Y:',  round(mdl_ch,2), ins, "\t", guess_ch, "\tModel: ", round(model_ch,2))
                   #'\n\t\tB->Y:',  round(mdl_base,2),  "\t", guess_base,

            out_log.printto('\t\tG(pa,Z):', round(mdl_pi - mdl_pa, 2),
                   '\n\t\tG(pa_conf,z_conf):', round(conf_pa - conf_z, 2),
                   '\n\t\tG(pa,ch):', round( mdl_ch - mdl_pa, 2),
                   '\n\t\tG(pa_conf,ch_conf):', round(conf_pa - conf_ch, 2),
                   '\n\t\tG(pa_model,ch_model):', round(model_ch - model_pa, 2))
                   #'\n\t\tG(pa-B,Z-B):', round(( mdl_base -  mdl_pa) - ( mdl_base - mdl_pi), 2),


            scores_pa_z['MDL-nonlinear-ILP'][i][0], scores_pa_z['MDL-nonlinear-ILP'][i][1], \
            scores_pa_z['MDL-nonlinear-ILP'][i][2] = mdl_pa, mdl_pi, mdl_pi - mdl_pa

            scores_pa_ch['MDL-nonlinear-ILP'][i][0], scores_pa_ch['MDL-nonlinear-ILP'][i][1], \
            scores_pa_ch['MDL-nonlinear-ILP'][i][2] =  mdl_pa, mdl_ch, mdl_ch - mdl_pa

            scores_model['MDL-nonlinear-ILP'][i][0], scores_model['MDL-nonlinear-ILP'][i][1], \
            scores_model['MDL-nonlinear-ILP'][i][2] = one_model_pa, one_model_ch, one_model_ch - one_model_pa \
                #without correction: model_pa, model_ch, model_ch - model_pa

            scores_conf_ch_pa['MDL-nonlinear-ILP'][i][0], scores_conf_ch_pa['MDL-nonlinear-ILP'][i][1], \
            scores_conf_ch_pa['MDL-nonlinear-ILP'][i][2] = conf_pa, conf_ch, conf_pa - conf_ch

            scores_conf_z_pa['MDL-nonlinear-ILP'][i][0], scores_conf_z_pa['MDL-nonlinear-ILP'][i][1], \
            scores_conf_z_pa['MDL-nonlinear-ILP'][i][2] = conf_pa, conf_z, conf_pa - conf_z

           # scores_base['MDL-nonlinear-ILP'][i][0], scores_base['MDL-nonlinear-ILP'][i][1], \
           # scores_base['MDL-nonlinear-ILP'][i][2] =  ( mdl_base -  mdl_pa), ( mdl_base - mdl_pi), ( mdl_base -  mdl_pa) - ( mdl_base - mdl_pi)

            #Eval
            res, res1, res0, res_finegr, res_coarse, res_eq = eval_partition('MDL-nonlinear-ILP', Piguess = guess_ch,
                                                                     Piparents= Pistar, #TODOguess_pa,
                                                                     C_n=C_n, res=res,
                                                                     res1=res1, res0=res0, res_finegr=res_finegr,
                                                                     res_coarse=res_coarse, res_eq=res_eq)


            # ILP with Wasserstein distances -----------------
            if vsn.regression_per_pair :
                pair_distances_plain = regression_set.pair_wass_jointGP
            else:
                pair_distances_plain = regression_set.pair_wass_GP
            vars_ij, _, c = pi_search_ILP(pair_distances_plain, C_n, shift=.14 , wasserstein=True)
            guess_pi = pi_convertfrom_pair(vars_ij, c, C_n)
            if vsn.regression_per_pair :
                pair_distances_plain = regression_pa.pair_wass_jointGP
            else:
                pair_distances_plain = regression_pa.pair_wass_GP

            vars_ij, _, c = pi_search_ILP(pair_distances_plain, C_n, shift=.14 , wasserstein=True)
            guess_pa = pi_convertfrom_pair(vars_ij, c, C_n)

            if vsn.regression_per_pair :
                pair_distances_plain = regression_ch.pair_wass_jointGP
            else:
                pair_distances_plain = regression_ch.pair_wass_GP

            vars_ij, _, c = pi_search_ILP(pair_distances_plain, C_n, shift=.14 , wasserstein=True)
            guess_ch = pi_convertfrom_pair(vars_ij, c, C_n)

            ilp_wass_results = pi_matchto_pi(Pizero, guess_ch, C_n)

            out_log.printto('\tWasserstein:\tZ->Y:', guess_pi,
                   "\n\t\tpa(Y)->Y:", guess_pa, "\n\t\tch(Y)->Y:", guess_ch)

            #Eval
            res, res1, res0, res_finegr, res_coarse, res_eq = eval_partition('Wasserstein-ILP', Piguess = guess_ch,
                                                                     Piparents= Pistar,# guess_pa,
                                                                     C_n=C_n, res=res,
                                                                     res1=res1, res0=res0, res_finegr=res_finegr,
                                                                     res_coarse=res_coarse, res_eq=res_eq)

            out_res.printto("\nTime:", round(time.perf_counter() - initial_time, 5), "seconds")
            out_res.printto("\nAt:", time.asctime())

    # Results -----------------
    def print_res(resx, name, out_res):
        out_res.printto(name, '\nTP: %s TN: %s \nFP: %s FN: %s' % (resx[3], resx[0], # resx[0], resx[3], TRUE POS ARE TRUE  NEGs here!
                                                         resx[2],resx[1] #same for FP, FN
                                                          ))
        out_res.printto('F1:', round(f1( resx[3], resx[0], resx[1]), 2)) #f1(resx[0], resx[2], resx[3]),2))

    # TODO consider the gain of using the one variable in addt. to the base set,
    #  rather than comparing baseset+child and baseset+parent directly?
    # or is this something we just substract out bc its part of both?
    # or is it relevant to "scale" the scores correctly?
    # if relevant to scale the scores, can we do so using the confidence trick?



    out_res.printto('\n\n--- Predicted Partitions (anticausal direction) ---')
    for key in res1:
        out_res.printto(key, '(No/One); (Fine/Coarse); (Equal/Incmp); All): \n\t\t\t(%s / %s) ; (%s / %s);  (%s / %s);  %s' %
               (res0[key], res1[key], res_finegr[key], res_coarse[key], res_eq[key],
                counts - (res0[key] + res1[key] + res_finegr[key] + res_coarse[key]), counts),
               '\n\t\t\t(%s / %s) ; (%s / %s);  (%s / %s)'  % (round(res0[key]/counts, 2),   round(res1[key]/counts,2),
               round(res_finegr[key]/counts,2), round(res_coarse[key]/counts,2), round(res_eq[key]/counts,2),
               round((counts - (res0[key]+ res1[key] +res_finegr[key] + res_coarse[key]) )/counts, 2)))



    out_res.printto('\n\n--- Context Pair (Non)Assignment ---')
    for key in res:
        count = res[key]
        print_res(count, key, out_res)

    #printo('\n*** Exact Partition Matches ***', f_name=res_file_nm)
    #for key in res:
    #    printo(key,'\nMatch/Miss: %s / %s' % (res0[key][0], res0[key][1]), '\n', f_name=res_file_nm)

    out_res.printto('\n--- Score differences causal/anticausal ---')
    for key in scores_pa_z:
        out_res.printto('\n' , key)
        if (key=='Vario' or key=='Wasserstein-ILP'or key=='MDL-nonlinear-ILP'):
            todo = [[scores_pa_z, 'L_{PA, p}-L_{PA,ch}:'],
                           [scores_pa_ch, 'L_PA - L_CH:']]
        else:
            todo =  [[scores_pa_z, 'L_{PA, p}-L_{PA,ch}:'],
                           [scores_conf_ch_pa, 'Conf(PA), Conf(CH), Conf(CH)-Conf(PA):'],
                           [scores_conf_model, '.., Como(CH)-Como(PA):'],
                           [scores_conf_data, '.., Coda(CH)-Coda(PA):'],
                           [scoco, 'Coco(CH)-Coco(PA):'],
                           [scoco_model, 'Cocomo(CH) - Cocomo(PA):'],
                           [scoco_data, 'Cocoda(CH) - Cocoda(PA):'],
                           [scores_pa_ch, 'L_PA - L_CH:'],
                           [scores_model, 'Lmo_PA - Lmo_CH:'],
                           [scores_data, 'Lda_PA - Lda_CH:'],
                           [scoc, 'Lco_PA - Lco_CH:'],
                           [scoc_model, 'Lcomo_PA - Lcomo_CH:'],
                           [scoc_data, 'Lcoda_PA - Lcoda_CH:'],
                     #  [scoc, 'L_PA(Picaus) - L_CH(Picaus):'],
                     #  [scoc_model, 'Lmo_PA(Picaus) - Lmo_CH(Picaus):'],
                     #  [scoc_data, 'Lda_PA(Picaus) - Lda_CH(Picaus):'],
                           [scoo, 'L_PA(Pione) - L_CH(Pione):'],
                           [scoo_model, 'Lmo_PA(Pione) - Lmo_CH(Pione):'],
                           [scoo_data, 'Lda_PA(Pione) - Lda_CH(Pione):'],
                           [scon, 'L_PA(Pin) - L_CH(Pin):'],
                           [scon_model, 'Lmo_PA(Pin) - Lmo_CH(Pin):'],
                           [scon_data, 'Lda_PA(Pin) - Lda_CH(Pin):']
                           ]
        ct = 0
        for scores, nm in todo:
            spc = ""
            if ct == 4 or ct == 7 or ct == 10 or ct == 13 or ct == 16:
                spc = "\n"
            ct = ct +1
            s1 = mean(scores[key][:,0])
            s2 = mean(scores[key][:,1])
            s3= mean(scores[key][:,2])
            # shift the numbers to be positive
            #if s1 < 0 or s2 < 0:
            #    s1, s2, s3 = s1 - min(s1, s2) + 0.01, s2 - min(s1, s2)+ 0.01, s3 - min(s1, s2)+ 0.01
            #if s3 < 0:
            #    s1, s2, s3 = s1-s3+ 0.01, s2-s3+ 0.01, s3-s3+ 0.01
            s12 = s1 + s2
            if s12 == 0:
                s12 = 0.01

            s1n = 100 * s1 / s12
            s2n = 100 * s2 / s12
            s3n = 100 * s3 / s12
            out_res.printto(spc, "[", nm, ']\t(%s - %s)  = %s' % (round(s1n, 2), round(s2n, 2), round(s3n, 2)), '\t\tScores: %s - %s  = %s' % (round(s1,2), round(s2,2), round(s3,2)),
                   '\t\t[+- %s  -  +- %s   = +- %s]' %
                   (round(stdev(scores[key][:, 0]), 2), round(stdev(scores[key][:, 1]), 2),
                    round(stdev(scores[key][:, 2]), 2)))

    out_res.printto("Time:", round(time.perf_counter() - initial_time, 5), "seconds")
    print("Time:", round(time.perf_counter() - initial_time, 5), "seconds")
    return res


def eval_partition(key, Piguess, Piparents, C_n,
                   res, res1, res0, res_finegr, res_coarse, res_eq):
    Pizero = [[c_i] for c_i in range(C_n)]
    Pione = [[c_i for c_i in range(C_n)]]

    tpfp = pi_matchto_pi(Pizero, Piguess, C_n)  # tp, fp, fn, tn, and we want pairwise comparison
    for j in range(len(tpfp)):
        res[key][j] = res[key][j] + tpfp[j]
    # how many guesses are the no-group partition (best case)
    is_no_group = pi_matchto_pi_exact(Pizero, Piguess)[0]
    res0[key] = res0[key] + is_no_group
    # how many are the one-group partition (not intended)
    is_one_group = pi_matchto_pi_exact(Pione, Piguess)[0]
    res1[key] = res1[key] + is_one_group

    # how many are more fine grained (intended)
    is_finegrained = pi_check_finegrained(Piguess, Piparents, C_n)
    # how many are less grained (bad)
    is_coarse = pi_check_finegrained(Piparents, Piguess, C_n)

    if ((is_finegrained == 1) and (is_coarse == 1)):
        res_eq[key] = res_eq[key] + 1
    else:
        if not (is_no_group == 1) and not (is_one_group==1):
            res_coarse[key] = res_coarse[key] + is_coarse
            res_finegr[key] = res_finegr[key] + is_finegrained
    return res, res1, res0, res_finegr, res_coarse, res_eq


def pi_check_finegrained(Pifine, Picoarse, C_n):
    pi_valid(Pifine, C_n)
    pi_valid(Picoarse, C_n)

    fine = True
    map_coarse = pi_group_map(Picoarse, C_n)
    for pi in Pifine:
        for c_i in pi:
            for c_j in pi:
                if c_i == c_j:
                    continue
                fine = fine and map_coarse[c_i] == map_coarse[c_j]

    if fine:
        return 1
    else:
        return 0

def test_pi_finegrained():
    res, res1, res0, res_finegr, res_coarse, res_eq = dict(), dict(), dict(), dict(), dict(), dict()
    res["k"] = 0
    res1["k"] = 0
    res0["k"] = 0
    res_finegr["k"] = 0
    res_coarse["k"] = 0
    res_eq["k"] = 0
    eval_partition("k", [[0, 1], [2, 3, 4]], [[0, 1], [2, 3, 4]], 5, res, res1, res0, res_finegr, res_coarse, res_eq)
    eval_partition("k", [[0, 1], [2, 3, 4]], [[0, 1, 2, 3, 4]], 5, res, res1, res0, res_finegr, res_coarse, res_eq)
    #eval_partition("k",[[0, 1], [2, 3, 4]],  [[0], [1], [2, 3, 4]], 5, res, res1, res0, res_finegr, res_coarse)
    #eval_partition("k", [[0, 1, 2, 3, 4]], [[0, 1], [2, 3, 4]], 5, res, res1, res0, res_finegr, res_coarse)
    # eval_partition("k", [[0],[1],[2],[3],[4]],[[0,1],[2,3,4]], 5,res, res1, res0, res_finegr, res_coarse )
    #eval_partition("k", [[0, 1], [2, 3, 4]], [[2, 3], [0,1,4]],  5, res, res1, res0, res_finegr, res_coarse)


def is_one_group(pi, C_n):
    pi_valid(pi, C_n)
    one = True
    map = pi_group_map(pi, C_n)
    for ind in map:
        if not(ind==0):
            one = False
    #for grp in pi:
    #    for c_i in grp:
    #        for c_j in grp:
    #            if not (map[c_i] == map[c_j]):
    #                one = False
    return one