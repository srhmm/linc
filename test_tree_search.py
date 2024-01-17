import csv
import time
from statistics import mean, variance, stdev

import numpy as np

import function_types
import intervention_types
import out_naming
from gen_context_data import gen_context_data
from test_pi_search import f1
from pi_tree import eval_edge_orientation
from tree_search import tree_search
from utils_pi import pi_group_map


def test_tree_search(vsn,
                     C_n = 5,
                     D_n = 500,
                     node_n = 3,
                     fun_type=function_types.FunctionType.GP_PRIOR,
                     iv_type=intervention_types.IvType.CONST,
                     iv_per_node=[1,1],
                     initial_seed=100,
                     iters = 1,
                     run_id=0,
                     run_name="",
                     cmp_sid = True,
                     vb=True, tofile=True
                     ):

    st_time = time.time()
    file_nm = out_naming.unique_name(C_n, D_n, node_n, fun_type, iv_type, vsn, iv_per_node)
    out, outres, dag_folder = out_naming.folders_test_tree_search(file_nm)

    outres.printto("*** RUN ***")
    outres.printto("Run ID: ", run_id)
    outres.printto("Repeats: ", iters) 
    outres.printto("Seeds: ", initial_seed, "--", initial_seed+iters) 
    outres.printto("DAGs in: ", dag_folder) 

    TP, FP, TN, FN, FP_anticausal = 0, 0, 0, 0, 0
    shds, sids, tms = [np.inf for _ in range(iters)],  [np.inf for _ in range(iters)],  [0 for _ in range(iters)]
    counts = 0
    asyms_caus, asyms_rev = [0 for _ in range(iters)],  [0 for _ in range(iters)]
    count_caus, count_rev = 0,0
    it = -1

    for iteration in range(iters):
        it_time = time.perf_counter()
        it = it  + 1
        seed = iteration + initial_seed
        print("\n\nIteration", iteration + 1,  "/", iters)
        out.printto("\n*** Iteration", iteration + 1, "/", iters, "***")

        Dc, G, Gc, _, _, _, _, Pis, obs = gen_context_data(D_n=D_n, C_n=C_n, partition_search=False,
                                                           fun_type=fun_type,
                                                           iv_type_target=iv_type,  # doesnt matter here,
                                                           iv_type_covariates=iv_type,
                                                           iid_contexts=False, iv_in_groups=True,
                                                           iv_per_node=iv_per_node,
                                                           node_n=node_n, seed=seed)
        T = tree_search(Dc, vsn,
                        revisit_children=True, revisit_queue=True, revisit_parentsets=True, prnt=G.arcs,
                        vb=vb, tofile=tofile)
        out.printto("DAG estim\n\t", T.adj)
        out.printto("DAG truth\n\t",str(G.arcs))
        out.printto("Partitions")
        for node in T.nodes:
            out.printto("\tnode: "+str(node)+", estim: "+str(T.node_pi[node]) + ", truth: "+str(Pis[node]))

        for i in T.nodes:
            for j in T.nodes:
                if T.is_edge(i, j):
                    if (i, j) in G.arcs:
                        TP = TP + 1
                    else:
                        FP = FP + 1
                        if (j, i) in G.arcs:
                            FP_anticausal = FP_anticausal + 1
                else:
                    if (i, j) in G.arcs:
                        FN = FN + 1
                    else:
                        TN = TN + 1

        trueG = np.array([[(0 if n == 0 else 1) for n in G.weight_mat[j, :]] for j in range(len(G.weight_mat))])
        estimG = T.get_adj()
        np.save(dag_folder+"true_"+str(iteration), trueG)
        np.save(dag_folder+"estim_"+str(iteration), estimG)
        with open(dag_folder+"pi_true_"+str(iteration)+".csv", 'w', newline="") as f:
            for node in T.nodes:
                writer = csv.writer(f)
                writer.writerows([pi_group_map(Pis[node], C_n)])

        with open(dag_folder + "pi_estim_" + str(iteration)+".csv", 'w', newline="") as f:
            for node in T.nodes:
                writer = csv.writer(f)
                writer.writerows([pi_group_map(T.node_pi[node], C_n)])
        if cmp_sid:
            shd, sid = T.get_shd(trueG), T.get_sid(trueG).item()
        else:
            shd, sid = 0,0
        shds[it] = shd
        sids[it] = sid

        asym_caus_i, asym_rev_i, has_caus, has_rev, ct_caus, ct_rev = investigate_edges(T,G,out)

        asyms_caus[it] = asym_caus_i
        asyms_rev[it] = asym_rev_i
        count_caus = count_caus + ct_caus
        count_rev = count_rev + ct_rev

        out.printto("SHD:", shd)
        out.printto("SID:", sid)
        counts =  counts + 1
        print("TREE iteration time: ", time.perf_counter() - it_time)
        tms[it] = time.perf_counter() - it_time

    assert(counts == iters)
    outres.printto("\nNodes: "+str(node_n)+"\nContexts: "+str(C_n)+"\nContext samples: "+str(D_n)+"\nFunType:"+str(fun_type)+"\nIvType:"+str(iv_type)+
                "\n*** Result ***\nCounts: ("+str(TP)+","+ str(FP)+ ","+ str(TN)+ ","+str(FN)+ ","+str(FP_anticausal)+
              ")\nF1: " +str(round(f1(TP, FP, FN), 2)), "\nPrecis & Rec:" , str(round(TP / max(TP + FP, 0.001), 2) ), ",", str(round( TP / max(TP + FN, 0.001), 2) ) )
    if counts==0:
        counts = 0.01
    if count_rev == 0:
        count_rev= 0.01
    if count_caus == 0:
        count_caus= 0.01
    z = 1.96
    if cmp_sid:
        outres.printto("SHD:", round(sum(shds) /counts, 2))
        outres.printto("SID:", round(sum(sids) /counts, 2))

        mn_sid, var_sid, dev_sid = mean(sids), variance(sids), stdev(sids)
        confint_sid = z * dev_sid / np.sqrt(len(sids))
        outres.printto("SID avg:", round(mn_sid, 3), "\t-var:", round(var_sid, 2), "-std:", round(dev_sid, 2),
              "-cf:", round(confint_sid, 3))
        fct = node_n * (node_n - 1)
        sid_norm = [sid / fct for sid in sids]
        outres.printto("SID scaled:", round(mean(sid_norm), 3), "\t-var:", round(variance(sid_norm), 2), "-std:",
              round(stdev(sid_norm), 2), "-cf:", round(z * stdev(sid_norm) / np.sqrt(len(sids)), 2))

    outres.printto("Gain causal orientation:", round(sum(asyms_caus) /count_caus, 2), ", how often: ", count_caus)
    outres.printto("Gain anticausal orientation:", round(sum(asyms_rev) /count_rev, 2), ", how often: ", count_rev)
    mn, var, dev = mean(asyms_caus), variance(asyms_caus), stdev(asyms_caus)
    confint = z * dev / np.sqrt(len(asyms_caus))
    outres.printto("Gain causal:", round(mn, 3), "\t-var:", round(var, 3), "-std:", round(dev, 3),
                   "-cf:", round(confint ,  3))
    mn, var, dev = mean(asyms_rev), variance(asyms_rev), stdev(asyms_rev)
    confint = z * dev / np.sqrt(len(asyms_rev))
    outres.printto("Gain anticausal:", round(mn, 3), "\t-var:", round(var, 3), "-std:", round(dev, 3),
                   "-cf:", round(confint ,  3))


    outres.printto("Time:", round(time.time() - st_time, 2))
    outres.printto("Time per DAG:", round(sum(tms)/iters, 2))
    outres.printto("\nAt:", time.asctime(), "\n***\n")
    outres.close()
    out.close()


def investigate_edges(T, G, out):
    """ Considers each cause-effect pair i,j s.t. i->j in true DAG.
    To evaluate how informative our score is about causal directions (and to take out of account mistakes we make due to the greedy search algorithm),
    we consider the following score differences:
        -informed score difference: {true_pa(j)\i, i} -> j (correct edge) vs. {true_pa(i) & j} -> i (flipping edge locally)
        -local score difference: {estim_pa(j)\i, i} -> j (correct orientation) vs. {estim_pa(i) & j} -> i (flipping edge locally)

    Args:
        T: estimated DAG, as a Tree object
        G: true DAG
        out: Out, for printing

    Returns:

    """
    out.printto("Edge Analysis")
    asym_caus, asym_rev = 0, 0
    ct_caus, ct_rev = 0,0
    has_caus, has_rev = True , True #not needed for this version
    for j in range(len(G.weight_mat)):
        for i in range(len(G.weight_mat[j])):

            #Disregard false negative and true positive edges
            is_relevant = (not G.weight_mat[i][j]==0) and (T.is_edge(i,j) or T.is_edge(j, i))
            if not is_relevant:
                continue

            ct_caus = ct_caus + 1
            ct_rev = ct_rev + 1
            #if i->j: gain of incl i as parent of j:  pa(j),i->j - pa(j)\i ->j
            # should be larger than gain of incl j as parent of i: pa(i),i->j - pa(i),j

            gain_ij, score_ij, mdl_ij, pi_ij, \
            gain_ji, score_ji, mdl_ji, pi_ji = eval_edge_orientation(i, j, T)

            out.printto("\tEdge", i, "->" , j, ": Score caus", round(score_ij,2),  "Score rev", round(score_ji,2), "\t/Gain caus", round(gain_ij,2), "/Gain rev", round(gain_ji,2))
            print("\tEdge", i, "->", j, ": Score caus", round(score_ij, 2), "Score rev", round(score_ji, 2),
                        "\t/Gain caus", round(gain_ij, 2), "/Gain rev", round(gain_ji, 2))
            asym_caus = gain_ij
            asym_rev = gain_ji


    return asym_caus, asym_rev, has_caus, has_rev, ct_caus, ct_rev



#Not sure how much sense comparing edges btw variable pairs (i,j) to (j,i) makes, when afterwards we use scores for edges (pa_j, j) instead.
def investigate_single_edges(T, G, out):
    out.printto("Edge Analysis")
    asym_caus, asym_rev = 0, 0
    ct_caus, ct_rev = 0,0
    has_caus, has_rev = False, False
    for j in range(len(G.weight_mat)):
        for i in range(len(G.weight_mat[j])):

            # Investigate all edges i->j that are either in the true or in the discovered graph
            if (not G.weight_mat[i][j]==0) or T.is_edge(i,j):

                # Score for Discovered_parents(i)->i, score for {j}->i and score for {i}->j
                score_actual, score_ji, score_ij = 0,0,0
                skip = False
                pa = [node for node in range(T.X_n) if node in T.pa[j]]
                hash_key = f'j_{str(j)}_pa_{str(pa)}'
                if T.score_cache.__contains__(hash_key):
                    score_actual = T.score_cache[hash_key]

                pa = [i]
                hash_key = f'j_{str(j)}_pa_{str(pa)}'
                if T.score_cache.__contains__(hash_key):
                    score_ij = T.score_cache[hash_key]
                # Disregard false negative edges that were never considered during search
                else:
                    skip=True

                pa = [j]
                hash_key = f'j_{str(i)}_pa_{str(pa)}'
                if T.score_cache.__contains__(hash_key):
                    score_ji = T.score_cache[hash_key]
                else:
                    skip=True

                if T.is_edge(i,j):
                    out.printto("\tC", i, "->" , j, ":", round(score_actual,2), "\tAsym:", round(score_ij,2), "<", round(score_ji,2))
                    if not skip and (not G.weight_mat[i][j]==0):
                        asym_caus = asym_caus + score_ji - score_ij
                        has_caus = True
                        ct_caus = ct_caus+1
                else:
                    if T.is_edge(j,i):
                        out.printto("\tX", i, "->" , j, ":", round(score_actual,2), "\tAsym:",  round(score_ij,2), ">", round(score_ji,2))
                        if not skip and (not G.weight_mat[i][j]==0):
                            asym_rev = asym_rev + score_ij - score_ji
                            has_rev = True
                            ct_rev = ct_rev+1
                    else:
                        out.printto("\t--", i, "->", j,":", round(score_actual,2), "\tAsym:",  round(score_ij,2), ",",round( score_ji,2))

    return asym_caus, asym_rev, has_caus, has_rev, ct_caus, ct_rev

