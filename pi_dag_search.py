import itertools
import time

from out import Out
from pi_dag import PiDAG, q_entry
from pi_tree import PiTree, is_insignificant, eval_reverse_edge, eval_local_edges, print_update, print_deletion, \
    print_current
from upq import UPQ
import numpy as np


def pi_dag_search(XYc, vsn):
    """ Tree search for the causal DAG

    :param XYc: Data, len(XYc)=C_n contexts, len(XYc[1])=D_n samples per context, len(XYc[1][1])=X_n variables
    :return: T : pi_dag, estimated DAG with edges {X1,...Xn}->Y and one context partition,e.g. [[0,1],[2]], for each Y
    """
    C_n = len(XYc)
    #D_n = min([XYc[i].shape[0]] for i in range(len(XYc)))[0]
    X_n = XYc[0].shape[1]
    q = UPQ()
    pi_dag = PiDAG(C_n, X_n, XYc, vsn)

    out = Out("tree/log_nodes."+str(X_n)+".txt", vb=False, tofile=False)
    out.printto("\n\n*** Tree Search ***")

    st = time.perf_counter()
    initial_time = time.perf_counter()
    #out.printto("Initial edges")
    q = pi_dag.initial_edges(q)
    print("TREE initial edges: ", time.perf_counter() - st, "sec")

    # FORWARD: Adding Edges
    out.printto("Forward phase")
    st = time.perf_counter()

    while q.pq:
        try:
            pi_edge = q.pop_task()

            j = pi_edge.j
            parent = pi_edge.i

            # Check whether adding the edge would result in a cycle
            if pi_dag.has_cycle(parent, j):
                continue
            gain, score, mdl, pi, pa,score_cur, mdl_cur, pi_cur, pa_cur = pi_dag.eval_edge_addition(j, parent)  # computes or looks up score of edge {currentParents(j), i} -> j

            # Check whether gain is significant
            if is_insignificant(gain, vario_score=vsn.vario_in_tree_search, emp=True, mdl_gain=vsn.mdl_gain):
                continue

            pi_dag.add_edge(parent, j, score, mdl, pi)

            # Reconsider children under current model and remove if reversing the edge improves score
            if True:
                for ch in pi_dag._nodes:  #parallel
                    if not pi_dag.is_edge(j, ch):
                        continue
                    gain = pi_dag.eval_edge_flip(j, ch)  # evaluates j <- child when j->child is currently an edge

                    if not is_insignificant(gain, vario_score=vsn.vario_in_tree_search, emp=True, mdl_gain=vsn.mdl_gain):
                        # Remove the edge, update the gain of both edges
                        pi_dag.remove_edge(j, ch)

                        edge_fw = pi_dag.pair_edges[j][ch]
                        edge_bw = pi_dag.pair_edges[ch][j]
                        assert edge_fw.i == j and edge_fw.j == ch
                        assert edge_bw.i == ch and edge_bw.j == j

                        assert not (q.exists_task(edge_fw))  # since this was included in the model, i.e. removed from queue at some point
                        # might have been skipped due to insig: assert (q.exists_task(edge_backward))
                        if (q.exists_task(edge_bw)):
                            q.remove_task(edge_bw)

                        gain_bw, _, _, _, _ ,_,_,_,_ = pi_dag.eval_edge_addition(edge_bw.i, edge_bw.j)
                        gain_fw,_, _, _, _ ,_,_,_,_ = pi_dag.eval_edge_addition(edge_fw.i, edge_fw.j)
                        q.add_task(edge_bw, -gain_bw * 100)
                        q.add_task(edge_fw, -gain_fw * 100) #TODO is it instead good to not add again?

                        #out.printto("\t\t " + print_deletion(target, child, T, gain_forward, gain_backward))
                    #else:
                    #    out.printto("\t\t(--): " + str(edge_forward.i) + "--> "+ str(edge_forward.j) + " not worth REV: gain = " + str(gain))
            print("")
            # Reconsider edges Xk->Xj in q given the current model as their score changed upon adding Xi->Xj
            if True:
                for mom in pi_dag._nodes:
                    # Do not consider Xi,Xj, or current parents/children of Xi
                    if j == mom or parent == mom \
                        or pi_dag.is_edge(mom, j) or pi_dag.is_edge(j, mom):
                        continue
                    edge_candidate = pi_dag.pair_edges[mom][j] #pi_dag.init_edges[mom][target]
                    gain_bw,score,mdl,pi, _ ,_,_,_,_ = pi_dag.eval_edge_addition(j, mom)

                    if (q.exists_task(edge_candidate)):  # ow. insignificant /skipped
                        q.remove_task(edge_candidate)
                        q.add_task(edge_candidate,  -gain * 100)

        except (KeyError):  # empty or all remaining entries are tagged as removed
            pass

        # with open('out_tree_search.txt', 'a') as f:
        #    print("Adding" ,pi_edge, "\tGain", diff,  file=f)

    print("TREE forward phase: ", time.perf_counter() - st, "sec")
    st = time.perf_counter()
    out.printto("Backward phase")

    # BACKWARD: Refining Edges
    if True:
        for j in pi_dag._nodes:
            parents = pi_dag.parents_of(j) #[p for p in pi_dag._nodes if pi_dag.is_edge(p, j)]
            #out.printto("\tNode: " + str(target), "\t<--", parents, round(T.get_graph_mdlcost(),2))
            if len(parents) <= 1:
                continue
            max_gain = -np.inf
            arg_max = None

            # Consider all graphs G' that use a subset of the target's current parents
            min_size = 1 # min_size = 0 before
            for k in range(min_size, len(parents)):
                parent_sets = itertools.combinations(parents, k)
                for parent_set in parent_sets:
                    gain = pi_dag.eval_local_edges(j, parent_set, pi_dag.children_of(j))
                    if gain > max_gain:
                        max_gain = gain
                        arg_max = parent_set

            #TODO is insignificant?
            if (arg_max is not None) and (not is_insignificant(max_gain, vario_score=vsn.vario_in_tree_search, emp=True, mdl_gain=vsn.mdl_gain)):

                # In addition consider flipping some edges
                if False: #TODO refined_backward:
                    #instead of removing edges, consider flipping them:
                    remaining = [n for n in parents if not n in arg_max]
                    arg_max_reversing = None
                    max_gain_reversing = -np.inf
                    for k in range(1, len(remaining)):
                        subsets = itertools.combinations(remaining, k)
                        for child_set in subsets:
                            gain = eval_local_edges(target, T, arg_max, child_set)
                            if gain > max_gain_reversing:
                                max_gain_reversing = gain
                                arg_max_reversing = child_set
                    if (arg_max_reversing is not None and max_gain_reversing>0):
                        T.update_edges(target, arg_max, arg_max_reversing)
                        out.printto("\t\t(UP-REV): " , arg_max, max_gain, arg_max_reversing, max_gain_reversing, T.get_graph_score())
                    else:
                        T.update_edges(target, arg_max)
                        out.printto("\t\t(UP): "+ print_update(target, T, arg_max, max_gain, T.get_graph_score()))
                else:
                    pi_dag.update_local_edges(j, arg_max, pi_dag.children_of(j))
                    #out.printto("\t\t(UP): "+ print_update(target, T, arg_max, max_gain, T.get_graph_score()))
            #else:
                #out.printto("\t\t(--): pa=" + str(parents), "insig. gain of", round(max_gain,2))
    print("TREE backward phase: ", time.perf_counter() - st, "sec")

    print("TREE overall: ", time.perf_counter() - initial_time, "sec")
    #out.printto("Scores estimated")
    #for x in T.score_cache:
    #    out.printto("\t" + str(x) + ": " + str(round(T.score_cache[x], 3)))
    #out.printto("Final Edges")
    #out.printto("\t", T.adj)
    #out.close()
    return pi_dag




def partition_search(XYc, target, parents, out,
                     vario_score=False, emp=True):
    ''' Partition search for a node with known causal parents
    (for simplicity, add edges one at a time to a tree as in tree search; tree class is just a wrapper here)
    (similar to test_partition_dag)

    :param XYc: Data, len(XYc)=C_n contexts, len(XYc[1])=D_n samples per context, len(XYc[1][1])=X_n variables
    :param target: one of X_n nodes
    :param parents: causal parents in the generating DAG
    :param vario_score: T/F
    :param emp: T/F
    :return:
    '''
    C_n, D_n, X_n = len(XYc), len(XYc[1]), len(XYc[1][1])

    q = UPQ()
    T = PiTree(C_n, X_n, XYc, vario_score, emp)
    _ = T.initial_edges(q, out, D_n)

    for parent in parents:
        if T.has_cycle(parent, target):
            continue
        pi_edge = T.init_edges[target][parent]
        gain, score, mdl, pi, _ = T.eval_edge(pi_edge)  # computes or looks up score of edge {currentParents(j), i} -> j

        #if is_insignificant(gain, vario_score=vario_score, emp=emp):
        #    print ("insignificant")

        T.add_edge(parent, target, score, mdl, pi) #round(gain, 2),
    return T

