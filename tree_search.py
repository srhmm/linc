import itertools
import time

from out import Out
from pi_tree import PiTree, is_insignificant, eval_reverse_edge, eval_local_edges, print_update, print_deletion, \
    print_current
from upq import UPQ
import numpy as np


def tree_search(XYc, vsn,
                refined_backward=False, # adds a step to edge removal compared to Mian et al.
                revisit_children=True,
                revisit_queue=True,
                revisit_parentsets=True,
                vb=True, tofile=True,
                prnt = "(not provided)"):
    """ Tree search for the causal DAG

    :param XYc: Data, len(XYc)=C_n contexts, len(XYc[1])=D_n samples per context, len(XYc[1][1])=X_n variables
    :param revisit_children: debug
    :param revisit_queue: debug
    :param revisit_parentsets: whether backward search phase activated
    :param vario_score: T/F
    :param emp: T/F
    :return: T : pi_tree, estimated DAG with partitions at each edge
    """
    # C_n, D_n, X_n = XYc.shape[0], XYc.shape[1], XYc.shape[2]
    #C_n, D_n, X_n = len(XYc), len(XYc[1]), len(XYc[1][1])
    C_n = len(XYc)
    D_n = min([XYc[i].shape[0]] for i in range(len(XYc)))[0]
    X_n = XYc[0].shape[1]
    q = UPQ()
    T = PiTree(C_n, X_n, XYc, vsn) #vsn.mdl_gain, vsn.ilp_in_tree_search, regression_per_group=vsn.regression_per_group)

    out = Out("tree/log_nodes."+str(len(T.nodes))+".txt", vb=vb, tofile=tofile)
    out.printto("\n\n*** Tree Search ***")
    out.printto("True edges")
    out.printto("\t" + str(prnt))

    st = time.perf_counter()
    initial_time = time.perf_counter()
    #out.printto("Initial edges")
    q = T.initial_edges(q, out, D_n)
    print("TREE initial edges: ", time.perf_counter() - st, "sec")

    # FORWARD: Adding Edges
    out.printto("Forward phase")

    st = time.perf_counter()

    while q.pq:
        try:
            pi_edge = q.pop_task()

            target = pi_edge.j
            parent = pi_edge.i

            # Check whether adding the edge would result in a cycle
            if T.has_cycle(parent, target):
                continue
            gain, score, mdl, pi, _ = T.eval_edge(pi_edge)  # computes or looks up score of edge {currentParents(j), i} -> j

            # Check whether gain is significant
            if is_insignificant(gain, vario_score=vsn.vario_in_tree_search, emp=True, mdl_gain=vsn.mdl_gain):
                out.printto("\t(INS): " + print_current(pi_edge, T) )
                continue

            out.printto("\t(ADD): " + print_current(pi_edge, T))

            T.add_edge(parent, target, score, mdl, pi)

            # Reconsider children under current model and remove if reversing the edge improves score
            if revisit_children:
                for child in T.nodes:  #parallel
                    if not T.is_edge(target, child):
                        continue
                    edge_forward = T.edges[child][target]  # target -> child
                    gain = eval_reverse_edge(edge_forward, T)  # evaluates target <- child

                    if not is_insignificant(gain, vario_score=vsn.vario_in_tree_search, emp=True, mdl_gain=vsn.mdl_gain):

                    # TODO alternative:
                    #if gain > 0:

                        # Remove the edge, update the gain of both edges

                        out.printto("\t\t(REV): " + print_current(edge_forward, T))
                        T.remove_edge(edge_forward.i, edge_forward.j)
                        edge_backward = T.init_edges[target][child]  # child->target. TODO can take from init edges?

                        # might have been skipped due to insig: assert (q.exists_task(edge_backward)) #since the reverse edge is not in the model (otherwise cycle)
                        assert not (q.exists_task(
                            edge_forward))  # since this was included in the model, i.e. removed from queue at some point
                        if (q.exists_task(edge_backward)):
                            q.remove_task(edge_backward)

                        gain_backward, _, _, _, _ = T.eval_edge(edge_backward)
                        gain_forward, _, _, _, _ = T.eval_edge(edge_forward)
                        q.add_task(edge_backward, -gain_backward * 100)
                        q.add_task(edge_forward, -gain_forward * 100) #TODO is it instead good to not add again?

                        out.printto("\t\t " + print_deletion(target, child, T, gain_forward, gain_backward))
                    else:
                        out.printto("\t\t(--): " + str(edge_forward.i) + "--> "+ str(edge_forward.j) + " not worth REV: gain = " + str(gain))

            # Reconsider edges Xk->Xj in q given the current model
            if revisit_queue:
                for mom in T.nodes: #parallel
                    if target == mom or parent == mom \
                            or T.is_edge(mom, target) or T.is_edge(target, mom):
                        continue
                    edge_candidate = T.init_edges[mom][target]
                    gain, score, mdl, pi, _ = T.eval_edge(edge_candidate)
                    neg_gain = -gain * 100

                    if (q.exists_task(edge_candidate)):  # ow. insignificant /skipped
                        q.remove_task(edge_candidate)
                        q.add_task(edge_candidate, neg_gain)

        except (KeyError):  # empty or all remaining entries are tagged as removed
            pass

        # with open('out_tree_search.txt', 'a') as f:
        #    print("Adding" ,pi_edge, "\tGain", diff,  file=f)

    print("TREE forward phase: ", time.perf_counter() - st, "sec")
    st = time.perf_counter()
    out.printto("Backward phase")

    # BACKWARD: Refining Edges
    if revisit_parentsets:
        for target in T.nodes:
            parents = [parent for parent in T.nodes if T.is_edge(parent, target)]
            out.printto("\tNode: " + str(target), "\t<--", parents, round(T.get_graph_mdlcost(),2))
            if len(parents) <= 1:
                continue
            max_gain = -np.inf
            arg_max = None

            # Consider all graphs G' that use a subset of the target's current parents
            min_size = 1 # min_size = 0 before
            for k in range(min_size, len(parents)):
                parent_sets = itertools.combinations(parents, k)
                for parent_set in parent_sets:
                    gain = eval_local_edges(target, T, parent_set, None)
                    if gain > max_gain:
                        max_gain = gain
                        arg_max = parent_set

            #TODO is insignificant?
            if (arg_max is not None) and (not is_insignificant(max_gain, vario_score=vsn.vario_in_tree_search, emp=True, mdl_gain=vsn.mdl_gain)):

                # In addition consider flipping some edges
                if refined_backward:
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
                    T.update_edges(target, arg_max)
                    out.printto("\t\t(UP): "+ print_update(target, T, arg_max, max_gain, T.get_graph_score()))
            else:
                out.printto("\t\t(--): pa=" + str(parents), "insig. gain of", round(max_gain,2))
    print("TREE backward phase: ", time.perf_counter() - st, "sec")

    print("TREE overall: ", time.perf_counter() - initial_time, "sec")
    out.printto("Scores estimated")
    for x in T.score_cache:
        out.printto("\t" + str(x) + ": " + str(round(T.score_cache[x], 3)))
    out.printto("Final Edges")
    out.printto("\t", T.adj)
    out.close()
    return T




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

