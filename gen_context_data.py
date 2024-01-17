from graphical_models import GaussIntervention, ConstantIntervention
from causaldag import rand
import numpy as np

from intervention_types import IvType
from nonlinear_dag import NonlinearDAG
from linear_dag import LinearDAG
from function_types import FunctionType
from utils import data_scale, cantor_pairing
from utils_pi import pi_group_map, pi_rand


def gen_context_data(C_n, D_n, node_n, seed,
                     fun_type : FunctionType,
                     iv_type_target : IvType,
                     iv_type_covariates : IvType,
                     iid_contexts,
                     partition_search,
                     partition_Y=None, iv_per_node=None,
                     node_density=.5, scale=True,  iv_in_groups=True, vb=False,
                     overlap_observational = True,
                    ):
    """
    From a random DAG (directed erdos), generate data per context.

    :param C_n: number of contexts
    :param D_n: number of samples per context
    :param node_n: number of nodes
    :param node_density: density of connections in DAG
    :param seed: random seed
    :param partition_search: for partition search turn on, for DAG search turn off
    :param partition_Y: if partition search, the partition for a target node
    :param scale: whether data should be scaled (yes)
    :param fun_type: functional model
    :param iid_contexts: no interventions, for example a single context that we split into iid parts
        (overrides iv_per_node, iv_in_groups, and booelean iv types below;
        does not override coefficient changes on target node if partition-search==T)
    :param iv_per_node: range indicating number of contexts where an intervention may happen.
        for example, [0,2] means, for each node Xi, one observational context group
        and 0-2 additional context groups (randomly chosen) with an intervention of the given type.
    :param iv_in_groups: turn on if multiple contexts are affected by the same intervention, turn off if one context aat
        for example, TRUE: a node can have partition [[1, 2, 3], [0, 4]] with iv. in C1,C2,C3,
        FALSE: partitions are of shape [[1], [2, 3, 0, 4]] with iv. in C1.
    :return: Dc (data per context), G (DAG), Gc (Dag per context, shows arc weights in case iv_coefficient=T)
           target, parents, children, confounder (node indices, for partition search)
           node_partitions, observational_group (partition and observation group therein, per node)
    """
    # Number of Changes (interventional context groups) for each node
    if iv_per_node is None:
        k_range = [None, None]  # for each node, in any number of contexts there may be an intervention
    else:
        k_range = [iv_per_node[0] + 1, iv_per_node[1] + 1]  # +1 since one group is observational
        assert 1 <= min(k_range) <= max(k_range) <= C_n
    if iid_contexts:
        k_range = [1, 1]  # no interventions, all nodes in one group
    if partition_Y is None and partition_search:
        partition_Y = [[c_i for c_i in range(C_n - 1)]] + [[C_n - 1]]

    np.random.seed(seed)
    rst = np.random.RandomState(seed)
    dag = rand.directed_erdos(nnodes=node_n, density=node_density)
    dag = rand.rand_weights(dag)

    #Linear Gaussian DAG and Nonlinear DAG
    #lin_dag = LinearDAG(dag.nodes, dag.arc_weights)
    #n_dag = NonlinearDAG(dag.nodes, dag.arc_weights)

    # Partition search only---
    # Choose a target Y in the graph, preferably, with parent nodes, if necessary with confounder
    # (if not possible, postprocessing will disconsider this DAG)
    good_targets = [n for n in range(node_n) if n not in dag.sources()]
    if len(good_targets) == 0:  # choose any
        good_targets = [n for n in range(node_n)]

    confounder = None
    do_confounding = partition_search and iv_type_target == IvType.CONFOUNDING
    if do_confounding:
        # Ensure there is confounder Z and parent X for target Y s.t. X->Y<-Z, Z->X
        for y in good_targets:
            for z in dag.parents_of(y):
                for x in dag.parents_of(y):
                    if not z == x and (z, x) in dag.arcs:
                        confounder = z
                        target = y
                        parents = [p for p in dag.parents_of(y) if p is not confounder]
                        ##print("CONFOUNDER", confounder, "in", dag.parents_of(y))
                        break
    if confounder is None:
        target = good_targets[rst.randint(low=0, high=len(good_targets))]
        parents = dag.parents_of(target)
        confounder = None

    do_confounding = partition_search and iv_type_target == IvType.CONFOUNDING and confounder is not None
    children = dag.children_of(target)

    # ---partition search only

    # Per context: Data and DAGs ----------
    Dc = [None for _ in range(C_n)]
    Gc = [None for _ in range(C_n)]

    # Per variable: context groups, observational group, arc weights of groups
    partitions_X = [None for _ in range(node_n)]
    observational_X = [0 for _ in range(node_n)]
    arc_weights_X = [None for _ in range(node_n)]

    # Partition for each node ----------
    # showing in which groups of contexts an intervention takes place
    for node_X in dag.nodes:
        partition_X = pi_rand(C_n, np.random.RandomState(cantor_pairing(node_X, seed)),# the same for all c_i
                              permute=True,
                              k_min=k_range[0], k_max=k_range[1],
                              single_context_groups=not iv_in_groups)
        if iid_contexts:
            assert len(partition_X) == 1
        else:
            if k_range[0] is not None:
                assert k_range[0] <= len(partition_X) <= k_range[1]

        if partition_search and node_X == target:
            partition_X = partition_Y
        partitions_X[node_X] = partition_X

        if iv_in_groups:
            observational_X[node_X] = np.random.RandomState(cantor_pairing(node_X, seed)).choice(range(len(partition_X)))
        else:
            if len(partition_X) > 1:
                for interventional_group in range(len(partition_X) - 1):
                    assert len(partition_X[interventional_group]) == 1
            observational_X[node_X] = len(partition_X) - 1
        if overlap_observational:
            observational_X[node_X] = 0
        assert 0 <= observational_X[node_X] < len(partition_X)

        iv_type = iv_type_covariates
        if node_X == target:
            iv_type = iv_type_target
        arc_weights_X[node_X] = gen_arc_weights_pi(partition_X, dag, seed, iv_type)


    #n_dag = NonlinearDAG(dag.nodes, dag.arc_weights)
    # DAG and data per context ----------
    for c_i in range(C_n):
        lin_dag_c = LinearDAG(dag.nodes, dag.arcs) #lin_dag.copy()

        # Confounding only in some contexts (partition for Y says which)
        if do_confounding:
            assert partition_search and iid_contexts #TODO is iid_contexts necessary?
            pi_confd = partition_Y
            pi_k = pi_group_map(pi_confd, C_n)[c_i]

            if (confounder, target) in lin_dag_c.arcs:
                w_ij = arc_weights_X[target][pi_k][(confounder, target)]
                a = ": Confounded,"
                if c_i in pi_confd[observational_X[target]]:
                    a = ", Observational, "
                    #w_ij = 0 #TODO if this is removed, there is a parameter change instead (if ivType covariates is parameter change)
                lin_dag_c.set_arc_weight(confounder, target, w_ij)
                ##print("\tContext", c_i, a, confounder, "->", target, "w", round(w_ij,2))

        # Each node has its partition
        for node_X in dag.nodes:
            pi_X = partitions_X[node_X]
            obs_X = observational_X[node_X]
            pi_map = pi_group_map(pi_X, C_n)
            pi_k = pi_map[c_i]

            iv_type_X = iv_type_covariates
            if node_X == target:
                iv_type_X = iv_type_target
            change_parameters = iv_type_X == IvType.PARAM_CHANGE # or iv_type_X is IvType.CONST)

            # Nothing to be done for an observational context (except setting arc weight if param change)
            if c_i in pi_X[obs_X] and not change_parameters:
                continue

            # Interventional Group: arc weight of DAG changes
            if change_parameters:
                weights_to_p = arc_weights_X[node_X][pi_k]
                for (i, j) in dag.arcs:
                    if j == node_X and not (i == confounder and node_X == target):
                        w_ij = weights_to_p[(i, j)]
                        lin_dag_c.set_arc_weight(i, j, w_ij) #a new DAG where causal weight in context is the group's
            else:
                if iv_type_X == IvType.SHIFT:
                    lin_dag_c.set_node_bias(node_X, 2) #TODO
                if iv_type_X == IvType.SCALE:
                    lin_dag_c.set_node_variance(node_X, 20)
                if iv_type_X == IvType.CONFOUNDING:
                    pass #nothing to be done
                #interv_dict_c[node_X] = gen_intervention(pi_k, iv_type_X)

        # Sample data in context c ----------
        # Linear Gaussian Models
        if fun_type is FunctionType.LINEAR_GAUSS:
            data_c = lin_dag_c.sample(nsamples=D_n)
            # Nonlinear:
            #   Ndag_c = NonlinearDAG(Gdag_c.nodes, Gdag_c.arc_weights)
            #   data_c = n_dag.sample_context_data(D_n, C_n, c_i, seed, partitions_X)
            if scale:
                data_c = data_scale(data_c)
            Dc[c_i] = data_c
            Gc[c_i] = lin_dag_c


    if fun_type is FunctionType.GP_PRIOR:
        ndag = NonlinearDAG(dag.nodes, dag.arc_weights)
        Dc = ndag.sample_data(D_n, C_n, seed, partitions_X, target, iv_type_target, iv_type_covariates)

        for c_i in range(C_n):
            Gc[c_i] = ndag

    if vb:
        print("\n*** DAG Information ")
        print("\nTARGET Y =", target)
        for p in dag.nodes:
            if iv_type_target == IvType.CONFOUNDING:
                if (not p in parents) and (not (p==confounder)):
                    continue
            else:
                if not p in parents and not p in children:
                    continue
            if fun_type is FunctionType.LINEAR_GAUSS:
                if p==confounder:
                    print("CONFOUNDER X =", p, "\n\t X->Y:\t", partitions_X[target])
                else:
                    if p in parents:
                        print("PARENT X =", p, "\n\t X->Y:\t", partitions_X[target])
                    else:
                        print("CHILD Z =", p, "\n\tpa(Z)->Z:\t", partitions_X[p])


                for (i,j) in dag.arcs:
                    if j == target and i == p\
                            or j == p:
                        w = [0 for _ in range(C_n)]
                        for c_i in range(C_n):
                            if (i, j) in Gc[c_i].arc_weights:
                                w[c_i] = round(Gc[c_i].arc_weights[(i, j)],2)

                        print("\t", i, "->", j, "\t", w)
            else:
                for (i,j) in dag.arcs:
                    if j == target and i == p:
                        print("EDGE", i, "->", j)
                        print("\tPartition", partitions_X[j])

    return Dc, dag, Gc, \
           target, parents, children, confounder, \
           partitions_X, observational_X




def gen_arc_weights_pi_random(partition, dag):
    arc_weights_k = [0 for _ in range(len(partition))]

    for pi_k in range(len(partition)):
        Gdag_k = rand.rand_weights(dag)
        arc_weights_k[pi_k] = Gdag_k.arc_weights
    return arc_weights_k

def gen_arc_weights_pi(partition, dag, seed, iv_type):

    arc_weights_k = [{} for _ in range(len(partition))]
    if len(partition) == 1:
        for (i, j) in dag.arcs:
            arc_weights_k[0][(i, j)] = dag.arc_weights[(i, j)]
        return arc_weights_k

    mn = 80 / max(len(partition) - 1, 1)  # weights at least in [0,80]
    mx = 100 / max(len(partition) - 1, 1)

    if True: #for both IVType.Paramchange AND Hardintervention

        for (i, j) in dag.arcs:
            seed_ij = cantor_pairing(seed, cantor_pairing(i, j))
            for k in range(len(partition)):
                if k % 2 == 0:
                    sign = 1
                    off = k/2
                else:
                    sign = -1
                    off = -(k-1)/2
                arc_weights_k[k][(i, j)] =  np.random.RandomState(seed_ij).normal(loc=sign * dag.arc_weights[(i, j)] + off, scale=.5)

            # special case: if iv_type is hard/surgical intervention: set  arc weights in one group to zero
            #if iv_type == IvType.CONST:
            #    arc_weights_k[0][(i, j)] = 0
        return arc_weights_k


    return arc_weights_k

def test_arc_weights_k() :
    seed = 1
    from intervention_types import IvType
    np.random.seed(seed)

    rst = np.random.RandomState(seed)
    dag = rand.directed_erdos(nnodes=4, density=.3)
    dag = rand.rand_weights(dag)
    x = gen_arc_weights_pi([[0], [1], [2]], dag, 1, IvType.PARAM_CHANGE)

