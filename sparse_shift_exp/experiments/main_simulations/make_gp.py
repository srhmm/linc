from enum import Enum

from causaldag import rand
import numpy as np
import more_itertools as mit
from sklearn import preprocessing

from graphical_models import GaussDAG
import numpy as np
from typing import List, Union, Set, Tuple, Dict, Any

from sklearn.gaussian_process.kernels import RBF


def cantor_pairing(x, y):
    return int((x + y) * (x + y + 1) / 2 + y)

# see https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning/10036764#10036764
def accel_asc(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]



def pi_enum(n_contexts, permute=True, k_min=1, k_max=None):
    """
            Enumerates all partitions of a given size

            Parameters
            ----------
            n_contexts:
                number of contexts.
            permute:
                if yes, all partitions, if no, skip permutations of context numbering, e.g. [[1],[2,3]]=[[1,2],[3]]
            k_min:
                min length of the partition = min number of groups, 1<=kmin<=kmax<=n_contexts
            k_max:
                max length
            Returns
            -------
            (n_contexts)
                list of partitions.
            Examples
            --------
            pi_enum(5) = [[[0,1,2,3,4]], ... [[0],[1],[2],[3],[4]]
            """
    if k_max is None:
        k_max = n_contexts
    assert 1 <= k_min <= k_max <= n_contexts

    lst = range(n_contexts)
    if permute:
        return [part for k in range(k_min, k_max + 1) for part in mit.set_partitions(lst, k)]

    parts = [part for part in accel_asc(n_contexts)]

    # change the format  from group sizes, e.g. [1,1,1] to lists with context indices per group, e.g. [[0],[1],[2]]
    for p in range(len(parts)):
        sizes = parts[p]
        partition = [[] for _ in range(len(sizes))]
        next = 0
        for i in range(len(sizes)):
            partition[i] = [next + j for j in range(sizes[i])]
            next = next + sizes[i]
        parts[p] = partition
    parts = [p for p in parts if k_min <= len(p) <= k_max]

    return parts

def pi_rand(n_contexts, random_state=np.random.RandomState(1), permute=True,
            k_min=None, k_max=None, single_context_groups=False):
    """ Chooses a partition at random

    :param n_contexts: number of contexts
    :param random_state: random seed state
    :param permute: whether to consider the order of context groups
    :param k_min: min number of groups
    :param k_max: max number of groups
    :param single_context_groups: partitions are of shape [[0],[1],[2,3,4]] ie all but one have size one
    (application: one observational group, interventional contexts)
    :return:
    """
    #default: a random number of groups
    if k_max is None and k_min is None:
        k_max = n_contexts
        k_min = 1
    #if k_max == k_min == 0:
    #    return[[c_i] for c_i in range(n_contexts)] #special case: "no groups" means singleton groups -> only confusing
    if k_max == n_contexts and k_min == n_contexts:
        return[[c_i] for c_i in range(n_contexts)]

    assert 1 <= k_min <= k_max <= n_contexts
    # make the partitions balanced: choose k at random, rather than one of pi_lst at random
    # (for example, if permuted, there are many different size-4-partitions, only one size-5 partition for 5 contexts)
    if k_min == k_max:
        k = k_min
    else:
        k = random_state.randint(low=k_min, high=k_max+1)
    if single_context_groups:
        #indices of the intervened contexts
        intervened = random_state.randint(low=0, high=n_contexts, size=k)
        Pi = [[c_i] for c_i in intervened]
        Pi = Pi+[[c_i for c_i in range(n_contexts) if c_i not in intervened]]
    else:
        Pi_lst = pi_enum(n_contexts, permute=permute)
        Pi_lst = [pi for pi in Pi_lst if len(pi)==k]
        Pi = Pi_lst[random_state.randint(low=0, high=len(Pi_lst))]
    return Pi


def sample_data_gp(dag, C_n, D_n, node_n, iv_type, iv_targets, seed):
    ddag = rand.directed_erdos(len(dag), 0)
    for i in range(len(dag)):
        for j in range(len(dag)):
            if dag[j][i]  == 1:
                ddag.add_arc(i, j)
    ddag =  rand.rand_weights(ddag)

    np.random.seed(seed)
    rst = np.random.RandomState(seed)

    # Per variable: context groups, observational group, arc weights of groups
    partitions_X = [None for _ in range(node_n)]
    arc_weights_X = [None for _ in range(node_n)]

    # Partition for each node ----------
    # showing in which groups of contexts an intervention takes place
    for node_X in range(len(dag)):
        k = 0
        for targets in iv_targets:
            if node_X in targets:
                k= k+1
        if k == C_n :
            partition_X = [[c_i] for c_i in range(C_n)]
        else:
            partition_X = pi_rand(C_n, np.random.RandomState(cantor_pairing(node_X, seed)),  # the same for all c_i
                              permute=True,
                              k_min=k+1, k_max=k+1,
                              single_context_groups=False)

        partitions_X[node_X] = partition_X


        #arc_weights_X[node_X] = gen_arc_weights_pi(partition_X, dag, seed, iv_type)

    ndag = NonlinearDAG(ddag.nodes, ddag.arc_weights)
    Dc = ndag.sample_data(D_n, C_n, seed, partitions_X, 0, iv_type, iv_type)

    return Dc



class IvType(Enum):
    PARAM_CHANGE = 1
    # noise interventions
    SCALE = 2
    SHIFT = 3
    # constant
    CONST = 4
    GAUSS = 5
    # hidden variables
    CONFOUNDING = 6
    HIDDEN_PARENT = 7

def data_scale(y):
    scaler = preprocessing.StandardScaler().fit(y)
    return(scaler.transform(y))


class NonlinearDAG(GaussDAG):
    def __init__(
            self,
            nodes: List,
            arcs: Union[Set[Tuple[Any, Any]], Dict[Tuple[Any, Any], float]]
    ):
        super(NonlinearDAG, self).__init__(set(nodes), arcs)


    def sample_data(self, D_n: int, C_n: int, seed: int, partitions_X: list,
                    target, ivtype_target: IvType, ivtype_covariates: IvType,
                    SCALE=2., SHIFT=2., scale=True) -> np.array:
        samples = [np.zeros((D_n, len(self._nodes))) for _ in range(C_n)]
        noise = [np.zeros((D_n, len(self._nodes))) for _ in range(C_n)]

        for ix, (bias, var) in enumerate(zip(self._biases, self._variances)):
            for ic in range(C_n):
                #seed_ic = cantor_pairing(ic, ix)
                noise[ic][:, ix] = np.random.RandomState(seed).normal(loc=bias, scale=var ** .5, size=D_n)
        #    noise_shifted[:, ix] = noise[:, ix] + SHIFT
        #    noise_scaled[:, ix] = np.random.RandomState(seed).uniform()

        t = self.topological_sort()

        for node in t:
            ix = self._node2ix[node]
            parents = self._parents[node]
            partition = partitions_X[node]
            group_map = pi_group_map(partition, C_n)
            if node == target:
                ivtype = ivtype_target
            else:
                ivtype = ivtype_covariates
            if ivtype is IvType.PARAM_CHANGE:
                partition_param = partition
            else:
                partition_param = [[c_i for c_i in range(C_n)]]

            if len(parents) != 0:
                parent_ixs = [self._node2ix[p] for p in self._parents[node]]
                for c_i in range(C_n):
                    # A seed that is the same for each group member (and specific to the target variable ix)
                    seed_group = cantor_pairing(group_map[c_i], ix)
                    seed_context = cantor_pairing(c_i, ix)

                    parent_vals = samples[c_i][:, parent_ixs]

                    # Nonlinear function f with additive noise, y=f(X) + N
                    X, yc = gen_data_gp(X=parent_vals, D_n=D_n, C_n=C_n, seed=seed, partition=partition_param,
                                        kernel_function=kernel_rbf)

                    samples[c_i][:, ix] = yc[c_i] + noise[c_i][:, ix]

                    # Keep the observational group of contexts (index 0) as is
                    if group_map[c_i] > 0:

                        # Apply an intervention to all interventional groups of contexts (index > 0)
                        if ivtype is IvType.SCALE:
                            samples[c_i][:, ix] = yc[c_i] + noise[c_i][:, ix] * SCALE #SCALE * np.random.RandomState(seed).uniform(-1,1, size=D_n)
                        if ivtype is IvType.SHIFT:
                            samples[c_i][:, ix] = yc[c_i] + noise[c_i][:, ix] + SHIFT
                        if ivtype is IvType.CONST:
                            CONST = np.random.RandomState(seed_group).randint(0, 5, 1)
                            samples[c_i][:, ix] = CONST + np.random.RandomState(seed_context).uniform(-.5,.5, size=D_n)
            # No causal parents
            else:
                for ic in range(C_n):
                    seed_context = cantor_pairing(ic, ix)
                    #noise = np.random.RandomState(seed_context).normal(loc=LOC, scale=VAR ** .5, size=D_n)
                    samples[ic][:, ix] = noise[ic][:, ix]

        if scale:
                for ic in range(C_n):
                    samples[ic] = data_scale(samples[ic])
        return samples


def kernel_rbf(X1, X2, length_scale=1):
    return RBF(length_scale=length_scale).__call__(X1, eval_gradient=False)
def gen_data_gp_example(seed=1):
    domain_min=-10
    domain_max=10
    D_n = 50
    C_n = 5
    X = np.expand_dims(np.linspace(domain_min, domain_max, D_n), 1)
    gen_data_gp(X, D_n, C_n, seed,  show = True)

def gen_data_gp(X, D_n, C_n, seed,
                partition = [[0,1], [2,3,4]],
                domain_min=-10, domain_max=10, group_variance=0.2,
                kernel_function=kernel_rbf, #kernel_exponentiated_quadratic,
                show=False):
    """ draw samples from a Gaussian Processs

    :param D_n: number of samples
    :param seed: random seed
    :param f_n: number of functions drawn
    :param domain_min: domain X
    :param domain_max: domain X
    :param kernel_function: kernel
    :param show: plotting
    :return:
    """

    E = kernel_function(X, X)
    f_n = len(partition)
    ys = np.random.RandomState(seed).multivariate_normal(
        mean=np.zeros(D_n), cov=E,
        size=f_n)
    y = np.empty((C_n, D_n))
    for pi_k in range(len(partition)):
        for c_i in partition[pi_k]:
            y[c_i, :] = ys[pi_k, :] + np.random.RandomState(cantor_pairing(seed,c_i)).normal(scale=group_variance,size=D_n) #noise[pi_k, :]#
    if show:
        pass
    return X, y


def pi_group_map(Pi, C_n):
    """
            Converts partition as list-of-context-lists to list-of-groupindices.
            Parameters
            ----------
            Pi:
                Partition
            Returns
            -------
            (ncontexts)
                list of group indices
            Examples
            --------
                pi_group_map([[0,1],[2,3,4]], 5) = [0,0,1,1,1]."""
    group_map = [0 for _ in range(C_n)]
    for pi_k in range(len(Pi)):
        for c_j in Pi[pi_k]:
            group_map[c_j] = pi_k
    return (group_map)
