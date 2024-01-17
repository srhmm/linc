import itertools
import more_itertools as mit
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def pi_valid(partition: list, C_n):
    """ Ensures valid format of partition.
    :param partition: partition, list of list of context indices.
    :param C_n: num contexts.
    """
    for pi_k in partition:
        for c_i in pi_k:
            assert 0 <= c_i < C_n
    partition0 = [[c_i for c_i in range(C_n)]]
    pi1 = set([c_i for pi_k in partition for c_i in pi_k])
    pi0 = set([c_i for pi_k in partition0 for c_i in pi_k])
    assert len(pi1) == len(pi0)


def pi_matchto_pi(pi_star: list, pi_guess: list, C_n):
    """ Matches two partitions by comparing group membership of each pair of contexts.

    :param pi_star: ground truth
    :param pi_guess: predicted
    :param C_n: number of contexts

    :return (tp, fp, fn, tn, ari, ami)
                tp, fp, fn, tn, adjusted rand score, adjusted mutual information
    """
    pi_valid(pi_star, C_n)
    pi_valid(pi_guess, C_n)
    return pi_matchto_pi_pairwise(pi_star, pi_guess, C_n)


def pi_matchto_pi_exact(pi_star, pi_guess):
    """ Matches partitions exactly.

    :param pi_star: ground truth
    :param pi_guess: predicted
    :param C_n: number of contexts
    :return: match, no match
    """
    pistar = [set(pi_k) for pi_k in pi_star]
    piguess = [set(pi_k) for pi_k in pi_guess]
    match = True

    for pi_k in pistar:
        for pi_j in piguess:
            if len(pi_k & pi_j) > 0:
                match = match and len(pi_k & pi_j) == len(pi_k) == len(pi_j)
    if match:
        return 1, 0
    return 0, 1


def pi_matchto_pi_pairwise(pi_star, pi_guess, C_n):
    """ Matches partitions pairwise.

            Parameters
            ----------
            pi_star:
                ground truth partition.
            pi_guess:
                predicted partition.
            Returns
            -------
            (tp, fp, fn, tn, ari, ami)
                tp, fp, fn, tn, adjusted rand score, adjusted mutual information
            Examples
            --------
            pi_matchto_pair([[0,1],[2]], [[0,1,2]]) = (tp:1,fp:2,fn:0,tn:0, ari:.., ami:..).
            """
    tp, fp, fn, tn = 0, 0, 0, 0
    map_star = pi_group_map(pi_star, C_n)
    map_guess = pi_group_map(pi_guess, C_n)
    assert (len(map_guess) == len(map_star))
    assert (C_n == len(map_guess))
    ari = adjusted_rand_score(map_star, map_guess)
    ami = adjusted_mutual_info_score(map_star, map_guess)

    for i, j in itertools.combinations(range(C_n), 2):
        shared_star = map_star[i] == map_star[j]
        shared_guess = map_guess[i] == map_guess[j]

        if shared_guess:
            if shared_star:
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if shared_star:
                fn = fn + 1
            else:
                tn = tn + 1
    return tp, fp, fn, tn, ari, ami


def pi_convertfrom_pair(vars_ij, contexts_ij, C_n):
    group_list = list()
    for cur_i in range(C_n):
        group_with_i = list()
        for v in vars_ij:
            i, j = contexts_ij[v][0], contexts_ij[v][1]
            if (i is not cur_i and j is not cur_i):
                continue
            if vars_ij[v].varValue == 1:
                if i is cur_i:
                    group_with_i.append(j)
                else:
                    group_with_i.append(i)
        group_with_i.append(cur_i)
        group_with_i = set(group_with_i)
        if group_with_i not in group_list:
            group_list.append(group_with_i)
    return ([[c_i for c_i in pi_k] for pi_k in group_list])


def pi_matchto_pair(pi_star, vars_ij, contexts_ij, C_n):
    pi_guess = pi_convertfrom_pair(vars_ij, contexts_ij, C_n)
    return pi_matchto_pi(pi_star, pi_guess, C_n)


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
    # default: a random number of groups
    if k_max is None and k_min is None:
        k_max = n_contexts
        k_min = 1
    # if k_max == k_min == 0:
    #    return[[c_i] for c_i in range(n_contexts)] #special case: "no groups" means singleton groups -> only confusing
    if k_max == n_contexts and k_min == n_contexts:
        return [[c_i] for c_i in range(n_contexts)]

    assert 1 <= k_min <= k_max <= n_contexts
    # make the partitions balanced: choose k at random, rather than one of pi_lst at random
    # (for example, if permuted, there are many different size-4-partitions, only one size-5 partition for 5 contexts)
    if k_min == k_max:
        k = k_min
    else:
        k = random_state.randint(low=k_min, high=k_max + 1)
    if single_context_groups:
        # indices of the intervened contexts
        intervened = random_state.randint(low=0, high=n_contexts, size=k)
        Pi = [[c_i] for c_i in intervened]
        Pi = Pi + [[c_i for c_i in range(n_contexts) if c_i not in intervened]]
    else:
        Pi_lst = pi_enum(n_contexts, permute=permute)
        Pi_lst = [pi for pi in Pi_lst if len(pi) == k]
        Pi = Pi_lst[random_state.randint(low=0, high=len(Pi_lst))]
    return Pi



def pi_map_to_pi(group_map):
    num_groups = max(group_map) + 1
    pi = [[] for _ in range(num_groups)]
    for i in range(len(group_map)):
        pi[group_map[i]].append(i)
    return pi


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


def pi_enum_split(base_pi):
    partitions = []
    for k in range(len(base_pi)):
        set = base_pi[k]
        if len(set) == 1:
            continue
        modifications = pi_enum(len(set), True, 2, 2)
        for modified_group in modifications:
            new_groups = []
            pi = base_pi.copy()
            for i, subgroup in enumerate(modified_group):
                new_subgroup = []
                for elem in subgroup:
                    new_subgroup.append(set[elem])
                # print(new_subgroup)
                new_groups.append(new_subgroup)
            pi.remove(set)
            for new_group in new_groups:
                pi.append(new_group)
            partitions.append(pi)
    return partitions
