import numpy as np
import math
import itertools
from sklearn import preprocessing


def printt(arg, fnm='out.txt'):
    with open(fnm, 'a') as f:
        print(arg, file=f)



def is_insignificant(gain, alpha=0.05):
    return gain < 0 or 2 ** (-gain) > alpha

def cantor_pairing(x, y):
    return int((x + y) * (x + y + 1) / 2 + y)


def logg(val):
    if val == 0:
        return 0
    else:
        return math.log(val)


def data_scale(y):
    scaler = preprocessing.StandardScaler().fit(y)
    return (scaler.transform(y))


#def partition_to_map(part):
#    return partition_to_vector(part)  # pi_group_map(part) # or partition to vec


#def map_to_partition(mp):
#    return pi_map_to_pi(mp)


def map_to_shifts(mp):
    return [1 if x != y else 0 for k, x in enumerate(mp) for y in mp[k + 1:]]


def shifts_to_map(shifts, n_c):
    mp = [0 for _ in range(n_c)]
    for ci in range(n_c):
        cur_idx = mp[ci]
        # assign all pairs (ci, c2) without a mechanism shift to the same group
        for ind, (c1, c2) in enumerate(itertools.combinations(range(n_c), 2)):
            if c1 != ci:
                continue
            if shifts[ind] == 0:
                mp[c2] = cur_idx
            else:
                mp[c2] = cur_idx + 1
    return mp


def pval_to_map(pval_mat, alpha=0.05):
    n_c = pval_mat.shape[0]

    cur_idx = 0
    mp = [0 for _ in range(n_c)]
    n_pairs_changing = [0 for _ in range(n_c)]
    for c1 in range(n_c):
        for c2 in range(n_c):
            if (not (c1 == c2)) and pval_mat[c1][c2] < alpha:
                n_pairs_changing[c1] += 1
    contexts_reordered = np.argsort(n_pairs_changing)
    for i, c1 in enumerate(contexts_reordered):

        # find the largest group of contexts seen so far that c1 can be added to
        cur_idx = mp[c1]
        # assign all pairs (ci, c2) without a mechanism shift to the same group
        for j in range(i, len(contexts_reordered)):  # range(c1, n_c):
            c2 = contexts_reordered[j]
            if c2 == c1:
                continue
            if pval_mat[c1][c2] > alpha:
                mp[c2] = mp[c1]  # cur_idx
            else:
                mp[c2] = cur_idx + 1
        # cur_idx = cur_idx + 1
    return pi_decrease_naming(mp)


def pi_decrease_naming(map):
    nms = np.unique(map)
    renms = [i for i in range(len(nms))]

    nmap = [renms[np.min(np.where(nms == elem))] for elem in map]

    assert (len(np.unique(nmap)) == len(np.unique(map)))
    return nmap


def pi_join(map_1, map_2):
    # All mechanism changes in map_1 and map_2
    map = [map_1[ci] + map_2[ci] * (max(map_1) + 1) for ci in range(len(map_1))]
    return pi_decrease_naming(map)


def data_sub(Dc, node_set):
    return np.array(
        [Dc[c_i][:, node_set] for c_i in range(len(Dc))])  # np.array([Dc[c_i][k, :] for c_i in range(len(Dc))])


#
# def powerset(iterable, emptyset = False):
#     s = list(iterable)
#     if emptyset :
#         return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
#     else:
#         return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def data_groupby_partition(Xc, yc, Pi):
    """Given a partition Pi={pi_k}, pool data within each group pi_k"""
    n_pi = len(Pi)
    n_c = Xc.shape[0]

    assert ((len(Xc.shape) == 3) & (len(yc.shape) == 2))
    assert ((yc.shape[0] == n_c) & (yc.shape[1] == Xc.shape[1]))

    Xpi = [np.concatenate([Xc[c_i] for c_i in Pi[pi_k]]) for pi_k in range(n_pi)]
    ypi = [np.concatenate([yc[c_i] for c_i in Pi[pi_k]]) for pi_k in range(n_pi)]
    return (Xpi, ypi)
