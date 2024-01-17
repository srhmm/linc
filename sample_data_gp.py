from itertools import combinations

import numpy as np
from graphical_models import rand

from nonlinear_dag import NonlinearDAG
from utils_pi import pi_rand


def sample_data_gp(dag, C_n, D_n, node_n, iv_type, iv_targets, seed):
    np.random.seed(seed)

    ddag = rand.directed_erdos(len(dag), 0)
    for i, j in combinations(range(node_n), 2):
        if dag[j][i] == 1:
            ddag.add_arc(i, j)

    ddag = rand.rand_weights(ddag)

    # Sample a partition of contexts for each node
    node_partitions = [None for _ in range(node_n)]

    for node in range(node_n):
        k = 0
        #TODO
        for targets in iv_targets:
            if node in targets:
                k = k+1
        if k == C_n :
            partition = [[c_i] for c_i in range(C_n)]
        else:
            partition = pi_rand(C_n, np.random.RandomState(cantor_pairing(node, seed)),
                                  permute=True,
                                  k_min=k+1,
                                  k_max=k+1)

        node_partitions[node] = partition

    ndag = NonlinearDAG(ddag.nodes, ddag.arc_weights)
    Dc = ndag.sample_data(D_n, C_n, seed, node_partitions, 0, iv_type, iv_type)

    return Dc




def cantor_pairing(x, y):
    return int((x + y) * (x + y + 1) / 2 + y)