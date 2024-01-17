from graphical_models import GaussDAG
import numpy as np
from typing import List, Union, Set, Tuple, Dict, Any

from function_types import gen_data_gp, kernel_rbf
from intervention_types import IvType
from utils import cantor_pairing, data_scale
from utils_pi import pi_group_map


class NonlinearDAG(GaussDAG):
    """ Causal DAG with nonlinear functional dependencies, random GP paths w/ additive noise.
    """
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
                noise[ic][:, ix] = np.random.RandomState(seed).normal(loc=bias, scale=var ** .5, size=D_n)

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
                            samples[c_i][:, ix] = yc[c_i] + noise[c_i][:, ix] * SCALE
                        if ivtype is IvType.SHIFT:
                            samples[c_i][:, ix] = yc[c_i] + noise[c_i][:, ix] + SHIFT
                        if ivtype is IvType.CONST:
                            CONST = np.random.RandomState(seed_group).randint(0, 5, 1)
                            samples[c_i][:, ix] = CONST + np.random.RandomState(seed_context).uniform(-.5,.5, size=D_n)
            # No causal parents
            else:
                for ic in range(C_n):
                    samples[ic][:, ix] = noise[ic][:, ix]

        if scale:
                for ic in range(C_n):
                    samples[ic] = data_scale(samples[ic])
        return samples

#example
#dag = rand.directed_erdos(nnodes=5, density=.5)
#Gdag = rand.rand_weights(dag)
#Ndag=  NonlinearDAG(dag.nodes, dag.arcs)
#Dc = Ndag.sample(50, 1)