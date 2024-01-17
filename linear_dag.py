from graphical_models import GaussDAG
import numpy as np
from typing import List, Union, Set, Tuple, Dict, Any


class LinearDAG(GaussDAG):
    """ Causal DAG with parametric functional dependencies.
    """

    def __init__(
            self,
            nodes: List,
            arcs: Union[Set[Tuple[Any, Any]], Dict[Tuple[Any, Any], float]]
    ):
        super(LinearDAG, self).__init__(set(nodes), arcs)


def sample_linear(self, nsamples: int = 1,
                  biases=None, variances=None) -> np.array:
    if biases is None:
        biases = self._biases
    if variances is None:
        variances = self._variances
    samples = np.zeros((nsamples, len(self._nodes)))
    noise = np.zeros((nsamples, len(self._nodes)))
    for ix, (bias, var) in enumerate(zip(biases, variances)):
        noise[:, ix] = np.random.normal(loc=bias, scale=var ** .5, size=nsamples)
    t = self.topological_sort()
    for node in t:
        ix = self._node2ix[node]
        parents = self._parents[node]
        if len(parents) != 0:
            parent_ixs = [self._node2ix[p] for p in self._parents[node]]
            parent_vals = samples[:, parent_ixs]
            samples[:, ix] = np.sum(parent_vals * self._weight_mat[parent_ixs, node], axis=1) + noise[:, ix]
        else:
            samples[:, ix] = noise[:, ix]
    return samples


def sample_polynomial(self, nsamples: int = 1,
                      biases=None, variances=None) -> np.array:
    if biases is None:
        biases = self._biases
    if variances is None:
        variances = self._variances
    samples = np.zeros((nsamples, len(self._nodes)))
    noise = np.zeros((nsamples, len(self._nodes)))
    for ix, (bias, var) in enumerate(zip(biases, variances)):
        noise[:, ix] = np.random.normal(loc=bias, scale=var ** .5, size=nsamples)
    t = self.topological_sort()
    for node in t:
        ix = self._node2ix[node]
        parents = self._parents[node]
        if len(parents) != 0:
            parent_ixs = [self._node2ix[p] for p in self._parents[node]]
            parent_vals = samples[:, parent_ixs]
            samples[:, ix] = np.sum(parent_vals * self._weight_mat[parent_ixs, node], axis=1) + noise[:, ix]
        else:
            samples[:, ix] = noise[:, ix]
    return samples
