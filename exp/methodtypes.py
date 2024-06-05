from abc import ABC, abstractmethod

import networkx as nx

from exp.util.eval import eval_dag
from gpcd.scoring import DataType


class MethodType(ABC):

    @abstractmethod
    def fit(self, data, truths, params, options):
        pass

    def eval_graph(self, truths, logger, tsp=True):
        true_dag = truths.dag.T if tsp else truths.dag
        self.metrics.update(eval_dag(true_dag, self.graph, len(truths.dag)))
        self.metrics.update(eval_dag(true_dag, self.graph, len(truths.dag)))
        logger.info(
            f"Result for {self.nm()}:\tf1: {self.metrics['f1']:.2f}, shd: {self.metrics['shd']:.2f}, sid: {self.metrics['sid']:.2f}, "
            f"tp: {self.metrics['tp']}, fp: {self.metrics['fp']}"
        )


class ContextMethod(MethodType, ABC):
    def __init__(self):
        self.metrics = {}
        self.graph = nx.DiGraph()
        self.fit_times = []

    @staticmethod
    def check_data(data, truths, params):
        assert len(data.data_c) == params["C"]
        assert all(
            data.data_c[c].shape == (params["S"], params["N"])
            for c in range(params["C"])
        )
        assert len(truths.dag) == params["N"]

    @staticmethod
    def valid_data_type():
        return DataType.CONT_MCONTEXT

    def eval_results(self, truths, logger, tsp=True):
        self.metrics = {}
        super(ContextMethod, self).eval_graph(truths, logger, tsp)


class ContinuousMethod(MethodType, ABC):
    def __init__(self):
        self.metrics = {}
        self.untimed_graph = nx.DiGraph()
        self.top_order = []

    @staticmethod
    def check_data(data, truths, params):
        assert data.shape == (params["S"], params["N"])
        assert len(truths.graph.nodes) == params["N"]

    @staticmethod
    def valid_data_type():
        return DataType.CONTINUOUS

    def eval_results(self, truths, logger):
        self.metrics = {}
        super(ContinuousMethod, self).eval_graph(self, truths, logger)
