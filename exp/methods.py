import warnings

import cdt.causality.graph as algs
import networkx
import networkx as nx
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc

from exp.methodtypes import ContextMethod, ContinuousMethod
from gpcd.gp_fourier_features_mdl import FourierType
from gpcd.greedy_dag_search import dag_search
from gpcd.scoring import GPParams, ScoreType, GPType


##Multiple Context methods
class LincMethod(ContextMethod):
    @staticmethod
    def nm():
        return "Linc"

    def fit(self, data, truths, params, options):
        super(LincMethod, self).check_data(data, truths, params)

        gp_params: GPParams = GPParams(ScoreType.GP, GPType.EXACT, None, None, None)

        adj, _obj = dag_search(
            options.exp_type,
            truths,
            data.data_c,
            gp_params,
            options.logger,  # todo or dag logger
            options.verbosity,
        )
        self.graph = adj
        self._scores = _obj  # may not contain any edges if it was used in exhaustive search to traverse different dags.
        super(LincMethod, self).eval_results(self, truths, options.logger)


class LincQFFMethod(ContextMethod):
    @staticmethod
    def nm():
        return "LincQFF"

    def fit(self, data, truths, params, options):
        super(LincQFFMethod, self).check_data(data, truths, params)
        gp_params = GPParams(
            ScoreType.GP, GPType.FOURIER, FourierType.QUADRATURE, 100, None
        )
        adj, _obj = dag_search(
            options.exp_type,
            truths,
            data.data_c,
            gp_params,
            options.logger,
            options.verbosity,
        )
        self.graph = adj
        self._scores = _obj  # may not contain any edges if it was used in exhaustive search to traverse different dags.
        super(LincQFFMethod, self).eval_results(self, truths, options.logger)


class LincRFFMethod(ContextMethod):
    @staticmethod
    def nm():
        return "LincRFF"

    def fit(self, data, truths, params, options):
        super(LincRFFMethod, self).check_data(data, truths, params)
        gp_params = GPParams(
            ScoreType.GP, GPType.FOURIER, FourierType.RANDOM, None, 200
        )
        adj, _obj = dag_search(
            options.exp_type,
            truths,
            data.data_c,
            gp_params,
            options.logger,
            options.verbosity,
        )
        self.graph = adj
        self._scores = _obj  # may not contain any edges if it was used in exhaustive search to traverse different dags.
        super(LincRFFMethod, self).eval_results(self, truths, options.logger)


##Continuous methods


class CAMMethod(ContinuousMethod):
    @staticmethod
    def nm():
        return "CAM"

    def fit(self, data, truths, params, options):
        super(CAMMethod, self).check_data(data, truths, params)
        obj = algs.CAM()
        data = pd.DataFrame(data)
        self.untimed_graph = obj.predict(data)
        self.top_order = list(nx.topological_sort(self.untimed_graph))
        super(CAMMethod, self).eval_results(self, truths, options.logger)


class GESMethod(ContinuousMethod):
    @staticmethod
    def nm():
        return "GES"

    def fit(self, data, truths, params, options):
        super(GESMethod, self).check_data(data, truths, params)
        obj = algs.GES()
        data = pd.DataFrame(data)
        self.untimed_graph = obj.predict(data)
        self.top_order = list(nx.topological_sort(self.untimed_graph))
        super(GESMethod, self).eval_results(self, truths, options.logger)


class PCMethod(ContinuousMethod):
    @staticmethod
    def nm():
        return "PC"

    def fit(self, data, truths, params, options):
        super(PCMethod, self).check_data(data, truths, params)

        result_pc = pc(data)
        result_pc.to_nx_graph()
        self.untimed_graph = result_pc.nx_graph
        try:
            self.top_order = list(nx.topological_sort(self.untimed_graph))
        except networkx.exception.NetworkXUnfeasible:
            self.top_order = []
            warnings.warn(f" exception {networkx.exception.NetworkXUnfeasible}")
        super(PCMethod, self).eval_results(self, truths, options.logger)


class GLOBEMethod(ContinuousMethod):
    @staticmethod
    def nm():
        return "GLOBE"

    def fit(self, data, truths, params, options):
        super(GLOBEMethod, self).check_data(data, truths, params)
        obj = None  # GlobeWrapper(2, False, False)
        data = pd.DataFrame(data)
        data.to_csv("temp.csv")
        obj.loadData("temp.csv")
        adjacency = obj.run()
        graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph())
        self.untimed_graph = graph
        self.top_order = list(nx.topological_sort(self.untimed_graph))
        super(GLOBEMethod, self).eval_results(self, truths, options.logger)


##Time Series methods

"""
class VarlingamMethod(TimeMethod):
    @staticmethod
    def nm():
        return "Varlingam"

    def fit(self, data, truths, params, options):
        super(VarlingamMethod, self).check_data(data, truths, params)
        datasets = data.data_c
        D, N = [data_c[k].shape[0] for k in data_c][0], [
            data_c[k].shape[1] for k in datasets
        ][0]
        model = lingam.VARLiNGAM()
        adj_C = []
        data_C = None
        for c in datasets:
            from sklearn import preprocessing

            data_c = datasets[c][
                :, range(N)
            ]  # drop unobserved columns for cnode, snode
            data_c = preprocessing.normalize(data_c)  # otherwise linalg error
            model.fit(data_c)
            adj_c = model.adjacency_matrices_[0]
            adj_C.append(adj_c)
            data_C = data_c if data_C is None else np.vstack((data_C, data_c))
        model.fit(data_C)
        adj_c = model.adjacency_matrices_[0]

        if options.pool_contexts:
            graph = nx.from_numpy_array(adj_c, create_using=nx.DiGraph())
            return graph, model

        f1best = -np.inf
        best = None
        for test_adj in adj_C:
            test_graph = nx.from_numpy_array(test_adj, create_using=nx.DiGraph())
            f1 = directional_f1(truths.dag, test_graph)
            if f1 > f1best:
                f1best = f1
                best = test_adj

        graph = nx.from_numpy_array(best, create_using=nx.DiGraph())

        self.untimed_graph = graph
        super(VarlingamMethod, self).eval_result(self, truths, options.logger)
        return self.metrics
"""


def valid_method_types() -> ContextMethod | ContinuousMethod:
    return [
        LincMethod,
        LincQFFMethod,
        LincRFFMethod,
        CAMMethod,
        GESMethod,
        PCMethod,
        GLOBEMethod,
    ]


def id_to_method(idf: int) -> ContextMethod | ContinuousMethod:
    return valid_method_types()[idf]
