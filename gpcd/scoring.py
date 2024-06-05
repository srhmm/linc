import logging
from enum import Enum

import numpy as np
from sklearn.gaussian_process.kernels import RBF

from gpcd.gaussian_process_mdl import GaussianProcessMDL
from gpcd.gp_fourier_features_mdl import FourierType, GP_Fourier_Features_MDL
from gpcd.other.util import data_scale


# todo add discrepancy gen.py step/local minimization


class GPType(Enum):
    EXACT = 0
    FOURIER = 1

    def __eq__(self, other):
        return self.value == other.value


class ScoreType(Enum):
    BIC = 0
    GP = 1

    def __eq__(self, other):
        return self.value == other.value


class DataType(Enum):
    CONTINUOUS = 0
    CONT_MCONTEXT = 3

    def __eq__(self, other):
        return self.value == other.value

    def is_multicontext(self):
        return self.value == self.CONT_MCONTEXT.value


# TODO more sophisticated kernel choice and hyperparameter search.
class GPParams:
    """Hyperparameters for GP fitting."""

    def __init__(
        self,
        score_type: ScoreType,
        gp_type: GPType,
        fourier_type: FourierType | None,
        n_qff_basis_functions: int | None,
        n_rff_features: int | None,
        _k=RBF,  # todo kernel types
        alpha=1e-1,
        length_scale=1.0,
        length_scale_bounds=(1e-2, 1e2),
        n_restarts_optimizer=9,
        scale=True,
    ):
        self.score_type = score_type
        self.gp_type = gp_type
        self.fourier_type = fourier_type

        if self.gp_type == GPType.FOURIER:
            assert self.fourier_type is not None
            if self.fourier_type == FourierType.QUADRATURE:
                assert n_qff_basis_functions is not None
            if self.fourier_type == FourierType.RANDOM:
                assert n_rff_features is not None

        self.n_qff_basis_functions = n_qff_basis_functions
        self.n_rff_features = n_rff_features
        self._k = _k
        self.alpha = alpha
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.n_restarts_optimizer = n_restarts_optimizer
        self.kernel = 1 * _k(
            length_scale=length_scale, length_scale_bounds=length_scale_bounds
        )

        self.scale = scale

    def get_model(self):
        if self.gp_type == GPType.EXACT:
            model = GaussianProcessMDL(
                kernel=self.kernel,
                alpha=self.alpha,
                n_restarts_optimizer=self.n_restarts_optimizer,
            )
        else:
            model = GP_Fourier_Features_MDL(
                kernel=self.kernel,
                alpha=self.alpha,
                n_restarts_optimizer=self.n_restarts_optimizer,
            )
            if self.fourier_type == FourierType.QUADRATURE:
                model.set_ps(self.fourier_type, self.n_qff_basis_functions)
            elif self.fourier_type == FourierType.RANDOM:
                model.set_ps(self.fourier_type, self.n_rff_features)
            else:
                raise ValueError("Mismatch GPType FourierType")
        return model


def fit_gp(
    Xtr,
    ytr,
    gp_hyperparams: GPParams,
):
    Xtr, ytr = data_scale(Xtr), (
        data_scale(ytr.reshape(-1, 1)) if gp_hyperparams.scale else (Xtr, ytr)
    )
    gp_model = gp_hyperparams.get_model()
    gp_model.fit(Xtr, ytr)
    return gp_model


def fit_context_gp(X, y, gp_hyperparams: GPParams):
    return [fit_gp(X[c], y[c], gp_hyperparams) for c in range(len(X))]


def score_edge(
    data_c: dict,
    gp_hyperparams: GPParams,
    parents: list[int],
    target: int,
    logger: logging.Logger,
    verbosity=0,
    edge_info="",
) -> int:
    """Scores an edge {X_1, ... X_m} -> X_j
     :param data_c: data in each context
     :param gp_hyperparams: gp score hyperparams
     :param parents: candidate predecessors {X_1, ... X_m} of  X_j
     :param target: target variable X_j
     :param logger: logger
     :param verbosity: verbosity
     :param edge_info: ground truth info if available
    # >>> data, truths, params, options = gen_bivariate_example()
    # >>> gp_hyperparams = GPParams(ScoreType.GP, GPType.EXACT, None, None, None)
    # >>> score01 = score_edge(data.data_c, gp_hyperparams, [0], 1, None, 0)
    # >>> score10 = score_edge(data.data_c, gp_hyperparams, [1], 0, None, 0)
    # >>> assert all([s > 0 for s in [score01, score10]])
    """

    if verbosity > 0:
        logger.info(f"\tEval edge {parents}->{target}\t{edge_info}")

    data_covariates = (
        [data_c[c_i][:, parents] for c_i in range(len(data_c))]
        if len(parents) > 0
        else [
            np.random.normal(size=(len(data_c[c_i][:, [target]]), 1))
            for c_i in range(len(data_c))
        ]
    )
    data_target = [data_c[c_i][:, target] for c_i in range(len(data_c))]

    gps = fit_context_gp(data_covariates, data_target, gp_hyperparams)
    score = sum([float(gps[c_i].mdl_score_ytrain()[0]) for c_i in range(len(data_c))])
    return score
