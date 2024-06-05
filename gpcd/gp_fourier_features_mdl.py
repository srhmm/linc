from enum import Enum

import numpy as np
import torch
from scipy.linalg import cholesky
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_approximation import RBFSampler

from gpcd.util.embedding import HermiteEmbedding

GPR_CHOLESKY_LOWER = True

"""Fourier feature approximations for Gaussian processes regression with MDL description length"""


# Inherits from GaussianProcessRegressor (sklearn)
# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Modified by: Pete Green <p.l.green@liverpool.ac.uk>
# License: BSD 3 clause
class FourierType(Enum):
    QUADRATURE = 0
    RANDOM = 1

    def __eq__(self, other):
        return self.value == other.value


class GP_Fourier_Features_MDL(GaussianProcessRegressor):
    def set_ps(self, fourier_type, n_basisf_or_components):
        self.fourier_type = fourier_type
        self.n_basisf_or_components = n_basisf_or_components

    def fit(self, X, y):
        super(GP_Fourier_Features_MDL, self).fit(X, y)

        if self.fourier_type == FourierType.QUADRATURE:
            # self.n_basis_functions = 50
            X = torch.from_numpy(self.X_train_)
            emb = HermiteEmbedding(
                gamma=0.5,
                m=self.n_basisf_or_components,
                d=self.X_train_.shape[1],
                groups=None,
                approx="hermite",
            )  # Squared exponential with lenghtscale 0.5 with m basis functions
            Phi = emb.embed(X.double())
            K = torch.t(Phi) @ Phi
            # todo use torch throughout instead
            K = K.detach().cpu().numpy()
            Phi = Phi.detach().cpu().numpy()

        elif self.fourier_type == FourierType.RANDOM:
            # self.n_components = 100
            rbf_feature = RBFSampler(
                gamma=1, random_state=1, n_components=self.n_basisf_or_components
            )
            Phi = rbf_feature.fit_transform(self.X_train_)
            K = Phi.T @ Phi

        K[np.diag_indices_from(K)] += 1e-1
        mat = np.eye(K.shape[0]) + K * 1**-2

        alpha = np.linalg.solve(mat, Phi.T @ y)[:, 0]
        mdl_model_train = alpha @ mat @ alpha.T

        noise = 1
        mean_pred = noise**-2 * Phi @ np.linalg.solve(mat, Phi.T @ y)[:, 0]

        y_train = y
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]
        var_pred = np.einsum(
            "fn, fn -> n",
            y_train.T,
            np.linalg.solve(mat, Phi.T @ y_train),
        )
        mdl_pen_train = (
            1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0]) + 1**2 * K))
        )

        # plt.scatter(self.X_train_, self.y_train_)
        # plt.scatter(self.X_train_, mean_pred)

        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf

        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        # alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

        # Alg 2.1, page 19, line 7
        # -0.5 . y^T . alpha - sum(log(diag(L))) - n_samples / 2 log(2*pi)
        log_likelihood_dims = -0.5 * (Phi.T @ y).T @ alpha
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(axis=-1)
        mdl_lik_train = -log_likelihood
        mdl_score = mdl_lik_train + mdl_model_train + mdl_pen_train

        self.mdl_lik_train = mdl_lik_train
        self.mdl_model_train = mdl_model_train
        self.mdl_pen_train = mdl_pen_train
        self.mdl_train = mdl_score

    def mdl_score_ytrain(self):
        return (
            self.mdl_train,
            self.mdl_lik_train,
            self.mdl_model_train,
            self.mdl_pen_train,
        )
