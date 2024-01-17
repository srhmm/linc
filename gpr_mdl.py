from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
from scipy.linalg import cholesky
GPR_CHOLESKY_LOWER = True

"""Gaussian processes regression with regularization and MDL description length"""
# Inherits from GaussianProcessRegressor (sklearn)
# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Modified by: Pete Green <p.l.green@liverpool.ac.uk>
# License: BSD 3 clause

class GaussianProcessRegularized(GaussianProcessRegressor):

    def _mdl_score(self, X_test, y_test):

        """ Gaussian Process description length.

        :param X_test: predictors.
        :param y_test: test data.
        """
        # MDL data score/log likelihood
        if y_test.ndim == 1:
            y_test = y_test[:, np.newaxis]

        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_test[:self.alpha_.shape[0]],
                                                        self.alpha_.reshape(1,-1))
        log_likelihood_dims -= np.log(np.diag(self.L_)).sum()
        log_likelihood_dims -= self.K.shape[0] / 2 * np.log(2 * np.pi)
        log_lik = -log_likelihood_dims.sum(axis=-1)

        X_penalty = 1 / 2 * np.log(
            np.linalg.det(np.identity(self.K.shape[0]) + 1 ** 2 * self.K))
        if X_penalty == np.inf:
            X_penalty = 0
        mdl_score = log_lik + self.mdl_model_train + X_penalty

        return mdl_score, log_lik, self.mdl_model_train, X_penalty

    def fit(self, X, y):
        """ Gaussian Process model X->y and description length thereof.

        :param X: predictors.
        :param y: target.
        """
        super(GaussianProcessRegularized, self).fit(X,y)

        # Kernel for X
        kernel = self.kernel_
        K = kernel(self.X_train_, eval_gradient=False)
        self.K = K

        sigma = 1

        # Model Complexity parameter alpha and MDL model score
        K[np.diag_indices_from(K)] += (1e-1)
        mat = np.eye(X.shape[0]) + K *1 ** -2
        alpha =  np.linalg.solve(mat,  y)
        self.alpha_ = alpha
        mdl_model_train = alpha.T @ mat @ alpha

        # MDL data score/log likelihood
        mdl_lik_train = -self.log_marginal_likelihood_value_
        # precompute for self.mdl_score():
        L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        self.L_ = L

        # MDL remaining term, only depends on X
        mdl_pen_train = 1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0]) + sigma**2 * K))
        if mdl_pen_train == np.inf:
            mdl_pen_train = 0


        mdl_train = mdl_lik_train + mdl_model_train + mdl_pen_train

        self.mdl_lik_train = mdl_lik_train
        self.mdl_model_train = mdl_model_train
        self.mdl_pen_train = mdl_pen_train
        self.mdl_train = mdl_train

    def mdl_score_ytrain(self):
        return self.mdl_train, self.mdl_lik_train, self.mdl_model_train, self.mdl_pen_train

    def mdl_score_ytest(self, X_test, y_test):
        mdl, log_lik, m_penalty, X_penalty = self._mdl_score(X_test, y_test)
        return mdl, log_lik, m_penalty, X_penalty
