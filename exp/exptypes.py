from enum import Enum

import numpy as np


class ExpType(Enum):
    MEC = 0
    DAG = 1

    def __eq__(self, other):
        return self.value == other.value


class DataSimulator(Enum):
    OURS = 0
    CDNOD = 1
    SERGIO = 2


class NoiseType(Enum):
    GAUSS = 0
    UNIF = 1

    def to_noise(self):
        vals = [
            (NoiseType.GAUSS, lambda random_state: random_state.standard_normal),
            (NoiseType.UNIF, lambda random_state: random_state.uniform),
        ]
        return vals[self.value]

    def __eq__(self, other):
        return self.value == other.value


class InterventionType(Enum):
    CoefficientChange = 0
    NoiseShift = 1
    NoiseScaling = 2

    def __eq__(self, other):
        return self.value == other.value


class FunType(Enum):
    LIN = 0
    TAYLOR = 1
    NLIN = 2
    TANH = 3
    SINC = 4
    Q = 5
    CUB = 6

    def __str__(self):
        names = ["lin", "taylor", "nlin-exp", "tanh", "sinc", "q", "cub"]
        return names[self.value]

    def to_fun(self):
        vals = [
            (FunType.LIN, lambda x: x),
            (FunType.TAYLOR, lambda x: _taylor_fun(x)),
            (FunType.NLIN, lambda x: x + 5.0 * x**2 * np.exp(-(x**2) / 20.0)),
            (FunType.TANH, lambda x: np.tanh(x)),
            (FunType.SINC, lambda x: np.sinc(x)),
            (FunType.Q, lambda x: x**2),
            (FunType.CUB, lambda x: x**3),
        ]
        return vals[self.value]

    def __eq__(self, other):
        return self.value == other.value


def _taylor_fun(x):
    center = np.zeros(len(x))
    radius = 1
    f = sample_random_multivariate_taylor_series(10, len(x), center, radius)
    return f


def sample_random_multivariate_taylor_series(
    num_terms, num_variables, center, radius, coefficient_range=(-1, 1)
):
    """Sample a random multivariate Taylor series."""
    # Generate random coefficients for each term
    coefficients = np.random.uniform(
        coefficient_range[0], coefficient_range[1], size=(num_terms,)
    )

    # Generate random exponents for each variable in each term
    exponents = np.random.randint(0, 5, size=(num_terms, num_variables))

    # Construct the Taylor series function
    def taylor_series(*args):
        series_sum = 0
        for i in range(num_terms):
            term = coefficients[i]
            for j in range(num_variables):
                term *= (args[j] - center[j]) ** exponents[i, j]
            series_sum += term
        return series_sum

    return taylor_series
