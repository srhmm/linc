from pathlib import Path

import pytest

from gpcd.gp_fourier_features_mdl import FourierType
from gpcd.scoring import GPType, ScoreType, GPParams, score_edge
from gen import gen_bivariate_example


@pytest.fixture()
def get_path():
    def _(file_path: str):
        return Path(__file__).parent / file_path  # .read_text()

    return _


def test_bivariate():
    data, truths, params, options = gen_bivariate_example()
    gp_hyperparams = GPParams(ScoreType.GP, GPType.EXACT, None, None, None)
    cause, effect = (0, 1) if truths.dag[0][1] != 0 else (1, 0)
    causal = score_edge(data.data_c, gp_hyperparams, [cause], effect, None, 0)
    acausal = score_edge(data.data_c, gp_hyperparams, [effect], cause, None, 0)
    assert all([s > 0 for s in [causal, acausal]])


def test_bivariate_qff():
    data, truths, params, options = gen_bivariate_example()
    gp_hyperparams = GPParams(
        ScoreType.GP, GPType.FOURIER, FourierType.QUADRATURE, 100, None
    )
    cause, effect = (0, 1) if truths.dag[0][1] != 0 else (1, 0)
    causal = score_edge(data.data_c, gp_hyperparams, [cause], effect, None, 0)
    acausal = score_edge(data.data_c, gp_hyperparams, [effect], cause, None, 0)
    assert all([s > 0 for s in [causal, acausal]])
