from types import SimpleNamespace

import causaldag as cd
from graphical_models import DAG, GaussDAG
from numpy.random import SeedSequence

from exp.expoptions import ExpOptions
from exp.exptypes import ExpType, FunType, NoiseType, InterventionType, DataSimulator
from exp.gen import gen_data
from exp.methods import LincQFFMethod, LincMethod
from exp.util.sample_ours import gen_context_data


def gen_demo_example() -> [SimpleNamespace, SimpleNamespace]:
    f = FunType.TAYLOR
    dsim = DataSimulator.SERGIO
    params = {"C": 2, "N": 3, "S": 500, "I": 1, "F": f.to_fun()}
    options = gen_demo_options(params["C"], params["N"], params["S"], f, dsim)
    data, truths = gen_data(options, params, SeedSequence(options.seed), options.seed)
    return data, truths, params, options


def gen_bivariate_example() -> [SimpleNamespace, SimpleNamespace]:
    f = FunType.TAYLOR
    params = {"C": 2, "N": 2, "S": 500, "I": 1, "F": f.to_fun()}
    options = gen_demo_options(params["C"], params["N"], params["S"], f)

    arcs: DAG = cd.rand.directed_erdos((params["N"]), 0)
    weights: GaussDAG = cd.rand.rand_weights(arcs)
    weights.set_arc_weight(0, 1, 1)
    data_c, dag, weights, is_true_edge = gen_context_data(
        options, params, SeedSequence(options.seed).spawn(100), given_dag=weights
    )
    data = SimpleNamespace(data_c=data_c, sim=DataSimulator.OURS)
    truths = SimpleNamespace(dags=weights, dag=dag, is_true_edge=is_true_edge)

    return data, truths, params, options


def gen_demo_options(C, N, S, F, dsim=DataSimulator.OURS) -> ExpOptions:

    import logging

    logging.basicConfig()
    log = logging.getLogger("DEMO")
    log.setLevel("INFO")

    options = ExpOptions(
        exp_type=ExpType.MEC,
        data_simulator=dsim,
        methods=[LincMethod, LincQFFMethod],
        functions_F=[F],
        contexts_C=[C],
        nodes_N=[N],
        samples_S=[S],
        interventions_I=[1],
        noise_dist=NoiseType.GAUSS,
        iv_type=InterventionType.CoefficientChange,
        n_components=100,
        reps=3,
        quick=False,
        enable_SID_call=False,
        n_jobs=0,
        logger=log,
        seed=42,
        verbosity=0,
    )
    fh = logging.FileHandler(f"demo.log")
    fh.setLevel(logging.INFO)
    options.logger.addHandler(fh)
    options.out_dir = ""
    return options
