from exp.expoptions import ExpOptions
from exp.exptypes import FunType, ExpType, NoiseType, InterventionType, DataSimulator
from exp.methods import valid_method_types, id_to_method
from exp.run import run

if __name__ == "__main__":
    """
    >>> python run_exp.py --demo
    """
    import sys
    import argparse
    import logging
    from pathlib import Path

    logging.basicConfig()
    log = logging.getLogger("LINC")
    log.setLevel("INFO")

    ap = argparse.ArgumentParser("LINC")

    def enum_help(enm) -> str:
        return ",".join([str(e.value) + "=" + str(e) for e in enm])

    # Experiment Parameters
    ap.add_argument("-e", "--exp", default=0, help=f"{enum_help(ExpType)}", type=int)
    ap.add_argument(
        "-sim", "--sim", default=0, help=f"{enum_help(DataSimulator)}", type=int
    )
    ap.add_argument(
        "-m",
        "--methods",
        nargs="+",
        default=[0],
        help=f"{','.join(map(str, valid_method_types()))}",
        type=int,
    )
    ap.add_argument("-re", "--reps", default=10, type=int)
    ap.add_argument("-sd", "--seed", default=42, type=int)
    ap.add_argument("-nj", "--n_jobs", default=1, type=int)

    # Data generation parameters that one run iterates over
    ap.add_argument(
        "-s", "--samples", default=[500], nargs="+", help="n samples", type=int
    )
    ap.add_argument(
        "-n", "--nodes", default=[4, 5, 10, 15], nargs="+", help="n nodes", type=int
    )
    ap.add_argument(
        "-c",
        "--contexts",
        default=[3, 5, 12, 20],
        nargs="+",
        help="n contexts",
        type=int,
    )
    # unobserved
    ap.add_argument(
        "-f",
        "--functions_F",
        default=[1],
        nargs="+",
        help=f"{enum_help(FunType)}",
        type=int,
    )
    ap.add_argument(
        "-in",
        "--iv_nodes",
        default=[1],
        help="n interv nodes/context, shift sparsity",
        type=int,
    )

    # Data generation parameters held constant in one run
    ap.add_argument(
        "-nd", "--noise_dist", default=0, help=f"{enum_help(NoiseType)}", type=int
    )
    ap.add_argument(
        "-iv", "--iv_type", default=0, help=f"{enum_help(InterventionType)}", type=int
    )

    ap.add_argument("--demo", action="store_true", help="quick demo experiment")

    # Method hyperparameters
    # ap.add_argument(
    #    "-st", "--score_type", default=0, help=f"{enum_help(ScoreType)}", type=int
    # )
    # ap.add_argument(
    #    "-gp", "--gp_type", default=0, help=f"{enum_help(GPType)}", type=int
    # )
    # ap.add_argument(
    #    "-ff", "--fourier_type", default=0, help=f"{enum_help(FourierType)}", type=int
    # )
    ap.add_argument(
        "-ffnc",
        "--n_components",
        default=100,
        help=f"n RFF components or quadrature basis functions",
        type=int,
    )  # todo not used yet

    ap.add_argument(
        "-v",
        "--verbosity",
        help=f"level of detail of steps printed in dag search",
        default=2,
        type=int,
    )

    # Path
    ap.add_argument("-bd", "--base_dir", default="")
    ap.add_argument("-wd", "--write_dir", default="out/")

    argv = sys.argv[1:]
    nmsp = ap.parse_args(argv)

    options = ExpOptions(
        exp_type=ExpType(nmsp.exp),
        data_simulator=DataSimulator(nmsp.sim),
        methods=[id_to_method(m) for m in nmsp.methods],
        functions_F=[FunType(f) for f in nmsp.functions_F],
        contexts_C=nmsp.contexts,
        nodes_N=nmsp.nodes,
        samples_S=nmsp.samples,
        interventions_I=nmsp.iv_nodes,
        noise_dist=NoiseType(nmsp.noise_dist),
        iv_type=InterventionType(nmsp.iv_type),
        # score_type=ScoreType(nmsp.score_type),
        # gp_type=GPType(nmsp.gp_type),
        # fourier_type=FourierType(nmsp.fourier_type),
        n_components=nmsp.n_components,
        logger=log,
        reps=nmsp.reps,
        quick=nmsp.demo,
        enable_SID_call=False,
        n_jobs=nmsp.n_jobs,
        seed=nmsp.seed,
        verbosity=nmsp.verbosity,
    )

    # Logging
    out_dir = nmsp.write_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"{out_dir}run.log")
    fh.setLevel(logging.INFO)
    options.logger.addHandler(fh)
    options.out_dir = out_dir

    import warnings

    warnings.filterwarnings("ignore")

    run(options)
