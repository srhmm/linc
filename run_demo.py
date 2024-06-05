from exp.exptypes import ExpType
from exp.methods import LincMethod, LincQFFMethod, LincRFFMethod

if __name__ == "__main__":
    from gpcd.scoring import ScoreType, GPType, GPParams, score_edge
    from testing.gen import (
        gen_bivariate_example,
        gen_demo_example,
        gen_sergio_data,
        gen_multivariate_example,
    )

    data, truths, params, options = gen_bivariate_example()
    gp_hyperparams = GPParams(ScoreType.GP, GPType.EXACT, None, None, None)
    cause, effect = (0, 1) if truths.dag[0][1] != 0 else (1, 0)
    logger = None
    verbosity = 0

    causal = score_edge(data.data_c, gp_hyperparams, [cause], effect, logger, verbosity)
    acausal = score_edge(
        data.data_c, gp_hyperparams, [effect], cause, logger, verbosity
    )
    print(
        f"Causal dir. {causal:.2f} {'<' if causal < acausal else '>'} "
        f"anticausal dir. {acausal:.2f} "
    )

    ### Data simulated with SERGIO
    data, truths, params, options = gen_sergio_data()
    options.exp_type = ExpType.MEC
    print(
        f"SERGIO: {params['C']} experimental conditions over {params['N']} genes\nUnknown knockout interventions:"
        f"{', '.join(['Gene ' + str(tgt[0]) + ' in condition ' + str(i) for i, tgt in enumerate(truths.targets)])} "
    )

    methods = [LincMethod(), LincQFFMethod(), LincRFFMethod()]

    for mthd in methods:
        mthd.fit(data, truths, params, options)

        print(
            f"Result for {mthd.nm()}:\tf1: {mthd.metrics['f1']:.2f}, shd: {mthd.metrics['shd']:.2f}, "
            f"tp: {mthd.metrics['tp']}, fp: {mthd.metrics['fp']}, time: {mthd.fit_times[0]:.2f}"
        )

    ### Combining LINC with greedy DAG search
    data, truths, params, options = gen_multivariate_example()
    options.exp_type = ExpType.DAG
    print(
        f"Exp Settings: {params['C']} contexts with {params['N']} nodes and avg. {params['I']} intervened node(s), "
        f"\nfunctional form: {params['F'][0]}, noise distribution: {options.noise_dist}"
    )
    methods = [LincMethod(), LincQFFMethod(), LincRFFMethod()]

    for mthd in methods:
        mthd.fit(data, truths, params, options)
    mthd = LincMethod()
    mthd.fit(data, truths, params, options)

    print(
        f"Result for {mthd.nm()} with {str(options.exp_type)}:\tf1: {mthd.metrics['f1']:.2f}, "
        f"shd: {mthd.metrics['shd']:.2f}, "
        f"tp: {mthd.metrics['tp']}, fp: {mthd.metrics['fp']}"
    )
