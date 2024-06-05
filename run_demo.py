if __name__ == "__main__":
    from gpcd.scoring import ScoreType, GPType, GPParams, score_edge
    from testing.gen import gen_bivariate_example

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
        f"Causal dir. {causal:.2} {'<' if causal < acausal else '>'} "
        f"anticausal dir. {acausal:.2f} "
    )
