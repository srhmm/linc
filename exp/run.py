from collections import defaultdict

from joblib import Parallel, delayed
from numpy.random import SeedSequence

from exp.gen import gen_data


def run(options):
    cases = options.get_cases()
    reslts = [CaseReslts(case) for case in cases]

    for case, res in zip(cases, reslts):
        options.logger.info(f"CASE: {case}")
        # Random seeds for each it
        ss = SeedSequence(options.seed)
        cs = ss.spawn(options.reps)

        params = cases[case]
        run_params_rep = lambda rep: run_params(options, params, case, rep)

        if options.n_jobs > 1:
            metrics = Parallel(n_jobs=options.n_jobs)(
                delayed(run_params_rep)(rep_seed) for rep_seed in enumerate(cs)
            )
        else:
            metrics = [run_params_rep(rep_seed) for rep_seed in enumerate(cs)]
            print(metrics)
        res.add_reps(metrics)
    # todo res.write()


def run_params(options, params, case, rep_seed):
    options.logger.info(
        f"\n*** Rep {rep_seed[0] + 1}/{options.reps}***" f"\n\tParams: {case}"
    )

    # Generate Data
    data, truths = gen_data(options, params, rep_seed[1], options.seed)

    # Run methods
    metrics = defaultdict(tuple)
    for method in options.methods:
        method.fit(method, data, truths, params, options)
        metrics[method.nm()] = (method, method.metrics)
    return metrics


class CaseMethodReslts:
    def __init__(self, case, nm):
        self.case = case
        self.nm = nm
        self.metrics = defaultdict(list)
        self.objects = []

    def add_method_rep(self, obj, metrics):
        assert obj.nm() == self.nm
        for met in metrics:
            self.metrics[met] += [metrics[met]]
        self.objects.append(obj)


class CaseReslts:
    def __init__(self, case):
        self.case = case
        self.method_results = {}

    def add_reps(self, results):
        for rep in range(len(results)):
            for method_nm in results[rep]:
                obj, metrics = results[rep][method_nm]
                self._add_rep(obj, metrics)

    def _add_rep(self, method, metrics):
        nm = method.nm()
        if nm not in self.method_results:
            self.method_results[nm] = CaseMethodReslts(self.case, method.nm())
        self.method_results[nm].add_method_rep(method, metrics)
