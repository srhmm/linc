class ExpOptions:
    def __init__(self, **kwargs):
        _allowed_keys = {
            "exp_type",
            "data_simulator",
            "methods",
            "functions_F",
            "contexts_C",
            "nodes_N",
            "samples_S",
            "interventions_I",
            "noise_dist",
            "iv_type",
            "n_components",
            "logger",
            "reps",
            "seed",
            "quick",
            "n_jobs",
            "enable_SID_call",
            "verbosity",
        }
        assert all([k in _allowed_keys for k in kwargs])
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in _allowed_keys)

        self.methods = [
            m for m in self.methods
        ]  # if m.valid_exp_type() == self.exp_type]
        self.functions_F = [f.to_fun() for f in self.functions_F]

        # quick debug experiment
        self.reps = min(self.reps, 3) if self.quick else self.reps
        if self.quick:
            (
                self.contexts_C,
                self.nodes_N,
                self.samples_S,
                self.functions_F,
                self.interventions_I,
            ) = ([2], [2], [500], [self.functions_F[0]], [1])
        self.base_C, self.base_N, self.base_S, self.base_I = (
            self.contexts_C[0],
            self.nodes_N[0],
            self.samples_S[0],
            self.interventions_I[0],
        )
        self.pool_contexts = True

    def get_cases(self):
        fixed = {"C": self.base_C, "N": self.base_N, "S": self.base_S, "I": self.base_I}
        attrs = {
            "C": self.contexts_C,
            "N": self.nodes_N,
            "S": self.samples_S,
            "I": self.interventions_I,
        }

        combos = [
            ({nm: (attrs[nm][i] if nm == fixed_nm else fixed[nm]) for nm in attrs})
            for fixed_nm in fixed
            for i in range(len(attrs[fixed_nm]))
        ]
        test_cases = {  # keep one attribute fixed and get all combos of the others
            f"c_{val['C']}_n_{val['N']}_s_{val['S']}_I_{val['I']}_f_{str(f[0])}": {
                "C": val["C"],
                "S": val["S"],
                "N": val["N"],
                "I": val["I"],
                "F": f,
            }
            for val in combos
            for f in self.functions_F
        }
        test_cases = dict(
            sorted(
                test_cases.items(),
                key=lambda dic: (dic[1]["C"], dic[1]["N"], dic[1]["S"]),
            )
        )  # small runs first
        return test_cases
