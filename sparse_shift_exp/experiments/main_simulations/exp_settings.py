from sparse_shift.methods import MinChange, AugmentedPC, FullMinChanges, ParamChanges



# The base settings used in the LINC paper.
BASE_C = [5]
BASE_C_dag = [3]
BASE_s = [1]
BASE_s_C = [1/3] #important so that comparable shift fraction is used, not a fixed one.
BASE_n = [6]
BASE_p = [0.3]
BASE_d = [500]
BASE_rep_large = [50]
BASE_rep = [5]
BASE_sim_data= ['cdnod']
BASE_sim_dag= ['er']

# Varying one parameter at a time while keeping base settings fixed
ALL_C = [3,5,8,10]# [15] - no need to include smaller numbers as contexts added one at a time(?)
VARY_C = ALL_C

ALL_s = [x for x in [0,1,2,3,4,5,6]] #for 6 variables.
VARY_s = [x for x in ALL_s]

ALL_n =  [x for x in [2,3,4,5,6,9,12,15]]
VARY_n = [x for x in ALL_n if not (x in BASE_n)]

ALL_p =  [x for x in [0.3,0.5,0.7]]
VARY_p = [x for x in ALL_p if not (x in BASE_p)]
ALL_d = [x for x in [50,100,200,750,1000]]
VARY_d = [x for x in ALL_d if not (x in BASE_d)]


PARAMS_DICT = {
    "mec":[  {
                    "sparsity": [1 / 3],
                    "sample_size": BASE_d,
                    "dag_density": BASE_p,
                    "experiment": ["vars"],
                    "n_variables": [3],
                    "n_total_environments": [50],
                    'intervention_targets': [None],
                    "reps": BASE_rep,
                    "data_simulator": BASE_sim_data,
                    "dag_simulator": BASE_sim_dag
                },
        '''{
                   "sparsity": [1 / 3],
                   "sample_size": BASE_d,
                   "dag_density": BASE_p,
                   "experiment": ["vars"],
                   "n_variables": [5, 10],
                   "n_total_environments": [3],# [5, 10, 20, 50],  # [50],
                   'intervention_targets': [None],
                   "reps": [5],
                   "data_simulator": ['gp'], # BASE_sim_data,
                   "dag_simulator": BASE_sim_dag
               }, 
                        {
                            "sparsity": [1 / 3],
                            "sample_size": BASE_d,
                            "dag_density": BASE_p,
                            "experiment": ["vars"],
                            "n_variables": [3],
                            "n_total_environments": [5, 10, 20, 50],  # [50],
                            'intervention_targets': [None],
                            "reps": BASE_rep,
                            "data_simulator": BASE_sim_data,
                            "dag_simulator": BASE_sim_dag
                        },
                                {
                                    "sparsity": [1 / 3],
                                    "sample_size": BASE_d,
                                    "dag_density": BASE_p,
                                    "experiment": ["vars"],
                                    "n_variables": [30, 50],  # BASE_n,
                                    "n_total_environments": [5],# [50],
                                    'intervention_targets': [None],
                                    "reps": BASE_rep,
                                    "data_simulator": ['gp'],  # BASE_sim_data,
                                    "dag_simulator": BASE_sim_dag
                                },
                        {
                                    "sparsity": [ 1],
                                    "sample_size": BASE_d,
                                    "dag_density": BASE_p,
                                    "experiment": ["iv"],
                                    "n_variables": [2],
                                    "n_total_environments": BASE_C,
                                    'intervention_targets': [None],
                                    "reps": BASE_rep,
                                    "data_simulator": ['hard-iv'],  # BASE_sim_data,
                                    "dag_simulator": BASE_sim_dag
                                },
                        {
                                    "sparsity": [0,1],
                                    "sample_size": BASE_d,
                                    "dag_density": BASE_p,
                                    "experiment": ["iv"],
                                    "n_variables": [2],
                                    "n_total_environments": BASE_C,
                                    'intervention_targets': [None],
                                    "reps": BASE_rep,
                                    "data_simulator": ['funcf-iv'],  # BASE_sim_data,
                                    "dag_simulator": BASE_sim_dag
                                },
                        {
                                    "sparsity": [1],
                                    "sample_size": BASE_d,
                                    "dag_density": BASE_p,
                                    "experiment": ["iv"],
                                    "n_variables": [2],
                                    "n_total_environments": BASE_C,
                                    'intervention_targets': [None],
                                    "reps": BASE_rep,
                                    "data_simulator": ['cf-iv'],  # BASE_sim_data,
                                    "dag_simulator": BASE_sim_dag
                                },
                                #Interventions
                        {
                                    "sparsity": [0,1/3,6],
                                    "sample_size": BASE_d,
                                    "dag_density": BASE_p,
                                    "experiment": ["iv"],
                                    "n_variables": BASE_n,
                                    "n_total_environments": BASE_C,
                                    'intervention_targets': [None],
                                    "reps": BASE_rep,
                                    "data_simulator": ['funcf-iv'],  # BASE_sim_data,
                                    "dag_simulator": BASE_sim_dag
                                },
                        {
                                    "sparsity": [1/3,6],
                                    "sample_size": BASE_d,
                                    "dag_density": BASE_p,
                                    "experiment": ["iv"],
                                    "n_variables": BASE_n,
                                    "n_total_environments": BASE_C,
                                    'intervention_targets': [None],
                                    "reps": BASE_rep,
                                    "data_simulator": ['cf-iv'],  # BASE_sim_data,
                                    "dag_simulator": BASE_sim_dag
                                },
                        {
                                    "sparsity": [1/3,6],
                                    "sample_size": BASE_d,
                                    "dag_density": BASE_p,
                                    "experiment": ["iv"],
                                    "n_variables": BASE_n,
                                    "n_total_environments": BASE_C,
                                    'intervention_targets': [None],
                                    "reps": BASE_rep,
                                    "data_simulator": ['shift-iv'],  # BASE_sim_data,
                                    "dag_simulator": BASE_sim_dag
                                },
                        #Linear Gaussian
                                {"sparsity": [1/3,6],
                                 "sample_size": BASE_d,
                                 "dag_density": BASE_p,
                                 "experiment": ["linear-gaussian"],
                                 "n_variables": BASE_n,
                                 "n_total_environments":   BASE_C,
                                 'intervention_targets': [None],
                                 "reps": BASE_rep,
                                 "data_simulator": ["ling"],
                                 "dag_simulator": BASE_sim_dag
                                 },
                                        {"sparsity": BASE_s,
                                         "sample_size": BASE_d,
                                         "dag_density": BASE_p,
                                         "experiment": ["lingauss"],
                                         "n_variables": BASE_n,
                                         "n_total_environments": [5],  # BASE_C,
                                         'intervention_targets': [None],
                                         "reps": BASE_rep,
                                         "data_simulator": ['coef-iv'],
                                         "dag_simulator": BASE_sim_dag
                                         },
                                        {"sparsity": BASE_s,
                                         "sample_size": BASE_d,
                                         "dag_density": BASE_p,
                                         "experiment": ["lingauss"],
                                         "n_variables": BASE_n,
                                         "n_total_environments": [5],  # BASE_C,
                                         'intervention_targets': [None],
                                         "reps": BASE_rep,
                                         "data_simulator": ['scaling-iv'],
                                         "dag_simulator": BASE_sim_dag
                                         },
                                        {
                                            "sparsity": VARY_s,
                                            "sample_size": BASE_d,
                                            "dag_density": BASE_p,
                                            "experiment": ["spars"],
                                            "n_variables": [2],
                                            "n_total_environments": BASE_C,
                                            'intervention_targets': [None],
                                            "reps": BASE_rep,
                                            "data_simulator": ['cdnod'],  # BASE_sim_data,
                                            "dag_simulator": BASE_sim_dag
                                        },
                                        {
                                            "sparsity": VARY_s,
                                            "sample_size": BASE_d,
                                            "dag_density": BASE_p,
                                            "experiment": ["spars"],
                                            "n_variables": BASE_n,
                                            "n_total_environments": BASE_C,
                                            'intervention_targets': [None],
                                            "reps": BASE_rep,
                                            "data_simulator": ['hard-iv'],  # BASE_sim_data,
                                            "dag_simulator": BASE_sim_dag
                                        },
                                                #SERGIO
                                                        {
                                                        "sparsity": BASE_s, #Sparsity does not change for sergio.
                                                        "sample_size": BASE_d, #TODO vary later.
                                                        "dag_density": BASE_p, #TODO vary later.
                                                        "experiment": ["sergio"],
                                                        "n_variables": [-1], #will be changed to C-1 due to diagonal experimental design.
                                                        "n_total_environments": [15], #ALL_C,
                                                        'intervention_targets': [None],
                                                        "reps": BASE_rep,
                                                        "data_simulator": ["sergio"],
                                                        "dag_simulator": BASE_sim_dag
                                                    },
                                        
                                                      
                                                       # in the remaining exps, use vary_x
                                                        {
                                                            "sparsity": BASE_s,
                                                            "sample_size": BASE_d,
                                                            "dag_density": BASE_p,
                                                            "experiment": ["nvar"],
                                                            "n_variables": VARY_n,
                                                            "n_total_environments": BASE_C,
                                                            'intervention_targets': [None],
                                                            "reps": BASE_rep,
                                                            "data_simulator":  ['cdnod'],# BASE_sim_data,
                                                            "dag_simulator": BASE_sim_dag
                                                        },
                                                {
                                                            "sparsity": BASE_s_C,  # important so that comparable shift fraction is used, not a fixed one.
                                                            "sample_size": BASE_d,
                                                            "dag_density": BASE_p,
                                                            "experiment": ["nenv"],
                                                            "n_variables": BASE_n,
                                                            "n_total_environments": ALL_C,
                                                            'intervention_targets': [None],
                                                            "reps": BASE_rep,
                                                            "data_simulator": ['cdnod-d'],# BASE_sim_data,
                                                            "dag_simulator": BASE_sim_dag
                                                        }, 
                                                        
                                                ''',
    ],
    "DEBUG": [{
        "experiment": "debug",
        "n_variables": [5],
        "n_total_environments": [5],
        "sparsity": [3],
        'intervention_targets': [None],
        "sample_size": [500],
        "dag_density": [0.3],
        "reps": [5],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["er"],
    }],
"mec-rebuttal-vars": [
    {
            "experiment": ["n_variables"],
            "n_variables": [30],
            "n_total_environments": [3],
            "sparsity":[1],
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": [5],
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
       ],"mec-rebuttal-samp": [
    {
            "experiment": ["n_samples"],
            "n_variables": [3],
            "n_total_environments": [5], # [3],
            "sparsity":[1],
            'intervention_targets': [None],
            "sample_size":[500, 1000, 1500, 2000, 5000],
            "dag_density": BASE_p,
            "reps": [5],
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
       ],
}
"""
    # "environment_convergence": [{
    #     "n_variables": [6],
    #     "n_total_environments": [10],
    #     "sparsity": [1, 2, 4],
    #     'intervention_targets': [None],
    #     "sample_size": [500],
    #     "dag_density": [0.3],
    #     "reps": [20],
    #     "data_simulator": ["cdnod"],
    #     "dag_simulator": ["er"],
    # }],
    # "soft_samples": [{
    #     "n_variables": [6],
    #     "n_total_environments": [5],
    #     "sparsity": [1, 2, 3, 4, 5, 6],
    #     'intervention_targets': [None],
    #     "sample_size": [50, 100, 200, 300, 500],
    #     "dag_density": [0.3],
    #     "reps": [20],
    #     "data_simulator": ["cdnod"],
    #     "dag_simulator": ["er"],
    # }],
    "oracle_rates": [{
        "n_variables": [4, 6, 8, 10, 12],
        "n_total_environments": [5],
        "sparsity": [1/5, 1/3, 1/2, 2/3, 4/5],
        'intervention_targets': [None],
        "sample_size": [None],
        "dag_density": [0.1, 0.3, 0.5, 0.7, 0.9],
        "reps": [20],
        "data_simulator": [None],
        "dag_simulator": ["er"],
    }],
    "oracle_select_rates": [
        {
            "n_variables": [6, 8, 10, 12],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [1, 2, 3, 4, 5, 6, 7],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [15],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3, 0.5, 0.7, 0.9, 0.1],
            "reps": [20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            # Since 'ba' can't handle all of the same settings as 'er'
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3, 0.5, 0.7, 0.9],
            "reps": [20],
            "data_simulator": [None],
            "dag_simulator": ["ba"],
        },
    ],
    "bivariate_power": [{
        "n_variables": [2],
        "n_total_environments": [2],
        "sparsity": [None],
        'intervention_targets': [
            [[], [0]],
            [[], [1]],
            [[], []],
            [[], [0, 1]],
        ],
        "sample_size": [500],
        "dag_density": [None],
        "reps": [50],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["complete"],
    }],
    "bivariate_multiplic_power": [{
        "n_variables": [2],
        "n_total_environments": [2],
        "sparsity": [None],
        'intervention_targets': [
            [[], [0]],
            [[], [1]],
            [[], []],
            [[], [0, 1]],
        ],
        "sample_size": [500],
        "dag_density": [None],
        "reps": [50],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["complete"],
    }]
}

"""
SUB_METHODS =[
    (
        'mch_kci',
        'Min changes (KCI)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'kci',
            'test_kwargs': {
                "KernelX": "GaussianKernel",
                "KernelY": "GaussianKernel",
                "KernelZ": "GaussianKernel",
            },
        }
    )]
'''
(
        'full_pc_kci',
        'Full PC (KCI)',
        FullMinChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'kci',
            'test_kwargs': {},
        }
    ),
    (
        'mc',
        'MC',
        ParamChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
        }
    )
]'''
# save name, method name, algo, hpyerparams
ALL_METHODS = [
    (
        'mch_kci',
        'Min changes (KCI)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'kci',
            'test_kwargs': {
                "KernelX": "GaussianKernel",
                "KernelY": "GaussianKernel",
                "KernelZ": "GaussianKernel",
            },
        }
    ),
    (
        'mch_lin',
        'Min changes (Linear)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'invariant_residuals',
            'test_kwargs': {'method': 'linear', 'test': "whitney_levene"},
        }
    ),
    (
        'mch_gam',
        'Min changes (GAM)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'invariant_residuals',
            'test_kwargs': {'method': 'gam', 'test': "whitney_levene"},
        }
    ),
    (
        'mch_fisherz',
        'Min changes (FisherZ)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'fisherz',
            'test_kwargs': {},
        }
    ),
    (
        'full_pc_kci',
        'Full PC (KCI)',
        FullMinChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'kci',
            'test_kwargs': {},
        }
    ),
    (
        'mc',
        'MC',
        ParamChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
        }
    )
]

METHODS_DICT = {
    "DEBUG": ALL_METHODS,
    "pairwise_power": ALL_METHODS,
    "mec-sergio": SUB_METHODS,
    "mec": SUB_METHODS,
    "mec-rebuttal-vars": SUB_METHODS,
    "mec-rebuttal-ctx": SUB_METHODS,
    "mec-rebuttal-samp": SUB_METHODS,
    # "environment_convergence": ALL_METHODS,
    # "soft_samples": ALL_METHODS,
    "oracle_rates": [],
    "oracle_select_rates": [],
    "bivariate_power": ALL_METHODS,
    "bivariate_multiplic_power": ALL_METHODS,
    #     (
    #         'mch_kcd',
    #         'KCD',
    #         MinChange,
    #         {
    #             'alpha': 0.05,
    #             'scale_alpha': True,
    #             'test': 'fisherz',
    #             'test_kwargs': {'n_jobs': -2, 'n_reps': 100},
    #         }
    #     ),
    # ],
}

def get_experiment_params(exp):
    return PARAMS_DICT[exp]


def get_param_keys(exp):
    return list(PARAMS_DICT[exp][0].keys())


def get_experiments():
    return list(PARAMS_DICT.keys())


def get_experiment_methods(exp):
    return METHODS_DICT[exp]
