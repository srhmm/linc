from sparse_shift import MinChange, FullMinChanges, ParamChanges
from exp_synthetic.methods_linc import LINC

# The base settings used in the LINC paper.
BASE_C = [5]
BASE_C_dag = [3]

BASE_s = [1]
BASE_s_C = [1/3]
BASE_n = [6]
BASE_p = [0.3]
BASE_d = [500]
BASE_rep_large = [100]
BASE_rep = [35]
BASE_sim_data = ['cdnod']
BASE_sim_dag = ['er']

# Varying one parameter at a time while keeping base settings fixed
ALL_C = [3,4,5,8,10,15]
VARY_C = ALL_C

ALL_s = [x for x in [6,0,1,2,3,4,5]]
VARY_s = [x for x in ALL_s]

ALL_n =  [x for x in [2,3,4,5,6,9,12,15]]
VARY_n = [x for x in ALL_n if not (x in BASE_n)]

ALL_p =  [x for x in [0.3,0.5,0.7]]
VARY_p = [x for x in ALL_p if not (x in BASE_p)]

ALL_d = [x for x in [50,100,200,750,1000]]
VARY_d = [x for x in ALL_d if not (x in BASE_d)]


PARAMS_DICT = {
    "mec": [
{
            "sparsity": [1/3],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "experiment": ["vars"],
            "n_variables": [3], #BASE_n,
            "n_total_environments": [5, 10, 20, 30], # [50],
            'intervention_targets': [None],
            "reps": [2],
            "data_simulator": ['gp'], #['cdnod'],  # BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
        '''{
            "sparsity": [0,1],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "experiment": ["iv"],
            "n_variables": [2],
            "n_total_environments": BASE_C,
            'intervention_targets': [None],
            "reps": BASE_rep,
            "data_simulator": ['cdnod'],  # BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
        #UP NEXT:
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
{
            "sparsity": [0, 1/3, 6],
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
            "data_simulator": ['shift-iv'],  # BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
{
            "sparsity": [1/3],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "experiment": ["iv"],
            "n_variables": BASE_n,
            "n_total_environments": BASE_C,
            'intervention_targets': [None],
            "reps": BASE_rep,
            "data_simulator": ['scale-iv'],  # BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
        {"sparsity": [1 / 3, 6 ],
         "sample_size": BASE_d,
         "dag_density": BASE_p,
         "experiment": ["linear-gaussian"],
         "n_variables": BASE_n,
         "n_total_environments": BASE_C,
         'intervention_targets': [None],
         "reps": BASE_rep,
         "data_simulator": ["ling"],
         "dag_simulator": BASE_sim_dag
         },
        {  # SERGIO

            "sparsity": BASE_s,  # Sparsity does not change for sergio.
            "sample_size": BASE_d,  # TODO vary later.
            "dag_density": BASE_p,  # TODO vary later.
            "experiment": ["sergio"],
            "n_variables": [-1],  # will be changed to C-1 due to diagonal experimental design.
            "n_total_environments": [4,6,7,9,15],#ALL_C,
            'intervention_targets': [None],
            "reps": BASE_rep,
            "data_simulator": ["sergio"],
            "dag_simulator": BASE_sim_dag,
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

        #Bivariate, iid and noniid
        {
            "sparsity": [0,1],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "experiment": ["spars"],
            "n_variables": [2],
            "n_total_environments": BASE_C,
            'intervention_targets': [None],
            "reps": BASE_rep,
            "data_simulator": ['cdnod'],  # BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },#Linear Gaussian
        {"sparsity": BASE_s,
         "sample_size": BASE_d,
         "dag_density": BASE_p,
         "experiment": ["linear-gaussian"],
         "n_variables": BASE_n,
         "n_total_environments":   BASE_C,
         'intervention_targets': [None],
         "reps": BASE_rep,
         "data_simulator": ["linear-gaussian"],
         "dag_simulator": BASE_sim_dag
         },
        {"sparsity": BASE_s,
         "sample_size": BASE_d,
         "dag_density": BASE_p,
         "experiment": ["context"],
         "n_variables": BASE_n,
         "n_total_environments":   [15],
         'intervention_targets': [None],
         "reps": BASE_rep,
         "data_simulator": ["cdnod"],
         "dag_simulator": BASE_sim_dag
         }, 
        #Scaling IVs
        {"sparsity": BASE_s,
         "sample_size": BASE_d,
         "dag_density": BASE_p,
         "experiment": ["scale-iv"],
         "n_variables": BASE_n,
         "n_total_environments": BASE_C,
         'intervention_targets': [None],
         "reps": BASE_rep,
         "data_simulator": ['scale-iv'],
         "dag_simulator": BASE_sim_dag
         },
       
        #COEF IV
        {"sparsity": BASE_s,
         "sample_size": BASE_d,
         "dag_density": BASE_p,
         "experiment": ["coef-iv"],
         "n_variables": BASE_n,
         "n_total_environments":  BASE_C,
         'intervention_targets': [None],
         "reps": BASE_rep,
         "data_simulator": ['coef-iv'],
         "dag_simulator": BASE_sim_dag
         },
       #GPs
        {"sparsity": BASE_s,
         "sample_size": BASE_d,
         "dag_density": BASE_p,
         "experiment": ["gp"],
         "n_variables": BASE_n,
         "n_total_environments": [5],#BASE_C,
         'intervention_targets': [None],
         "reps": BASE_rep,
         "data_simulator": ['gp'],
         "dag_simulator": BASE_sim_dag
         },
#RUNNING: denser shifts via 3 variables in 5 contexts
        {"sparsity": [6], # all vars. shift
         "sample_size": BASE_d,
         "dag_density": BASE_p,
         "experiment": ["spars"],
         "n_variables": [6],  # all vars shift
         "n_total_environments": [3], # 3 contexts
         'intervention_targets': [None],
         "reps": BASE_rep,
         "data_simulator": ['cdnod'],  # BASE_sim_data,
         "dag_simulator": BASE_sim_dag
         },
        #TODO: why is rff not working?!
           {"sparsity": [6],  # all vars. shift
         "sample_size": BASE_d,
         "dag_density": BASE_p,
         "experiment": ["spars"],
         "n_variables": BASE_n,
         "n_total_environments": BASE_C,
         'intervention_targets': [None],
         "reps": BASE_rep,
         "data_simulator": ['hard-iv'],   
         "dag_simulator": BASE_sim_dag
         },
#RUNNING:
        {"sparsity": VARY_s,
         "sample_size": BASE_d,
         "dag_density": BASE_p,
         "experiment": ["spars"],
         "n_variables": [2],  # BASE_n,
         "n_total_environments": BASE_C,
         'intervention_targets': [None],
         "reps": BASE_rep,
         "data_simulator": ['cdnod'],  # BASE_sim_data,
         "dag_simulator": BASE_sim_dag
         },
        { "sparsity": VARY_s,
                    "sample_size": BASE_d,
                    "dag_density": BASE_p,
                    "experiment": ["spars"],
                    "n_variables": [6], # BASE_n,
                    "n_total_environments": BASE_C,
                    'intervention_targets': [None],
                    "reps": BASE_rep,
                    "data_simulator":['hard-iv'],  #  BASE_sim_data,
                    "dag_simulator": BASE_sim_dag
                },
      RUNNING : {
                    "sparsity": VARY_s,
                    "sample_size": BASE_d,
                    "dag_density": BASE_p,
                    "experiment": ["spars"],
                    "n_variables": [6], # BASE_n,
                    "n_total_environments": BASE_C,
                    'intervention_targets': [None],
                    "reps": BASE_rep,
                    "data_simulator":['cdnod'],  #  BASE_sim_data,
                    "dag_simulator": BASE_sim_dag
                },
         {
                "sparsity": BASE_s,
                "sample_size": BASE_d,
                "dag_density": BASE_p,
                "experiment": ["nvar"],
                "n_variables": VARY_n,
                "n_total_environments": BASE_C,
                'intervention_targets': [None],
                "reps": BASE_rep,
                "data_simulator":['cdnod-d'],  #  BASE_sim_data,
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
                    "data_simulator": ['cdnod-d'],  # BASE_sim_data,
                    "dag_simulator": BASE_sim_dag
                },
                {
                "sparsity": BASE_s_C, #important so that comparable shift fraction is used, not a fixed one.
                "sample_size": BASE_d,
                "dag_density": BASE_p,
                "experiment": ["nenv"],
                "n_variables": BASE_n,
                "n_total_environments": ALL_C,
                'intervention_targets': [None],
                "reps": BASE_rep,
                "data_simulator":['cdnod-b'], # BASE_sim_data,
                "dag_simulator": BASE_sim_dag
            },
        { #SERGIO
        
                    "sparsity": BASE_s,  # Sparsity does not change for sergio.
                    "sample_size": BASE_d,  # TODO vary later.
                    "dag_density": BASE_p,  # TODO vary later.
                    "experiment": ["sergio"],
                    "n_variables": [-1],  # will be changed to C-1 due to diagonal experimental design.
                    "n_total_environments": ALL_C,
                    'intervention_targets': [None],
                    "reps": BASE_rep,
                    "data_simulator": ["sergio"],
                    "dag_simulator": BASE_sim_dag,
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
                    "data_simulator": BASE_sim_data,
                    "dag_simulator": BASE_sim_dag
                },
         {
                "sparsity": BASE_s,
                "sample_size": BASE_d,
                "dag_density": BASE_p,
                "experiment": ["nvar"],
                "n_variables": VARY_n,
                "n_total_environments": BASE_C,
                'intervention_targets': [None],
                "reps": BASE_rep,
                "data_simulator": BASE_sim_data,
                "dag_simulator": BASE_sim_dag
            },
        #SERGIO
          "sparsity": BASE_s,  # Sparsity does not change for sergio.
          "sample_size": BASE_d,  # TODO vary later.
          "dag_density": BASE_p,  # TODO vary later.
          "experiment": ["sergio"],
          "n_variables": [-1],  # will be changed to C-1 due to diagonal experimental design.
          "n_total_environments": ALL_C,
          'intervention_targets': [None],
          "reps": BASE_rep,
          "data_simulator": ["sergio"],
          "dag_simulator": BASE_sim_dag},
          "mec": [ 
        #IID
        {
          "experiment": ["base"],
          # instead of sparsity (=# intervened variables / # contexts), directly control # contexts where variable X is intervened in:
          "n_changes_Y": [1],
          "n_changes_covariates": [5],
          "n_variables": BASE_n,
          "n_total_environments": BASE_C,
          'intervention_targets': [None],
          "sample_size": BASE_d,
          "dag_density": BASE_p,
          "reps": [50],
          "data_simulator": BASE_sim_data,
          "dag_simulator": BASE_sim_dag
        },  ''',
    ],
    #Discovering the mechanisms for a target node Y (partition context into groups that share a causal mechanism, and telling causal variables from noncausal ones)
"identifiability":[
    {
        "experiment": ["base"],
        # instead of sparsity (=# intervened variables / # contexts), directly control # contexts where variable X is intervened in:
        "n_changes_Y": [1],
        "n_changes_covariates": [5],
        "n_variables": BASE_n,
        "n_total_environments": BASE_C,
        'intervention_targets': [None],
        "sample_size": BASE_d,
        "dag_density": BASE_p,
        "reps": [10],
        "data_simulator": BASE_sim_data,
        "dag_simulator": BASE_sim_dag
    },
],
"mec-rebuttal-ctx": [
    {
            "experiment": ["n_total_environments"],
            "n_variables": [10], # [3],
            "n_total_environments": [50]  ,#BASE_C,[5, 10, 20,30,], #
            "sparsity":[1],
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": [5],
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
       ],
"mec-rebuttal-vars": [
    {
            "experiment": ["n_variables"],
            "n_variables": [30], #[ 26, 30],#  [17, 22, 24, 27, 30], #[15, 20, 25],
            "n_total_environments": [3],#BASE_C,
            "sparsity":[1],
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": [2],
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
       ],"mec-rebuttal-sid": [
    {
            "experiment": ["n_total_environments"], # ["n_variables"],
            "n_variables": BASE_n, # TODO  [5,10],
            "n_total_environments": [1,3,5,8,10], #BASE_C,
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": [10],
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
       ],
#1. Discovering the causal model within a MEC
"mec-prev": [
    {
            "experiment": ["base"], # Run all base parameters once st we can later skip (not vary) them
            "n_variables": BASE_n,
            "n_total_environments": BASE_C,
            "sparsity": BASE_s,
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
    {
            "experiment": ["sparsity"],
            "n_variables": BASE_n,
            "n_total_environments": BASE_C,
            "sparsity": VARY_s,
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
        {
            "experiment": ["dag_density"],
            "n_variables": BASE_n,
            "n_total_environments": BASE_C,
            "sparsity": BASE_s,
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": VARY_p,
            "reps": BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
        {
            "experiment": ["n_variables"],
            "n_variables": VARY_n,
            "n_total_environments": BASE_C,
            "sparsity": BASE_s,
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
        {
        "experiment": ["n_total_environments"],
        "n_variables": BASE_n,
        "n_total_environments": VARY_C,
        "sparsity": BASE_s,
        'intervention_targets': [None],
        "sample_size": BASE_d,
        "dag_density": BASE_p,
        "reps": BASE_rep,
        "data_simulator": ['cdnod'],
        "dag_simulator": ["er"],
    },
    {
        "experiment": ["sample_size"],
        "n_variables": BASE_n,
        "n_total_environments": BASE_C,
        "sparsity": BASE_s,
        'intervention_targets': [None],
        "sample_size": VARY_d,
        "dag_density": BASE_p,
        "reps": BASE_rep,
        "data_simulator": ['cdnod'],
        "dag_simulator": ["er"]
}
],
    #3. IID data
"iid": [
     {
         "experiment": ["sample_size"],
         "n_variables": BASE_n,
         "n_total_environments": [1],
         "sparsity": [0],
         'intervention_targets': [None],
         "sample_size": VARY_d,
         "dag_density": BASE_p,
         "reps": [10],#BASE_rep,
         "data_simulator": BASE_sim_data,
         "dag_simulator": BASE_sim_dag
     },
        # {
        #     "experiment": ["dag_density"],
        #     "n_variables": BASE_n,
        #     "n_total_environments": [1],
        #     "sparsity": [0],
        #     'intervention_targets': [None],
        #     "sample_size": BASE_d,
        #     "dag_density": VARY_p,
        #     "reps": BASE_rep,
        #     "data_simulator": BASE_sim_data,
        #     "dag_simulator": BASE_sim_dag
        # },
        # {
        #     "experiment": ["n_variables"],
        #     "n_variables": VARY_n,
        #     "n_total_environments":[ 1],
        #     "sparsity": [0],
        #     'intervention_targets': [None],
        #     "sample_size": BASE_d,
        #     "dag_density": BASE_p,
        #     "reps": BASE_rep,
        #     "data_simulator": BASE_sim_data,
        #     "dag_simulator": BASE_sim_dag
        # },
],
#3. Discovering DAGs (no MEC known)
"dag": [
    {
            "experiment": ["sparsity"],
            "n_variables": BASE_n,
            "n_total_environments": BASE_C_dag,
            "sparsity": [3,5],
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": [10], #BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
        {
            "experiment": ["dag_density"],
            "n_variables": BASE_n,
            "n_total_environments": BASE_C_dag,
            "sparsity": [0], #BASE_s,
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": VARY_p,
            "reps": [10], #BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
       # {
       #     "experiment": ["n_variables"],
       #     "n_variables": VARY_n,
       #     "n_total_environments": BASE_C_dag,
       #     "sparsity": BASE_s,
       #     'intervention_targets': [None],
       #     "sample_size": BASE_d,
       #     "dag_density": BASE_p,
       #     "reps": BASE_rep,
       #     "data_simulator": BASE_sim_data,
       #     "dag_simulator": BASE_sim_dag
       # },
], "rest": [
        {
        "experiment": ["n_total_environments"],
        "n_variables": BASE_n,
        "n_total_environments": VARY_C,
        "sparsity": BASE_s,
        'intervention_targets': [None],
        "sample_size": BASE_d,
        "dag_density": BASE_p,
        "reps": BASE_rep,
        "data_simulator": ['cdnod'],
        "dag_simulator": ["er"],
    },
    {
        "experiment": ["sample_size"],
        "n_variables": BASE_n,
        "n_total_environments": BASE_C_dag,
        "sparsity": BASE_s,
        'intervention_targets': [None],
        "sample_size": VARY_d,
        "dag_density": BASE_p,
        "reps": BASE_rep,
        "data_simulator": ['cdnod'],
        "dag_simulator": ["er"]
}
]}


# save name, method name, algo, hpyerparams
METHODS_MEC = [
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
    ),
    #(
    #    'globe',
    #    'globe',
    #    GLOBE,{},
    #),
(
        'linc_rff_nogain',
        'linc_rff_nogain',
        LINC,
        {
            'rff': True,
            'mdl_gain': False,
            'ILP': True,
            'known_mec' : True,
            'pi_search' : True
        },
    ),
# (
#         'linc_gp_nogain',
#         'linc_gp_nogain',
#         LINC,
#         {
#             'rff': False,
#             'mdl_gain': False,
#             'ILP': True,
#             'known_mec' : True,
#             'pi_search' : True
#         },
#     ),
]
METHODS_IDENTIFIABILITY_RFF=[(
        'linc_rff_ilp',
        'linc_rff_ilp',
        LINC,
        {
            'rff': True,
            'mdl_gain': False,
            'ILP': True,
            'known_mec' : False,
            'pi_search' : False
        },
    ),
    (
    'linc_rff_full',
    'linc_rff_full',
    LINC,
    {
        'rff': True,
        'mdl_gain': False,
        'ILP': False,
        'known_mec': False,
        'pi_search': False
    },
)
]

METHODS_SACHS = [
    (
        'linc_rff_nogain',
        'linc_rff_nogain',
        LINC,
        {
            'rff': False,
            'mdl_gain': False,
            'ILP': True,
            'clus': False,
            'known_mec': True,
            'pi_search': False
        },
    ),
(
        'mc',
        'MC',
        ParamChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
        }
    ),
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
        'full_pc_kci',
        'Full PC (KCI)',
        FullMinChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'kci',
            'test_kwargs': {},
        }
    )
]
METHODS_OURS=[
    (
    'linc_gp_nogain',
    'linc_gp_nogain',
    LINC,
    {
        'rff': False,
        'mdl_gain': False,
        'ILP': True,
        'clus': False,
        'known_mec': True,
        'pi_search': False
    },
) ]
''',(
    'linc_gp_nogain',
    'linc_gp_nogain',
    LINC,
    {
        'rff': False,
        'mdl_gain': False,
        'ILP': True,
        'clus': False,
        'known_mec': True,
        'pi_search': False
    },
),
    ]'''
METHODS_IDENTIFIABILITY=[(
        'linc_gp_ilp',
        'linc_gp_ilp',
        LINC,
        {
            'rff': False,
            'mdl_gain': False,
            'ILP': True,
            'clus' : False,
            'known_mec' : False,
            'pi_search' : False
        },
)
]
METHODS_REBUTTAL_CTX =[
(
        'linc_rff_ilp',
        'linc_rff_ilp',
        LINC,
        {
            'rff': True,
            'mdl_gain': False,
            'ILP': True,
            'clus' : False,
            'known_mec': True, #important
            'pi_search': True
        },
    )]
''' 
(
        'linc_rff_clus',
        'linc_rff_clus',
        LINC,
        {
            'rff': True,
            'mdl_gain': False,
            'ILP': True,
            'clus' : True,
            'known_mec': True, #important
            'pi_search': True
        },
    ),
 (         'linc_rff_exhaustive',
         'linc_rff_exhaustive',
         LINC,
         {
             'rff': True,
             'mdl_gain': False,
             'ILP': False,
             'known_mec' : True, #important
             'pi_search' : True,
             'clus': False
         },
     )] 
'''
METHODS_REBUTTAL_VARS =[(
        'linc_gp_nogain',
        'linc_gp_nogain',
        LINC,
        {
            'rff': False,
            'mdl_gain': False,
            'ILP': True,
            'clus' : False,
            'known_mec' : True, #important
            'pi_search' : True
        },
    ),
(
        'linc_rff_nogain',
        'linc_rff_nogain',
        LINC,
        {
            'rff': True,
            'mdl_gain': False,
            'ILP': True,
            'clus' : False,
            'known_mec' : True, #important
            'pi_search' : True
        },
    ),
]
METHODS_DAG= [(
        'linc_rff_dag',
        'linc_rff_dag',
        LINC,
        {
            'rff': True,
            'mdl_gain': True,
            'ILP': True,
            'known_mec': False, #important!
            'pi_search': False
        },
    ),
]

METHODS_DICT = {
    #"dag_fast" : METHODS_DAG_FAST,
    "dag": METHODS_DAG,
    "mec": METHODS_OURS,
    "mec-sergio":METHODS_OURS,
    "mec-rebuttal-ctx": METHODS_REBUTTAL_CTX,
    "mec-rebuttal-samp": METHODS_REBUTTAL_VARS,
    "mec-rebuttal-vars": METHODS_REBUTTAL_VARS,
    "mec-rebuttal-sid": METHODS_REBUTTAL_VARS,
    "identifiability": METHODS_IDENTIFIABILITY,
   # "iid": METHODS_C,  #METHODS_MEC,
    "quick": METHODS_MEC,
    #TODO quick
}


def get_experiment_params(exp):
    return PARAMS_DICT[exp]


def get_param_keys(exp):
    return list(PARAMS_DICT[exp][0].keys())


def get_experiments():
    return list(PARAMS_DICT.keys())


def get_experiment_methods(exp):
    return METHODS_DICT[exp]
