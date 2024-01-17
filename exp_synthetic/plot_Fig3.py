### These experiments are adapted from the Sparse Shift implementation (see LICENSE)

#!/usr/bin/env python
# coding: utf-8


import os
import tkinter
import matplotlib

from exp_synthetic.util_plot_preprocess import plot_preprocess
from exp_synthetic.settings import ALL_C, ALL_n, ALL_s, BASE_s, BASE_d, BASE_p, BASE_n, BASE_C, ALL_p, BASE_s_C
from exp_synthetic.util_tex_plots import tex_plot_robustness, tex_print_robustness
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



RESULTS_DIR = '../results'

RESULTS_DIR = '../plots_paper/results'

df = pd.read_csv(f'{RESULTS_DIR}/plot_fig2a.csv', sep=',', engine='python')

plot_df, x_var_rename_dict = plot_preprocess(df)

metrics = ['F1']
methods = [ 'LINC_gp' ,  'LINC_rff', 'MSS_KCI',  r'MC'  , 'Pooled_PC_KCI']

base_settings={
           # 'n_total_environments': BASE_C,
            '# Environments': BASE_C,
            '# Variables': BASE_n,
            '# Samples': BASE_d,
            'Edge density': BASE_p,
        #'sparsity': BASE_s, #Shift fraction': [1/6], #for 6 variables, 1 on avg. changes
        #    'data_simulator':  ['cdnod'],
}

ALL_IV_settings =  ['cdnod', 'funcf-iv', 'cf-iv', 'scale-iv', 'shift-iv', 'hard-iv']
BASE_IV_setting =  ['cdnod']

settings_plot_test={

'Environments': [
        [{'# Environments': [3], 'data_simulator': ['cdnod'],
          }, 'IID'],
        [{'# Environments': [15], 'data_simulator': ['cdnod']
          }, 'nIID'], ],
}
settings_plot={
'BeyondSparseShifts-Allinterventions': [
        [{'sparsity': [0], 'data_simulator': ALL_IV_settings,
          }, 'IID'],
        [{'sparsity': [6], 'data_simulator': ALL_IV_settings
          }, 'nIID'], ],
'BeyondSparseShifts-Softinterventions': [
        [{'sparsity': [0], 'data_simulator':  BASE_IV_setting,
          }, 'IID'],
        [{'sparsity': [6], 'data_simulator': BASE_IV_setting,
          }, 'nIID'], ],
'InterventionTypes-hard': [
        [{
            'data_simulator': ['cdnod'],
            }, 'Base'],
        [{
            'data_simulator': ['hard-iv'],
            }, 'IvHard'], ],
    'InterventionTypes-cf': [
        [{
            'data_simulator': ['funcf-iv', 'cdnod', 'cf-iv'], 'sparsity':[1/3]
        }, 'IvFunCf'],
        [{
            'data_simulator': ['cf-iv'], 'sparsity':[1/3]
        }, 'IvCf'], ],
    'InterventionTypes-noise': [
        [{
            'data_simulator': ['scale-iv', 'scaling-iv'], #'sparsity': [6]
        }, 'IvScale'],
        [{
            'data_simulator': ['shift-iv'] , 'sparsity': [1]
        }, 'IvShift'], ],
    'Bivariate-IID/nIID': [
        [{'sparsity': [0],   'data_simulator': ['cdnod'],
          '# Variables': [2]}, 'Bivariate-IID'],
        [{'sparsity': [1],# 'data_simulator': ['cdnod'],
          '# Variables': [2]}, 'Bivariate-nIID'], ],
'BeyondSparseShifts-OneC': [
        [{'# Environments': [1], #'n_total_environments':[1],
          #'sparsity' : [1/3], #  BASE_IV_setting,
          }, 'IID'],
        [{'# Environments': [5], 'n_total_environments':[5],
          }, 'nIID'], ],
}
#INCL_CDNOD=False
settings_other={
    #'Functions-nonlinear': [
    #    [{'data_simulator': ['cdnod'], }, 'cdnod-mss-paper'],
    #    [{'data_simulator': ['scale-iv'], }, 'cdnod-full'], ],
    #'Functions': [
    #    [{'data_simulator': ['gp'], }, 'gp'],
    #    [{'data_simulator': ['lin-gauss'], }, 'lin-uniform'], ],
    #'NoiseTypes': [
    #    [{'data_simulator': ['cdnod'], }, 'uniform'],
    #    [{'data_simulator': ['cdnod-noise-gaussian'], }, 'gaussian'], ],
    'NoiseTypesLinear': [
        [{'data_simulator': ['ling'], 'sparsity':[1/3] }, 'Linear gaussian'],
        [{'data_simulator': ['ling'], 'sparsity':[6]  }, 'Linear gaussian'], ],
    'InterventionTypes-hard': [
        [{
            'data_simulator': ['cdnod'],
            }, 'Base'],
        [{
            'data_simulator': ['hard-iv'],
            }, 'IvHard'], ],
    'InterventionTypes-cf': [
        [{
            'data_simulator': ['funcf-iv'], 'sparsity':[1/3]
        }, 'IvFunCf'],
        [{
            'data_simulator': ['cf-iv'], 'sparsity':[1/3]
        }, 'IvCf'], ],
    'InterventionTypes-noise': [
        [{
            'data_simulator': ['shift-iv'], 'sparsity':[1/3]
        }, 'IvShift'],
        [{
            'data_simulator': ['scaling-iv'], 'sparsity':[1/3]
        }, 'IvScaling'], ],
    'Mvariate-IID/nIID': [
        [{
            'sparsity': [0],
            '# Variables' : [6]}, 'Multivariate-IID'],
        [{  '# Variables' : [6]}, 'Multivariate-nIID'], ],
    'Bivariate-IID/nIID': [
        [{ 'sparsity': [0],
            '# Variables' : [2]}, 'Bivariate-IID'],
        [{ 'sparsity': [1],
            '# Variables' : [2]}, 'Bivariate-nIID'],],
    'Sparsity-funcf': [
        [{
            'data_simulator': ['funcf-iv'], 'sparsity':[1/3]
        }, 'IvFunCf (1/3)'],
        [{
            'data_simulator': ['funcf-iv'],'sparsity':[6]
        }, 'IvFunCf (6)'], ],
    'Sparsity-cf': [
        [{
            'data_simulator': ['cf-iv'], 'sparsity': [1 / 3]
        }, 'IvCf (1/3)'],
        [{
            'data_simulator': ['cf-iv'], 'sparsity': [6]
        }, 'IvCf (6)'], ],
    'Sparsity-scaling': [
        [{
            'data_simulator': ['scaling-iv'], 'sparsity': [1]
        }, 'IvScale (1/6)'],
        [{
            'data_simulator': ['funcf-iv'], 'sparsity': [0]
        }, 'Iv  (0)'], ],
    'Sparsity-shift': [
        [{
            'data_simulator': ['shift-iv'], 'sparsity': [1 / 3]
        }, 'IvShift (1/3)'],
        [{
            'data_simulator': ['shift-iv'], 'sparsity': [6]
        }, 'IvShift (6)'], ],
    'Sparsity-hard': [
        [{
            'data_simulator': ['hard-iv'], 'sparsity': [0]
        }, 'IvHard (1/3)'],
        [{
            'data_simulator': ['hard-iv'], 'sparsity': [6]
        }, 'IvHard (6)'], ],
        #First exp:

}

for metric in metrics:
    for name in settings_plot: #name, ax in zip(settings_plot, row):
            two_settings = settings_plot[name]
            plot_df_1 = plot_df
            plot_df_2 = plot_df
            columns_1, name_setting_1 = two_settings[0]
            columns_2, name_setting_2 = two_settings[1]
            for col in columns_1:
                plot_df_1= plot_df_1.loc[plot_df_1[col].isin(columns_1[col])]
            for col in columns_2:
                plot_df_2= plot_df_2.loc[plot_df_2[col].isin(columns_2[col])]

            for base in base_settings:
                if not base in columns_1:
                    plot_df_1 = plot_df_1.loc[plot_df_1[base].isin(base_settings[base])]
                if not base in columns_2:
                    plot_df_2 = plot_df_2.loc[plot_df_2[base].isin(base_settings[base])]
            plot_df_1= plot_df_1[ # IMPORTANT! otherwise average over all number of environments
                plot_df_1['# Environments'] == plot_df_1['# Environments'].max()]
            plot_df_2= plot_df_2[  plot_df_2['# Environments'] == plot_df_2['# Environments'].max()]

            # Create the tex files
            tex_plot_robustness(plot_df_1, plot_df_2, name_setting_1, name_setting_2, name, methods, metric, outfile='./plots_output/tex_fig3')

            # Print an overview
            tex_print_robustness(plot_df_1, plot_df_2, name_setting_1, name_setting_2, name, methods, metric )