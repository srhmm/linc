### These experiments are adapted from the Sparse Shift implementation (see LICENSE)

#!/usr/bin/env python
# coding: utf-8


# In[1]:
import os
import tkinter

import matplotlib

from exp_synthetic.util_plot_preprocess import plot_preprocess
from exp_synthetic.settings import ALL_C, ALL_n, ALL_s, BASE_s, BASE_d, BASE_p, BASE_n, BASE_C, ALL_p, BASE_s_C
from exp_synthetic.util_tex_plots import tex_plot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


matplotlib.use('TkAgg')

EXPERIMENT = 'scalability'
tag = ''

RESULTS_DIR = '../plots_paper/results'

df = pd.read_csv(f'{RESULTS_DIR}/plot_fig6.csv', sep=',', engine='python')
plot_df, x_var_rename_dict = plot_preprocess(df)

metrics = ['F1', 'RT']

plot_df['F1']=plot_df['F1'].replace(np.nan, 0)
methods = [ 'LINC_gp' ,  'LINC_rff', 'LINC_rff_clus', 'LINC_gp_clus', 'linc_gp_clus', 'linc_gp_nogain', 'MSS_KCI',  'MC', 'Pooled_PC_KCI']
BASE_sim = ['cdnod']
ALL_C = [5, 10 , 20,50]

settings_plot= {'VaryingC':
        [{
            'n_total_environments':   ALL_C,
            '# Variables': [3], #[6],#
            '# Samples': BASE_d,
            'Edge density': BASE_p,
            'sparsity': [ 1/3],
            'data_simulator': BASE_sim,
        }, '# Environments'],
    'VaryingN':
        [{
          'n_total_environments':  [3],
            #'# Variables': ALL_n,
            #'# Samples': BASE_d,
            #'Edge density': BASE_p,
            #'data_simulator': BASE_sim,
        }, '# Variables'],}


ax = 0

fig, axes = plt.subplots(
    len(metrics), 3,
    sharey='row', sharex='col',
    figsize=(1.5*3, 3)
)
identifier=''
if not os.path.exists('../results/'):
    os.makedirs('../results/')

for row, metric in zip(axes, metrics):
    for name, ax in zip( settings_plot, row):

            columns, g_var = settings_plot[name]
            plot_df_ax = plot_df
            for col in columns:
                plot_df_ax = plot_df_ax.loc[plot_df_ax[col].isin(columns[col])]
            #if g_var != '# Environments':
            #    plot_df_ax = plot_df_ax[ # IMPORTANT! otherwise average over all number of environments
            #        plot_df_ax['# Environments'] == plot_df_ax['# Environments'].max()]

            if metric=="RT":
                ax.set( yscale="log")
            plot_df_ax = plot_df_ax[(plot_df_ax[r'$\bf{Test}$'].isin(methods))]

            # Create the tex files
            tex_plot(df=plot_df_ax, x=g_var, y=metric ,identifier='', outfile='./plots_output/tex_fig6')
