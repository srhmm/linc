### These experiments are adapted from the Sparse Shift implementation (see sparse_shift and LICENSE)

import os

import matplotlib

from exp_synthetic.util_plot_preprocess import plot_preprocess
from exp_synthetic.settings import ALL_C, ALL_n, ALL_s, BASE_s, BASE_d, BASE_p, BASE_n, BASE_C, ALL_p, BASE_s_C, \
    BASE_sim_data, VARY_p, VARY_d
from exp_synthetic.util_tex_plots import tex_plot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

matplotlib.use('TkAgg')

SAVE_FIGURES = False
RESULTS_DIR = '../plots_paper/results'

df = pd.read_csv(f'{RESULTS_DIR}/plot_fig2a.csv', sep=',', engine='python')

plot_df, x_var_rename_dict = plot_preprocess(df)

grid_vars = list(x_var_rename_dict.values())

metrics = [ 'Precision', 'Recall', 'F1']
methods = ['Pooled_PC_KCI', 'MSS_KCI',  'MC',  'LINC_rff',  'LINC_ILP',  'LINC_gp']

settings_plot= {
    'Environments':
        [{
            'n_total_environments':  ALL_C,
            '# Variables': BASE_n,
            '# Samples': BASE_d,
            'Edge density': BASE_p,
            'sparsity': BASE_s_C,
            'data_simulator': BASE_sim_data,
        }, '# Environments'],
    'Variables':
        [{
            'n_total_environments':  BASE_C,
            '# Variables': ALL_n,
            '# Samples': BASE_d,
            'Edge density': BASE_p,
            'sparsity': BASE_s,
            'data_simulator': BASE_sim_data,
        }, '# Variables'],
    'Sparsity':
        [{
            'n_total_environments':BASE_C,
            '# Variables': BASE_n,
            '# Samples': BASE_d,
            'Edge density': BASE_p,
            'sparsity' : [0, 1, 2, 3, 4, 5, 6],
            #'Shift fraction': [0, 1 / 6, 2 / 6, 4 / 6, 5 / 6, 1],
            'data_simulator': BASE_sim_data,
        }, 'Shift fraction'],
    'Density':
        [{
            'n_total_environments':BASE_C,
            '# Variables': BASE_n,
            '# Samples': BASE_d,
            'Edge density': VARY_p,
            'sparsity': BASE_s,
            'data_simulator': BASE_sim_data,
        }, 'Edge density'],
    'Samples':
        [{
            'n_total_environments':BASE_C,
            '# Variables': BASE_n,
            '# Samples': VARY_d,
            'Edge density': BASE_p,
            'sparsity': BASE_s,
            'data_simulator': BASE_sim_data,
        }, '# Samples'],

    }

ax = 0

fig, axes = plt.subplots(
    len(metrics),
    len(grid_vars),
    sharey='row', sharex='col',
    figsize=(1.5*len(grid_vars), 3)
)
identifier=''
if not os.path.exists('../results/'):
    os.makedirs('../results/')

for row, metric in zip(axes, metrics):
    for var, name, ax in zip(grid_vars, settings_plot, row):

            columns, g_var = settings_plot[name]
            plot_df_ax = plot_df
            for col in columns:
                plot_df_ax = plot_df_ax.loc[plot_df_ax[col].isin(columns[col])]
            if g_var != '# Environments':
                plot_df_ax = plot_df_ax[ # IMPORTANT! otherwise average over all number of environments
                    plot_df_ax['# Environments'] == plot_df_ax['# Environments'].max()]

            if metric=="RT":
                ax.set( yscale="log")
            plot_df_ax = plot_df_ax[(plot_df_ax[r'$\bf{Test}$'].isin(methods))]

            # Create the tex files
            tex_plot(df=plot_df_ax, x=g_var, y=metric ,identifier='', outfile=f"./plots_output/tex_fig2a")

            palette = [2, 7, 6, 8, 9]

            sns.lineplot(
                data=plot_df_ax,
                x=g_var,
                y=metric,
                hue=r'$\bf{Test}$',
                ax=ax,
                palette=[
                    sns.color_palette("tab10")[i]
                    for i in palette
                ],
                legend='full',
                lw=2,
            )

            xmin = plot_df_ax[g_var].min()
            xmax = plot_df_ax[g_var].max()

            if xmax > 1:
                ax.set_xticks([
                    xmin,
                    int(xmin + (xmax - xmin) / 2),
                    xmax,
                ])
            else:
                ax.set_xticks([
                    np.round(xmin, 1),
                    np.round(xmin + (xmax - xmin) / 2 , 1),
                    np.round(xmax, 1),
                ])


leg_idx = 4

axes = np.concatenate(axes)

for i in range(len(axes)):
    axes[i].legend()
    plt.setp(axes[i].get_legend().get_title(), fontsize=22)
    if i == leg_idx:
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.setp(axes[i].get_legend().get_title(), fontsize=22)
    else:
        try:
            axes[i].get_legend().remove()
        except:
            pass
plt.subplots_adjust(hspace=0.15)
if SAVE_FIGURES:
    plt.savefig(f'plot.png')
plt.show()


