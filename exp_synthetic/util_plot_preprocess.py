import numpy as np


def plot_preprocess(df):
    df = df.loc[df['Precision'].notna(), :]
    df['Fraction of shifting mechanisms'] = df['sparsity'].map(float) / df['n_variables'].map(float)
    df['F1'] = 2 * df['Recall'] * df['Precision'] / (df['Recall'] + df['Precision'])

    df['F1'] = df['F1'].replace(np.nan, 0)

    x_var_rename_dict = {
        'sample_size': '# Samples',
        'Number of environments': '# Environments',
        'Fraction of shifting mechanisms': 'Shift fraction',
        'n_variables': '# Variables',
        'dag_density': 'Edge density',
    }
    plot_df = df.rename(
            x_var_rename_dict, axis=1
        ).rename(
            {'Method': r'$\bf{Test}$', 'Soft': r'$\bf{Score}$'}, axis=1
        ).replace(
            {
                'er': 'Erdos-Renyi',
                'ba': 'Hub',
                'PC (pool all)': 'Full PC (oracle)',
                'Full PC (KCI)': 'Pooled_PC_KCI',
                'Min changes (oracle)': 'MSS_oracle',
                'Min changes (KCI)': 'MSS_KCI',
                'Min changes (GAM)': 'MSS_GAM',
                'Min changes (Linear)': 'MSS_Linear',
                'Min changes (FisherZ)': 'MSS_FisherZ',
                'MC': 'MC',
                'linc_rff_nogain' : 'LINC_rff',
                'linc_rff_clus' : 'LINC_rff_clus',
                'linc_gp_clus' : 'LINC_gp_clus',
                'linc_rff_ilp' : 'LINC_ILP',
                'linc_gp_nogain': 'LINC_gp',
                False: 'Hard',
                True: 'Soft',
            }
    )
    plot_df  = plot_df.loc[plot_df[r'$\bf{Score}$']=='Soft']
    return plot_df,x_var_rename_dict