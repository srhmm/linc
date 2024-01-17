import os
from statistics import mean, stdev

import numpy as np


def tex_plot(df, x, y, identifier, method_col=r'$\bf{Test}$',
             data_sim="cdnod", outfile='./plots_output/tex'):
    ''' Creates files with results of the methods '''
    xnm = x
    if x == '# Environments':
        if data_sim == 'sergio':
            xnm = 'conditions'
        else:
            xnm = 'environments'
    if x == '# Samples':
        xnm = 'samples'
    if x == '# Variables':
        xnm = 'variables'
    if x == 'Edge density':
        xnm = 'edgeDensity'
    if x == 'Shift Faction':
        xnm = 'shift'
    write_file = open(outfile+f"/tex{identifier}_{xnm}_{y}.csv", "w+")
    methods = df[method_col].unique()
    s = "X"
    for method in methods:
        s = s+"\t"+str(method)+"_"+str(y)+"\t"+str(method)+"_"+str(y)+"_std"+"\t"+str(method)+"_"+str(y)+"_cnf"


    for x_val in sorted(df[x].unique()):
        df_x = df[df[x] == x_val]
        xprint = x_val
        s = s + "\n" + str(xprint)
        for method in methods:
            df_xm = df_x[df_x[method_col] == method]
            if y=="F1":
                df_xm[y] = df_xm[y].fillna(0)

            if y == 'RT':
                df_xm[y] = df_xm[y].fillna(0)
            if len(df_xm) > 1:
                mn = mean(df_xm[y])
                std = stdev(df_xm[y])
                cf =  round(1.96 * std / np.sqrt(len(df_xm[y])), 3)
                s = s + "\t"+str(round(mn, 3)) + "\t"+ str(round(std, 3))+ "\t" + str(round(cf, 3))
            else:
                if y == 'RT':
                    s = s + "\t0"+"\t0"+"\t0"
                else:
                    s = s + "\tna"+"\tna"+"\tna"

    write_file.write(s)
    write_file.flush()


def tex_plot_robustness(df1, df2, info1, info2, info_both, methods, metric, outfile=f"./plots_output"):

    write_file = open(outfile+f"/tex_robustness.csv", "w+")
    write_file.write(f'%%% SETTING: {info_both}, METRIC: {metric}'+'\n')
    write_file.write(r'\begin{tikzpicture}%'+'\n')
    write_file.write(r'\begin{axis}[ only marks,  height = 3.5cm, width= 3cm,   xtick = {0,...,6},  ytick = {0,2,4,6,8,10},'+'\n')
    write_file.write(f'   xmax = 6, ymax = 10, 	ylabel={metric}, xlabel={info1}, xmin = 0,  xmax = 1,  ymin = 0,  ymax = 1, '+'\n')
    write_file.write(r'   xtick={0,1} , ytick={0  ,0.2,0.4,0.6,0.8,1}, x tick label as interval,  ]%'+'\n')

    methods = methods[:5]
    offsets = [0.2, 0.35, 0.5, 0.65, 0.8]
    for i, m in enumerate(methods):
        mn, std, _, _ = tex_get_mn(df1, metric, m)
        mn, std = round(mn, 4), round(std, 4)
        write_file.write(f'%\n% Setting: {info1}, Metric: {metric}, Method: {m}'+'\n')
        write_file.write(r' \addplot+ [mark=*, error bars/.cd, y dir = both, y explicit] coordinates { '+ f'({offsets[i]}, {mn}) += (0.2, {std}) -= (0.2, {std})'+ '};'+'\n')

    write_file.write(r'\end{axis}%'+'\n')
    write_file.write(r'\begin{axis} [%'+'\n')
    write_file.write(r'   y axis line style={draw=none}, y tick style={draw=none}, y tick label style={draw=none}, only marks, xtick = {0,...,6}, ytick = {0,2,4,6,8,10}, '	+'\n')
    write_file.write(f'   xmin = 0, xmax = 1, ymin = 0, ymax = 1,  xlabel={info2},'+ r' xtick={0,1,2,3},  ytick={0,0.2,0.4,0.6,0.8,1}, x tick label as interval, yticklabels={   }, ]  '+'\n')


    for i, m in enumerate(methods):
        mn, std, _ , _= tex_get_mn(df2, metric, m)
        mn, std = round(mn, 4), round(std, 4)
        write_file.write(f'%\n% Setting: {info2}, Metric: {metric}, Method: {m}'+'\n')
        write_file.write(
            r' \addplot+ [mark=*, error bars/.cd, y dir = both, y explicit] coordinates { ' + f'({offsets[i]}, {mn}) += (0.2, {std}) -= (0.2, {std})' + '};'+'\n')

    write_file.write(r'\end{axis}%'+'\n')
    write_file.write(r'\end{tikzpicture}%'+'\n')

    write_file.flush()

def tex_print_robustness(df1, df2, info1, info2, info_both, methods, metric):
    print(f'SETTING: {info_both}, METRIC: {metric}')

    methods = methods[:5]
    offsets = [0.2, 0.35, 0.5, 0.65, 0.8]
    print(f'1. {info1}')

    def _ln(df, d):
        return len(df.loc[df['data_simulator'] == d])
    for i, m in enumerate(methods):
        mn, std, _, ln = tex_get_mn(df1, metric, m)
        mn, std = round(mn, 4), round(std, 4)
        df1_sub  =  df1[df1[r'$\bf{Test}$'] == m]

        print('\t',f'-{m} : {mn} +- {std}', '\t#', f'{ln}') # , "->", [f'{d}:, { _ln(df1_sub,d) }' for d in np.unique(df1_sub['data_simulator'])])

    print(f'2. {info2}')
    for i, m in enumerate(methods):
        mn, std, _, ln = tex_get_mn(df2, metric, m)
        mn, std = round(mn, 4), round(std, 4)
        df2_sub  =  df2[df2[r'$\bf{Test}$'] == m]
        print('\t',f'-{m} : {mn} +- {std}', '\t#', f'{ln}' ) #, "->", [f'{d}:, { _ln(df2_sub, d)}' for d in np.unique(df2_sub['data_simulator'])])


def tex_get_mn(df_x, metric, method):
    df_xm = df_x[df_x[r'$\bf{Test}$'] == method]
    if metric=='F1' or metric =='RT':
        df_xm[metric] = df_xm[metric].fillna(0)
    if len(df_xm) > 1:
        mn = mean(df_xm[metric])
        std = stdev(df_xm[metric])
        cf =  round(1.96 * std / np.sqrt(len(df_xm[metric])), 3)
    else:
        mn = df_xm[metric]
        std = 0
        cf = 0
    return mn, cf, std, len(df_xm)