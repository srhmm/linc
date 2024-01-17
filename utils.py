import sys
from enum import Enum

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing



def printt(arg, fnm='out_tree_search.txt'):
    with open(fnm, 'a') as f:
        print(arg, file=f)


def cantor_pairing(x, y):
    return int((x + y) * (x + y + 1) / 2 + y)


class DecisionType(Enum):
    TP = 1
    TN = 2
    FP = 3
    FN = 4

class AccuracyType(Enum):
    T = 1
    F = 2

def logg (val):
    if(val==0): return 0
    else: return math.log(val)

def printo(*args, f_name = 'some_results.txt'):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    #f_name = '../../linc/old_results_pi_search.txt'

    with open(f_name, 'a') as f:
        sys.stdout = f
        print(*args)
        sys.stdout = original_stdout

def data_scale(y):
    scaler = preprocessing.StandardScaler().fit(y)
    return(scaler.transform(y))
#
# def powerset(iterable, emptyset = False):
#     s = list(iterable)
#     if emptyset :
#         return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
#     else:
#         return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def data_groupby_pi(Xc, yc, Pi):
    """Given a partition Pi={pi_k}, pool data within each group pi_k"""
    n_pi = len(Pi)
    n_c = Xc.shape[0]

    assert((len(Xc.shape) == 3) & (len(yc.shape) == 2))
    assert((yc.shape[0] == n_c) & (yc.shape[1] == Xc.shape[1]))

    Xpi = [np.concatenate([Xc[c_i] for c_i in Pi[pi_k]]) for pi_k in range(n_pi)]
    ypi = [np.concatenate([yc[c_i] for c_i in Pi[pi_k]]) for pi_k in range(n_pi)]

    #print("Partition:", Pi)
    #print("Data Shape:", [Xpi[i].shape for i in range(len(Xpi))])

    return(Xpi, ypi)


def plot_gaussianProcessRegression(Xpi, ypi, Pi,
                                   mn_pred_k, std_pred_k):
    for k in range(len(Pi)):
        plot_gaussianProcessRegression_k(k, Xpi, ypi, Pi, mn_pred_k, std_pred_k)

def plot_gaussianProcessRegression_k(k, Xpi, ypi, Pi,
                                     mn_pred_k, std_pred_k,
                                   m1=".", m2="+",fill=False):


# plt.plot(Xpi[k], ypi[k], linestyle="dotted", label="Group " + str(Pi[k]))
    plt.scatter(Xpi[k], ypi[k], marker=m1, label="Group " + str(Pi[k]))
    mean_pred = mn_pred_k[k]
    std_pred = std_pred_k[k]
    plt.scatter(Xpi[k], mean_pred, marker=m2, label="Preds "+ str(Pi[k]))
    if fill:
        if (k==len(Pi)):
            plt.fill_between( Xpi[k].ravel(),  mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred, color="tab:orange",
            alpha=0.5, label=r"95% confidence interval")
        else:
            plt.fill_between( Xpi[k].ravel(),  mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred, color="tab:orange",alpha=0.5)
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    _ = plt.title("Partition "+str(Pi))
    return()
# from scikit learn


def plot_gaussianProcessRegression_kc(k, Xpi, ypi, Xc, yc, Pi,
                                     mn_pred_k, std_pred_k,
                                      m1=".", m2=".",fill=False):

    pi_k  = Pi[k]
    for c_j in pi_k:
        plt.scatter(Xc[c_j], yc[c_j], marker=m1, label="Context " + str(c_j))
    mean_pred = mn_pred_k[k]
    std_pred = std_pred_k[k]
    plt.scatter(Xpi[k], mean_pred, marker=m2, label="Preds "+ str(Pi[k]), c="gray")
    if fill:
        if (k==len(Pi)):
            plt.fill_between( Xpi[k].ravel(),  mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred, color="tab:orange",
            alpha=0.5, label=r"95% confidence interval")
        else:
            plt.fill_between( Xpi[k].ravel(),  mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred, color="tab:orange",alpha=0.5)
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    _ = plt.title("Partition "+str(Pi))
    return()