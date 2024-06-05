import numpy as np
from cdt import metrics

from gpcd.other.util_dag import directional_f1


def get_adj_from_true_links(true_links, tau_max, N, timed):
    if timed:
        true_adj = np.zeros((N * (tau_max + 1), N))
        for j in range(N):
            for entry in true_links[j]:
                ((i, lag), _, _) = entry
                assert lag <= 0 and -lag <= tau_max
                if i != j:
                    index = N * -lag + i
                    true_adj[index][j] = 1
    else:
        true_adj = np.zeros((N, N))
        for j in range(N):
            for entry in true_links[j]:
                ((i, lag), _, _) = entry
                if i != j:
                    true_adj[i][j] = 1
    return true_adj


def compare_adj_to_links(untimed_graph, truths):
    res_adj = np.array(
        [
            [1 if untimed_graph.has_edge(j, i) else 0 for i in untimed_graph.nodes]
            for j in untimed_graph.nodes
        ]
    )
    return _compare_adj_to_links(res_adj, truths.true_links, truths.tau_max, False)


def compare_adj_to_graph(untimed_graph, true_graph):
    true_dag = np.zeros((len(untimed_graph.nodes), len(untimed_graph.nodes)))
    res_dag = np.zeros((len(untimed_graph.nodes), len(untimed_graph.nodes)))
    for e1, e2 in untimed_graph.edges:
        res_dag[e1][e2] = 1
    for e1, e2 in true_graph.edges:
        true_dag[e1][e2] = 1

    return eval_dag(true_dag, res_dag, len(true_dag), enable_SID_call=False)


def compare_timed_adj_to_links(timed_graph, untimed_graph, truths):
    res_adj = np.zeros((len(timed_graph.nodes), len(untimed_graph.nodes)))
    for i in timed_graph.nodes:
        for j in untimed_graph.nodes:
            if timed_graph.has_edge(i, (j, 0)):
                idx = i[1] * truths.tau_max
                res_adj[idx][j] = 1
    return _compare_adj_to_links(res_adj, truths.true_links, truths.tau_max, True)


def _compare_adj_to_links(res_adj, true_links, tau_max, timed):
    N = res_adj.shape[1]
    if timed:
        assert res_adj.shape[0] == N * (tau_max + 1)
        true_adj = np.zeros((N * (tau_max + 1), N))
        for j in range(N):
            for entry in true_links[j]:
                ((i, lag), _, _) = entry
                assert lag <= 0 and -lag <= tau_max
                if i != j:
                    index = N * -lag + i
                    true_adj[index][j] = 1
        res = eval_dag(true_adj, res_adj, N)
    else:
        assert res_adj.shape[0] == N
        true_adj = np.zeros((N, N))
        for j in range(N):
            for entry in true_links[j]:
                ((i, lag), _, _) = entry
                if i != j:
                    true_adj[i][j] = 1

        res = eval_dag(true_adj, res_adj, N)

    # log.info(
    #    f"\t{name}\t\t(f1={np.round(res['f1'], 2)}, mcc={np.round(res['mcc'], 2)})\t(shd={np.round(res['shd'], 2)}, "
    #    f"sid={np.round(res['sid'], 2)})\t(tp={res['tp']}, tn={res['tn']}, fp={res['fp']}, fn={res['fn']})"
    # )

    return_dict = dict()

    for name, val in res.items():
        suff = "-timed" if timed else ""
        return_dict[name + suff] = val
    return return_dict


def compare_dag_list(test_adj_list, true_adj, N, choose_by="f1", higher_is_better=True):
    f1best = -np.inf if higher_is_better else np.inf
    best = None
    for test_adj in test_adj_list:
        result = eval_dag(true_adj, test_adj, N)
        if (
            higher_is_better
            and result[choose_by] > f1best
            or (not higher_is_better)
            and result[choose_by] < f1best
        ):
            f1best = result[choose_by]
            best = result

    return_dict = dict()
    for name, val in best.items():
        return_dict[name] = val
    return return_dict


def compare_dag(test_adj, true_adj, N):
    best = eval_dag(true_adj, test_adj, N)
    return_dict = dict()
    for name, val in best.items():
        return_dict[name] = val
    return return_dict


def match_dags(true_dag, test_dag):
    return not False in [
        (
            not False
            in [
                (
                    (test_dag[i][j] == 0 and true_dag[i][j] == 0)
                    or (test_dag[i][j] != 0 and true_dag[i][j] != 0)
                )
                for j in range(len(test_dag[i]))
            ]
        )
        for i in range(len(test_dag))
    ]


def eval_dag(true_dag, res_dag, N, enable_SID_call=False):
    assert true_dag.shape[1] == res_dag.shape[1] and res_dag.shape[1] == N
    assert true_dag.shape[0] == res_dag.shape[0]

    f1, tp, tn, fn, fp = directional_f1(true_dag, res_dag)

    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = 1 if den == 0 else (tp * tn - fp * fn) / np.sqrt(den)

    tpr = 0 if (tp + fn == 0) else tp / (tp + fn)
    fpr = 0 if (tn + fp == 0) else fp / (tn + fp)
    fdr = 0 if (tp + fp == 0) else fp / (tp + fp)

    acc = match_dags(true_dag, res_dag)

    true_ones = np.array(
        [[1 if i != 0 else 0 for i in true_dag[j]] for j in range(len(true_dag))]
    )
    shd = metrics.SHD(true_ones, res_dag)
    if enable_SID_call:
        sid = np.float(metrics.SID(true_ones, res_dag))
    else:
        sid = -1  # call to R script not working on some machines

    checksum = tp + tn + fn + fp
    assert checksum == true_dag.shape[0] * true_dag.shape[1]

    res = {
        "acc": acc,
        "shd": shd,
        "sid": sid,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "fdr": fdr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "mcc": mcc,
    }
    return res
