import numpy as np

from sparse_shift import dag_recall, dag_precision


def dag_tpfpfnbi(true_dag, cpdag):

    tp = len(np.where((true_dag + cpdag - cpdag.T) == 2)[0]) #only causal edge
    fp = len(np.where((true_dag + cpdag.T - cpdag) == 2)[0]) #only noncausal edge
    fn = len(np.where((true_dag - cpdag - cpdag.T) == 1)[0])
    bi = len(np.where((cpdag + cpdag.T) == 2)[0])

    #TODO bidirected edges not used in the metrics?
    return tp, fp, fn, bi

def dag_fpr(true_dag, cpdag):
    tp, fp, fn, _ = dag_tpfpfnbi(true_dag, cpdag)
    return tp / (tp + fn) if (tp + fn) > 0 else 1


# uses p value thresholds:
def average_mec_precision_score(true_dag, pvalues_mat):
    from sparse_shift.utils import dag2cpdag, cpdag2dags
    thresholds = np.unique(pvalues_mat)
    dags = np.asarray(cpdag2dags(dag2cpdag(true_dag)))

    # ap_score = 0
    # prior_recall = 0

    precisions = []
    recalls = []
    tpr = []
    fpr = []
    labels_true = []
    labels_soft = []

    for t in thresholds:
        axis = tuple(np.arange(1, pvalues_mat.ndim))
        n_changes = np.sum(pvalues_mat <= t, axis=axis) / 2
        min_idx = np.where(n_changes == np.min(n_changes))[0]
        cpdag = (np.sum(dags[min_idx], axis=0) > 0).astype(int)
        precisions.append(dag_precision(true_dag, cpdag))
        recalls.append(dag_recall(true_dag, cpdag))
        labels_true = labels_true + list(true_dag.ravel())

        tpr.append(dag_recall(true_dag, cpdag))
        fpr.append(dag_fpr(true_dag, cpdag))
        #estim_dag_weighted = (cpdag - cpdag.T) * (1-t)# disregard bidirected edges and weight by decr pval

        mx = max(cpdag.ravel())
        estim_dag_weighted = cpdag * (mx - t)
        mins = estim_dag_weighted[np.nonzero(estim_dag_weighted)]
        mn = .1
        if len(mins) > 0:
            mn = min(min(mins), .1)
        estim_dag_weighted[estim_dag_weighted == estim_dag_weighted.transpose()] *= mn
        estim_dag_weighted[np.isnan(estim_dag_weighted)] = 0

        labels_soft = labels_soft + list(estim_dag_weighted.ravel())
        # ap_score += precision * (recall - prior_recall)
        # prior_recall = recall


    sort_idx = np.argsort(recalls)
    recalls = np.asarray(recalls)[sort_idx]
    precisions = np.asarray(precisions)[sort_idx]

    labels_soft = np.asarray(labels_soft)[sort_idx]
    labels_true = np.asarray(labels_true)[sort_idx]

    from sklearn.metrics import auc, roc_curve, average_precision_score, roc_auc_score
    if (len(recalls)>1):
        aupr = auc(recalls, precisions)

        if (1 in labels_true):
            fprate, tprate, _ = roc_curve(labels_true, labels_soft)
            auroc_2 = auc(fprate, tprate)
            auroc_1 = roc_auc_score(labels_true, labels_soft)
            auroc = auc(fpr, tpr)
        else:
            if (len(np.nonzero(labels_soft))):
                auroc = 0
            else:
                auroc = 1
        #ap = average_precision_score(recalls, precisions)

    else:
        score = precisions[0] * recalls[0]
        aupr, auroc = score, score
    # if len(thresholds) == 1:
    #     ap_score = precisions[0] * recalls[0]
    # else:
    ap = (np.diff(recalls, prepend=0) * precisions).sum()

    return ap, aupr, auroc


def soft_score_var(scores, pvals):
    min_idx = np.where(scores == np.min(scores))[0]
    soft_scores = []
    if len(min_idx)>0:
        min_idx = min_idx[0]
        soft_scores =pvals[min_idx].ravel() + 0.001
    return soft_scores

def pr_scores(true_dag,estim_dag, soft_scores_variable):

    if (len(soft_scores_variable)==len(estim_dag)):#always have a nonzero weight per edge
        soft_dag = np.array([[estim_dag[i][j] * soft_scores_variable[j] for j in range(len(estim_dag[i]))] for i in
                    range(len(estim_dag))])
        soft_dag_rev = np.array([[estim_dag[i][j] * soft_scores_variable[i] for j in range(len(estim_dag[i]))] for i in
                    range(len(estim_dag))])
        #ap_opt, aupr_opt, auroc_opt = pr_scores_optimistic(true_dag, estim_dag)
        ap, aupr, auroc  = pr_scores_(true_dag, estim_dag, soft_dag)
        ap, aupr, auroc = np.round(ap, 4), np.round(aupr, 4), np.round(auroc, 4)
    else:

        ap, aupr, auroc  = None, None, None
    return ap, aupr, auroc


def pr_scores_(true_dag, estim_dag, soft_scores):

    # ap_score = 0
    # prior_recall = 0

    precisions = []
    recalls = []
    tpr = []
    fpr = []
    labels_true = []
    labels_soft = []

    precisions.append(dag_precision(true_dag, estim_dag))
    recalls.append(dag_recall(true_dag, estim_dag))
    labels_true = labels_true + list(true_dag.ravel())

    tpr.append(dag_recall(true_dag, estim_dag))
    fpr.append(dag_fpr(true_dag, estim_dag))
    #estim_dag_weighted = (cpdag - cpdag.T) * (1-t)# TODO disregard bidirected edges and weight by decr pval

    #soft_scores[np.isnan(soft_scores)] = 0
    labels_soft = labels_soft + list(soft_scores.ravel())
    # ap_score += precision * (recall - prior_recall)
    # prior_recall = recall

    sort_idx = np.argsort(recalls)
    recalls = np.asarray(recalls)[sort_idx]
    precisions = np.asarray(precisions)[sort_idx]

    sort_idx = np.argsort(labels_soft)
    sort_idx = sort_idx[::-1]
    labels_soft = np.asarray(labels_soft)[sort_idx]
    labels_true = np.asarray(labels_true)[sort_idx]


    from sklearn.metrics import auc, roc_auc_score

    # AUROC: considers labels
    if (1 in labels_true):
        auroc = roc_auc_score(labels_true, labels_soft)
    else:
        if (len(np.nonzero(labels_soft))):
            auroc = 0
        else:
            auroc = 1

    # Area under precision recall curve
    if (len(recalls)>1):
        aupr = auc(recalls, precisions)
    else:
        aupr = precisions[0] * recalls[0]

    # Same as average precision

    # if len(thresholds) == 1:
    #     ap_score = precisions[0] * recalls[0]
    # else:
    ap = (np.diff(recalls, prepend=0) * precisions).sum()

    return ap, aupr, auroc


#only considers estimated dag and does an optimistic estimate of ap, auc, auroc:
def pr_scores_optimistic(true_dag, estim_dag):

    # ap_score = 0
    # prior_recall = 0

    precisions = []
    recalls = []
    tpr = []
    fpr = []
    labels_true = []
    labels_soft = []

    for _ in range(1):
        cpdag = estim_dag
        precisions.append(dag_precision(true_dag, cpdag))
        recalls.append(dag_recall(true_dag, cpdag))
        labels_true = labels_true + list(true_dag.ravel())

        tpr.append(dag_recall(true_dag, cpdag))
        fpr.append(dag_fpr(true_dag, cpdag))
        #estim_dag_weighted = (cpdag - cpdag.T) * (1-t)# disregard bidirected edges and weight by decr pval

        mx = max(cpdag.ravel())
        estim_dag_weighted = cpdag
        mins = estim_dag_weighted[np.nonzero(estim_dag_weighted)]
        mn = .1
        if len(mins) > 0:
            mn = min(min(mins), .1)
        estim_dag_weighted[np.isnan(estim_dag_weighted)] = 0
        estim_dag_weighted = estim_dag_weighted.ravel() * 1.0
        estim_dag_weighted[estim_dag_weighted == estim_dag_weighted.transpose()] *= mn

        labels_soft = labels_soft + list(estim_dag_weighted.ravel())
        # ap_score += precision * (recall - prior_recall)
        # prior_recall = recall


    sort_idx = np.argsort(recalls)
    recalls = np.asarray(recalls)[sort_idx]
    precisions = np.asarray(precisions)[sort_idx]

    labels_soft = np.asarray(labels_soft)[sort_idx]
    labels_true = np.asarray(labels_true)[sort_idx]

    from sklearn.metrics import auc, roc_curve, average_precision_score, roc_auc_score
    if (len(recalls)>1):
        aupr = auc(recalls, precisions)

        if (1 in labels_true):
            fprate, tprate, _ = roc_curve(labels_true, labels_soft)
            auroc_2 = auc(fprate, tprate)
            auroc_1 = roc_auc_score(labels_true, labels_soft)
            auroc = auc(fpr, tpr)
        else:
            if (len(np.nonzero(labels_soft))):
                auroc, auroc_1 = 0,0
            else:
                auroc, auroc_1  = 1, 1
        #ap = average_precision_score(recalls, precisions)

    else:
        score = precisions[0] * recalls[0]
        aupr, auroc, auroc_1 = score, score, score
    # if len(thresholds) == 1:
    #     ap_score = precisions[0] * recalls[0]
    # else:
    ap = (np.diff(recalls, prepend=0) * precisions).sum()

    return ap, aupr, auroc #, auroc_1
