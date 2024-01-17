from pulp import LpProblem, LpMaximize, LpVariable, LpInteger, GUROBI, CPLEX_PY, CPLEX_CMD, LpSolverDefault
import itertools

"""Help functions for ILP-based search for a partition Pi of contexts into groups"""


def pi_search_ILP(pair_distances,
                  c_n,
                  shift=0, wasserstein=True):

    """ Formulate partition search as an ILP and solve it.
    ILP objective: find clustering that minimizes pairwise distances.

    :param pair_distances: dist(i,j) for (i,j) in itertools.combinations(contexts, 2).
        (A) wasserstein distance between residuals
        (B) MDL gain of using a joint model M_ij for both contexts cmp. to separate models M_i, M_j (no shift needed)
    :param c_n: number of contexts
    :param shift: if (A), + a suitable shift to have both pos. and neg. distances, if (B) not needed
    :param wasserstein: if (A)
    :return:
    """
    contexts = range(c_n)
    if wasserstein:
        #shift=.14
        pairwise_dists = [-i + shift for i in pair_distances]
    else:
        pairwise_dists = [i+shift for i in pair_distances]
    prob = LpProblem("ClusterOptimalTransport", LpMaximize)

    contexts_ij = dict((str(i) + str(j), [i,j]) for (i, j) in itertools.combinations(contexts, 2) if i != j)
    vars_ij = dict((str(i) + str(j), LpVariable("x" + str(i) + str(j), 0, 1, LpInteger)) for (i, j) in itertools.combinations(contexts, 2) if i != j)
    dists_ij = dict((str(i) + str(j), 0) for (i, j) in itertools.combinations(contexts, 2) if i != j)
    comb_i = -1

    for pair in itertools.combinations(contexts, 2):
        comb_i = comb_i + 1
        i, j = pair[0], pair[1]
        dists_ij[str(i)+str(j)] = pairwise_dists[comb_i]

    prob += sum(
        [dists_ij[str(i) + str(j)] * vars_ij[str(i) + str(j)] for (i, j) in itertools.combinations(contexts, 2)
         if i != j])

    comb_i = -1
    for ijk in itertools.combinations(contexts, 3):
        comb_i = comb_i + 1
        i, j, k  = ijk[0], ijk[1], ijk[2]
        ij, jk, ik = str(i)+str(j), str(j)+str(k),  str(i)+str(k)
        #print(str(vars_ij[ij])+"+"+str(vars_ij[jk] )+"-"+str( vars_ij[ik]))

        prob += vars_ij[ij] + vars_ij[jk] - vars_ij[ik] <= 1
        prob += vars_ij[ij] - vars_ij[jk] + vars_ij[ik] <= 1
        prob +=-vars_ij[ij] + vars_ij[jk] + vars_ij[ik] <= 1


    #prob.solve()
    #solver = CPLEX_PY(msg=0)
    #solver = CBC(msg=0)
    prob.solve()
    # cplex.Cplex.solve(prob)

    s = ""
    for v in vars_ij:
        if (vars_ij[v].varValue==1): s = s+str(vars_ij[v].name)+", "

    return vars_ij, dists_ij, contexts_ij

