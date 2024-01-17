

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.score.LocalScoreFunction import *

def score_g_mdl(Data, G, score_func, parameters):  # calculate the score for the current G
    # here G is a DAG
    score = 0
    for i, node in enumerate(G.get_nodes()):
        PA = G.get_parents(node)
        delta_score = score_func.score(Data, i, PA, parameters)
        score = score + delta_score
    return score




def insert_changed_score(Data, G, i, j, T, record_local_score, score_func, parameters):
    # calculate the changed score after the insert operator: i->j
    Tj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj
    Ti = np.union1d(np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
                    np.where(G.graph[:, i] != Endpoint.NULL.value)[0])  # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti)  # find the neighbours of Xj and are adjacent to Xi
    Paj = np.where(G.graph[j, :] == Endpoint.ARROW.value)[0]  # find the parents of Xj
    # the function local_score() calculates the local score
    tmp1 = np.union1d(NA, T).astype(int)
    tmp2 = np.union1d(tmp1, Paj)
    tmp3 = np.union1d(tmp2, [i]).astype(int)

    # before you calculate the local score, firstly you search in the
    # "record_local_score", to see whether you have calculated it before
    r = len(record_local_score[j])
    s1 = 0
    s2 = 0
    score1 = 0
    score2 = 0

    for r0 in range(r):
        if not np.setxor1d(record_local_score[j][r0][0:-1], tmp3).size:
            score1 = record_local_score[j][r0][-1]
            s1 = 1

        if not np.setxor1d(record_local_score[j][r0][0:-1],
                           tmp2).size:  # notice the difference between 0*0 empty matrix and 1*0 empty matrix
            score2 = record_local_score[j][r0][-1]
            s2 = 1
        else:
            if (not np.setxor1d(record_local_score[j][r0][0:-1], [-1]).size) and (not tmp2.size):
                score2 = record_local_score[j][r0][-1]
                s2 = 1

        if s1 and s2:
            break

    if not s1:
        score1 = score_func.score(Data, j, tmp3, parameters)#feval([score_func, Data, j, tmp3, parameters])
        temp = list(tmp3)
        temp.append(score1)
        record_local_score[j].append(temp)

    if not s2:
        score2 = score_func.score(Data, j, tmp2, parameters) #feval([score_func, Data, j, tmp2, parameters])
        # r = len(record_local_score[j])
        if len(tmp2) != 0:
            temp = list(tmp2)
            temp.append(score2)
            record_local_score[j].append(temp)
        else:
            record_local_score[j].append([-1, score2])

    ch_score = score1 - score2
    desc = [i, j, T]
    return ch_score, desc, record_local_score


def insert(G, i, j, T):
    # Insert operator
    # insert the directed edge Xi->Xj
    nodes = G.get_nodes()
    G.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))

    for k in range(len(T)):  # directing the previous undirected edge between T and Xj as T->Xj
        if G.get_edge(nodes[T[k]], nodes[j]) is not None:
            G.remove_edge(G.get_edge(nodes[T[k]], nodes[j]))
        G.add_edge(Edge(nodes[T[k]], nodes[j], Endpoint.TAIL, Endpoint.ARROW))

    return G



def delete_changed_score(Data, G, i, j, H, record_local_score, score_func, parameters):
    # calculate the changed score after the Delete operator
    Hj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj
    Hi = np.union1d(np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
                    np.where(G.graph[:, i] != Endpoint.NULL.value)[0])  # adjacent to Xi;
    NA = np.intersect1d(Hj, Hi)  # find the neighbours of Xj and are adjacent to Xi
    Paj = np.union1d(np.where(G.graph[j, :] == Endpoint.ARROW.value)[0], [i])  # find the parents of Xj
    # the function local_score() calculates the local score
    tmp1 = set(NA) - set(H)
    tmp2 = set.union(tmp1, set(Paj))
    tmp3 = tmp2 - {i}

    # before you calculate the local score, firstly you search in the
    # "record_local_score", to see whether you have calculated it before
    r = len(record_local_score[j])
    s1 = 0
    s2 = 0
    score1 = 0
    score2 = 0

    for r0 in range(r):
        if set(record_local_score[j][r0][0:-1]) == tmp3:
            score1 = record_local_score[j][r0][-1]
            s1 = 1

        if set(record_local_score[j][r0][
               0:-1]) == tmp2:  # notice the difference between 0*0 empty matrix and 1*0 empty matrix
            score2 = record_local_score[j][r0][-1]
            s2 = 1
        else:
            if (set(record_local_score[j][r0][0:-1]) == {-1}) and len(tmp2) == 0:
                score2 = record_local_score[j][r0][-1]
                s2 = 1

        if s1 and s2:
            break

    if not s1:
        score1 = score_func.score(Data, j, list(tmp3), parameters)#feval([score_func, Data, j, list(tmp3), parameters])
        temp = list(tmp3)
        temp.append(score1)
        record_local_score[j].append(temp)

    if not s2:
        score2 = score_func.score(Data, j, list(tmp2), parameters)#feval([score_func, Data, j, list(tmp2), parameters])
        r = len(record_local_score[j])
        if len(tmp2) != 0:
            temp = list(tmp2)
            temp.append(score2)
            record_local_score[j].append(temp)
        else:
            record_local_score[j].append([-1, score2])

    ch_score = score1 - score2
    desc = [i, j, H]
    return ch_score, desc, record_local_score