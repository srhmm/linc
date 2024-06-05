import random


def generate_cuts(T, n, min_dur=100, max_attempts=100):
    """
    Cur a range (0, T) into n bins of minimal width min_dur
    :param T: Length of the range
    :param n: Number of bins
    :param min_dur: Minimal bin width
    :param max_attempts: Maximal number of attempts to obtain the correct minimal width
    :return: The cutpoints, a flag indicating the success or failure of the task
    """
    attempts = 0
    durs = [False]
    while not all(durs) and attempts < max_attempts:
        cuts = [0] + sorted(random.sample(range(T), n - 1)) + [T]
        durs = [(cuts[i+1] - cuts[i]) >= min_dur for i in range(len(cuts)-1)]
        attempts += 1
    return cuts, all(durs)


def generate_seq(R, n):
    """
    Generate a sequence of length n from R symbols with each symbol at least once and no consecutive repetition
    :param R: Number of different symbols
    :param n: Length of the sequence
    :return:
    """
    assert n >= R
    seq = list(range(R))
    random.shuffle(seq)
    for i in range(R, n):
        choices = set(range(R))
        seq.append(random.choice(list(choices - {seq[i - 1]})))
    return seq

def r_partition_to_windows_T(r_partition, skip):
    return [(b + skip, b + d) for (b, d, r) in r_partition]

def windows_T_to_r_partition(windows_T, skip, regimes=None):
    regimes = regimes or [None] * len(windows_T)
    return [(b-skip, windows_T[i+1][0]-skip, regimes[i]) if i < len(windows_T)-1 else (b-skip, e-b+skip, regimes[i]) for i, (b, e) in enumerate(windows_T)]


def cuts_to_windows_T(cuts, skip):
    return [(cuts[i]+skip, cuts[i+1]-1) if i < len(cuts)-2 else (cuts[i]+skip, cuts[i+1]) for i in range(len(cuts) - 1)]


def partition_t(T, R, n, min_dur=100, equal_dur=True):
    """
    Generate a partition of n chunks over R different regimes and T datapoints
    :param T: Total length of the time series
    :param R: Number of different regimes
    :param n: Number of chunks
    :param min_dur: Minimal duration of the chunks
    :param equal_dur: If all chunks should have the same length (except the last)
    :return:
    """
    success = True
    if not equal_dur:
        cuts, success = generate_cuts(T, n, min_dur)
        p = [(cuts[i], cuts[i+1] - cuts[i], None) for i in range(len(cuts)-1)]
    if equal_dur or not success:
        dur = T//n
        p = [(i*dur, dur, None) for i in range(n-1)]
        p.append(((n-1)*dur, dur+T%n, None))
    p = [(p[i][0], p[i][1], r) for i, r in enumerate(generate_seq(R, n))]
    return p
