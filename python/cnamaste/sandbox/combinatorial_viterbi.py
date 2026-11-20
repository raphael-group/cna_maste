"""`all_clone_combinatorial_viterbi`, `eqn:ab_coupling`'s exact joint decode of the tumour clones, `O(n K^C)`: the
referee the `hmm_ab` solvers are tested against, no longer a `ab_decode.solver`. Not shipped."""

import numpy as np
from numba import njit


@njit(cache=True)
def all_clone_combinatorial_viterbi(
    emission_loglike: np.ndarray,
    log_stay: float,
    log_switch: float,
    log_start: np.ndarray,
    lengths: np.ndarray,
    clone_coupling: float,
) -> tuple[np.ndarray, float]:
    """`eqn:ab_coupling` exactly: max-sum over the product of C clones' chains and `-clone_coupling` per bin for every
    pair of clones whose states differ.

    A joint state holds one state per clone, `s = sum_c s_c K^c`; a step applies each clone's stay or switch along its
    own axis in turn, `O(C K^C)` per bin. Each segment of `lengths` starts afresh at `log_start`, per clone.

    Args:
        emission_loglike: `(C, K, n_obs)` log scores.
        log_stay: a clone's step that keeps its pair.
        log_switch: a clone's step to any other pair.
        log_start: `(K,)` the log prior at a segment's first bin.
        lengths: the segments' bin counts.
        clone_coupling: `eta_c`, nats per bin and disagreeing pair, >= 0.

    Returns:
        The `(C, n_obs)` copy paths and their summed loglike, the agreement prior included.
    """
    C, K, n = emission_loglike.shape
    S = K**C
    digit = np.empty((S, C), dtype=np.int64)
    pen = np.zeros(S)
    for s in range(S):
        t = s
        for c in range(C):
            digit[s, c] = t % K
            t //= K
        for c in range(C):
            for d in range(c + 1, C):
                if digit[s, c] != digit[s, d]:
                    pen[s] -= clone_coupling
    paths = np.empty((C, n), dtype=np.int64)
    back = np.empty((n, C, S), dtype=np.int8)
    V, W, new = np.empty(S), np.empty(S), np.empty(S)
    total, first = 0.0, 0
    for le in lengths:
        for g in range(first, first + le):
            for s in range(S):
                e = pen[s]
                for c in range(C):
                    e += emission_loglike[c, digit[s, c], g]
                    if g == first:
                        e += log_start[digit[s, c]]
                W[s] = e
            if g == first:
                V[:] = W
                continue
            for c in range(C):
                stride = K**c
                for s0 in range(S):
                    if digit[s0, c] != 0:
                        continue
                    b1, a1, b2 = -np.inf, 0, -np.inf
                    for k in range(K):
                        v = V[s0 + k * stride]
                        if v > b1:
                            b2, b1, a1 = b1, v, k
                        elif v > b2:
                            b2 = v
                    for k in range(K):
                        s = s0 + k * stride
                        stay = V[s] + log_stay
                        sw = (b2 if a1 == k else b1) + log_switch
                        if stay >= sw:
                            new[s], back[g, c, s] = stay, k
                        else:
                            prev = a1
                            if a1 == k:
                                prev, best = 0, -np.inf
                                for j in range(K):
                                    if j != k and V[s0 + j * stride] > best:
                                        best, prev = V[s0 + j * stride], j
                            new[s], back[g, c, s] = sw, prev
                V[:] = new
            for s in range(S):
                V[s] += W[s]
        s = 0
        for t in range(S):
            if V[t] > V[s]:
                s = t
        total += V[s]
        for g in range(first + le - 1, first - 1, -1):
            for c in range(C):
                paths[c, g] = digit[s, c]
            if g == first:
                break
            for c in range(C - 1, -1, -1):
                stride = K**c
                s += (back[g, c, s] - digit[s, c]) * stride
        first += le
    return paths, total
