"""The forward-backward pass, and its max-product form `get_viterbi`, over clone-major columns: one chain per clone and contig, `S` states, or `2K`
(phase, state) pairs stepped by the factored `T_phase ⊗ T_state`.

Emissions are encoded: per clone, a table of each class's log emission per state, `(n_clones, n_classes, S)` for each of
the read-depth and B-allele channels, and each bin's class, `(n_clones, n_bins)`. A bin's log emission is
`rdr[c, rdr_index[c, g]] + baf[c, baf_index[c, g]]`, formed per step. A dense emission is the table with the bins as
classes (`get_dense_encoded_emission`). Every array is C-order, `(n_clones, n_bins, S)`; phased, `S = 2K` is
`(phase, state)`.
"""

import numpy as np
import scipy.special
from numba import njit, prange

from cnamaste.emission import EncodedEmission
from cnamaste.hmm_utils import logsumexp


def get_dense_encoded_emission(log_emission: np.ndarray) -> EncodedEmission:
    """A dense log emission as a `EncodedEmission`: each bin its own read-depth class, one zero B-allele class.

    Args:
        log_emission: `(n_clones, n_bins, S)`, or `(n_clones, n_bins, 2, K)` phased.

    Returns:
        The encoded emission; each bin's log emission is its dense one plus 0.0.
    """
    n_clones, n_bins = log_emission.shape[:2]
    rdr = np.ascontiguousarray(np.reshape(log_emission, (n_clones, n_bins, -1)), dtype=np.float64)
    rdr_index = np.ascontiguousarray(np.broadcast_to(np.arange(n_bins, dtype=np.int64), (n_clones, n_bins)))
    baf = np.zeros((n_clones, 1, rdr.shape[2]))
    return EncodedEmission(rdr, rdr_index, baf, np.zeros((n_clones, n_bins), dtype=np.int64))


def get_contig_offsets(lengths: np.ndarray) -> np.ndarray:
    """Each contig's first bin, and the bin count last.

    Args:
        lengths: bins per contig.

    Returns:
        `(n_contigs + 1,)` offsets.
    """
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def get_phase_steps(log_sitewise_transmat: np.ndarray | None, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """The log phase stay and switch of each step, from bin g to g + 1; zero-width where unphased.

    Args:
        log_sitewise_transmat: `(n_bins,)` log switch probability, or None where unphased.
        n_bins: the bins.

    Returns:
        `(log_stay, log_switch)`, `(n_bins,)` each, or `(0,)` each.
    """
    if log_sitewise_transmat is None:
        return np.zeros(0), np.zeros(0)
    # NB contiguous, as a capture replays it: one compiled kernel for the run and the replay.
    log_switch = np.ascontiguousarray(np.asarray(log_sitewise_transmat, dtype=np.float64).reshape(-1)[:n_bins])
    return np.log(1.0 - np.exp(log_switch)), log_switch


@njit(cache=True, error_model="numpy")
def get_bin_emission(encoded_rdr, rdr_index, encoded_baf, baf_index, clone, bin_, out):
    """Write bin `bin_` of `clone`'s log emission per state into `out`, read depth plus B allele."""
    rdr_row = encoded_rdr[clone, rdr_index[clone, bin_]]
    baf_row = encoded_baf[clone, baf_index[clone, bin_]]
    for s in range(out.size):
        out[s] = rdr_row[s] + baf_row[s]


@njit(cache=True, error_model="numpy")
def get_phased_step(prev, log_transmat, log_stay, log_switch, penalize, out):
    """`out[q, j] = log sum_{p, i} exp(prev[p, i] + log T[(p, i), (q, j)])`, with T the phase step by `log_transmat`.

    A K x K sum per phase, then a 2 x 2 phase mix. With `penalize` (`hmm.penalize_phase_only_on_same_cnv`) the phase
    step applies on the copy diagonal alone, and a step off it takes either phase at 1/2. The backward step is this
    step on `log_transmat.T`, as the phase step is symmetric.

    Args:
        prev: `(2, K)` log probabilities.
        log_transmat: `(K, K)` copy state transitions.
        log_stay: log phase stay.
        log_switch: log phase switch.
        penalize: `hmm.penalize_phase_only_on_same_cnv`.
        out: `(2, K)`, written.
    """
    n_states = log_transmat.shape[0]
    buf = np.empty(n_states)
    mixed = np.empty(2)
    pair = np.empty(2)
    for j in range(n_states):
        for p in range(2):
            for i in range(n_states):
                buf[i] = -np.inf if penalize and i == j else prev[p, i] + log_transmat[i, j]
            mixed[p] = logsumexp(buf)
        for q in range(2):
            if penalize:
                for p in range(2):
                    pair[p] = (log_stay if p == q else log_switch) + prev[p, j] + log_transmat[j, j]
                diagonal = logsumexp(pair)
                pair[0], pair[1] = diagonal, np.log(0.5) + logsumexp(mixed)
            else:
                for p in range(2):
                    pair[p] = (log_stay if p == q else log_switch) + mixed[p]
            out[q, j] = logsumexp(pair)


@njit(parallel=True, cache=True, error_model="numpy")
def get_log_alpha(
    encoded_rdr, rdr_index, encoded_baf, baf_index, log_startprob, log_transmat, starts, log_stay, log_switch, penalize
):
    """`log_alpha[c, g, s] = log P(o_1 .. o_g, q_g = s)` per clone and contig; phased where `log_stay` is not empty.

    Args:
        encoded_rdr: `(n_clones, n_rdr_classes, S)` read-depth class table.
        rdr_index: `(n_clones, n_bins)` each bin's read-depth class.
        encoded_baf: `(n_clones, n_baf_classes, S)` B-allele class table.
        baf_index: `(n_clones, n_bins)` each bin's B-allele class.
        log_startprob: `(K,)`; phased, each phase takes half.
        log_transmat: `(K, K)` copy state transitions.
        starts: `get_contig_offsets`.
        log_stay: `(n_bins,)` log phase stay from bin g to g + 1, or `(0,)` unphased.
        log_switch: `(n_bins,)` log phase switch, or `(0,)`.
        penalize: `hmm.penalize_phase_only_on_same_cnv`.

    Returns:
        `(n_clones, n_bins, S)` forward log probabilities.
    """
    n_clones, n_bins = rdr_index.shape
    n_paired = encoded_rdr.shape[2]
    n_states = log_transmat.shape[0]
    phased = log_stay.size > 0
    n_contigs = starts.size - 1
    log_half = np.log(0.5)
    log_alpha = np.zeros((n_clones, n_bins, n_paired))

    for task in prange(n_clones * n_contigs):
        clone, contig = task // n_contigs, task % n_contigs
        first, last = starts[contig], starts[contig + 1]
        emission = np.empty(n_paired)
        buf = np.empty(n_states)
        out = np.empty((2, n_states))

        get_bin_emission(encoded_rdr, rdr_index, encoded_baf, baf_index, clone, first, emission)
        for s in range(n_paired):
            start = log_half + log_startprob[s % n_states] if phased else log_startprob[s]
            log_alpha[clone, first, s] = start + emission[s]

        for g in range(first + 1, last):
            get_bin_emission(encoded_rdr, rdr_index, encoded_baf, baf_index, clone, g, emission)
            if phased:
                prev = log_alpha[clone, g - 1].reshape(2, n_states)
                get_phased_step(prev, log_transmat, log_stay[g - 1], log_switch[g - 1], penalize, out)
                for s in range(n_paired):
                    log_alpha[clone, g, s] = out[s // n_states, s % n_states] + emission[s]
            else:
                for j in range(n_states):
                    for i in range(n_states):
                        buf[i] = log_alpha[clone, g - 1, i] + log_transmat[i, j]
                    log_alpha[clone, g, j] = logsumexp(buf) + emission[j]

    return log_alpha


@njit(parallel=True, cache=True, error_model="numpy")
def get_log_beta(encoded_rdr, rdr_index, encoded_baf, baf_index, log_transmat, starts, log_stay, log_switch, penalize):
    """`log_beta[c, g, s] = log P(o_{g+1} .. o_end | q_g = s)` per clone and contig; `get_log_alpha`'s arguments.

    Returns:
        `(n_clones, n_bins, S)` backward log probabilities.
    """
    n_clones, n_bins = rdr_index.shape
    n_paired = encoded_rdr.shape[2]
    n_states = log_transmat.shape[0]
    phased = log_stay.size > 0
    n_contigs = starts.size - 1
    log_transmat_t = np.ascontiguousarray(log_transmat.T)
    log_beta = np.zeros((n_clones, n_bins, n_paired))

    for task in prange(n_clones * n_contigs):
        clone, contig = task // n_contigs, task % n_contigs
        first, last = starts[contig], starts[contig + 1]
        emission = np.empty(n_paired)
        buf = np.empty(n_states)
        weighted = np.empty((2, n_states))
        out = np.empty((2, n_states))

        for g in range(last - 2, first - 1, -1):
            get_bin_emission(encoded_rdr, rdr_index, encoded_baf, baf_index, clone, g + 1, emission)
            if phased:
                for s in range(n_paired):
                    weighted[s // n_states, s % n_states] = log_beta[clone, g + 1, s] + emission[s]
                get_phased_step(weighted, log_transmat_t, log_stay[g], log_switch[g], penalize, out)
                for s in range(n_paired):
                    log_beta[clone, g, s] = out[s // n_states, s % n_states]
            else:
                for i in range(n_states):
                    for j in range(n_states):
                        buf[j] = log_beta[clone, g + 1, j] + log_transmat[i, j] + emission[j]
                    log_beta[clone, g, i] = logsumexp(buf)

    return log_beta


@njit(cache=True, error_model="numpy")
def get_uniform_switch_step(value, emission, log_stay, log_switch, new, back):
    """One max-product step of a chain that stays at `log_stay` and switches to every other state at `log_switch`.

    Args:
        value: `(K,)` the previous bin's best log scores.
        emission: `(K,)` this bin's log emission.
        log_stay: log stay.
        log_switch: log switch, per other state.
        new: `(K,)` written, this bin's best log scores.
        back: `(K,)` written, each state's best predecessor.
    """
    n_states = value.size
    b1, a1, b2 = -np.inf, 0, -np.inf
    for s in range(n_states):
        if value[s] > b1:
            b2, b1, a1 = b1, value[s], s
        elif value[s] > b2:
            b2 = value[s]
    for s in range(n_states):
        stay = value[s] + log_stay
        switch = (b2 if a1 == s else b1) + log_switch
        if stay >= switch:
            new[s], back[s] = stay + emission[s], s
        else:
            prev = a1
            if a1 == s:
                prev, best = 0, -np.inf
                for j in range(n_states):
                    if j != s and value[j] > best:
                        best, prev = value[j], j
            new[s], back[s] = switch + emission[s], prev


def is_uniform_switch(log_transmat: np.ndarray) -> bool:
    """Whether `log_transmat` has one diagonal value and one off-diagonal value, `hmm_utils.get_log_transmat`'s form.

    Args:
        log_transmat: `(K, K)`.

    Returns:
        Whether `get_viterbi` may step in O(K).
    """
    off = ~np.eye(log_transmat.shape[0], dtype=bool)
    diagonal, others = np.diag(log_transmat), log_transmat[off]
    return bool(np.all(diagonal == diagonal[0]) and (others.size == 0 or np.all(others == others[0])))


@njit(cache=True, error_model="numpy")
def get_viterbi(encoded_rdr, rdr_index, encoded_baf, baf_index, log_startprob, log_transmat, starts, uniform):
    """`get_log_alpha`'s max-product form, unphased: each clone's best path per contig, and its log score.

    A step keeps the stay where it ties the best predecessor, else takes the first best; the last bin the first best.
    Where `uniform` (`is_uniform_switch`), a step is O(K): the best and second-best predecessor, the second standing
    in where the best is the target itself. Serial, as `ab_decode` runs it in forked workers.

    Args:
        encoded_rdr: `(n_clones, n_rdr_classes, K)` read-depth class table.
        rdr_index: `(n_clones, n_bins)` each bin's read-depth class.
        encoded_baf: `(n_clones, n_baf_classes, K)` B-allele class table.
        baf_index: `(n_clones, n_bins)` each bin's B-allele class.
        log_startprob: `(K,)` the log prior at a contig's first bin.
        log_transmat: `(K, K)` copy state transitions.
        starts: `get_contig_offsets`.
        uniform: `log_transmat` stays at its diagonal and switches to every other state alike.

    Returns:
        `(n_clones, n_bins)` paths and `(n_clones, n_contigs)` log scores.
    """
    n_clones, n_bins = rdr_index.shape
    n_states = log_transmat.shape[0]
    n_contigs = starts.size - 1
    log_stay, log_switch = log_transmat[0, 0], log_transmat[0, n_states - 1]
    paths = np.empty((n_clones, n_bins), dtype=np.int64)
    scores = np.empty((n_clones, n_contigs))

    for task in range(n_clones * n_contigs):
        clone, contig = task // n_contigs, task % n_contigs
        first, last = starts[contig], starts[contig + 1]
        emission = np.empty(n_states)
        value = np.empty(n_states)
        new = np.empty(n_states)
        back = np.empty((last - first, n_states), dtype=np.int16)

        get_bin_emission(encoded_rdr, rdr_index, encoded_baf, baf_index, clone, first, emission)
        for s in range(n_states):
            value[s] = log_startprob[s] + emission[s]

        for g in range(first + 1, last):
            get_bin_emission(encoded_rdr, rdr_index, encoded_baf, baf_index, clone, g, emission)
            if uniform:
                get_uniform_switch_step(value, emission, log_stay, log_switch, new, back[g - first])
                for s in range(n_states):
                    value[s] = new[s]
                continue
            for j in range(n_states):
                best, arg = value[j] + log_transmat[j, j], j
                for i in range(n_states):
                    step = value[i] + log_transmat[i, j]
                    if i != j and step > best:
                        best, arg = step, i
                new[j], back[g - first, j] = best + emission[j], arg
            for s in range(n_states):
                value[s] = new[s]

        state = 0
        for s in range(1, n_states):
            if value[s] > value[state]:
                state = s
        scores[clone, contig] = value[state]
        for g in range(last - 1, first - 1, -1):
            paths[clone, g] = state
            state = back[g - first, state]

    return paths, scores


def get_viterbi_paths(
    encoded: EncodedEmission, log_startprob: np.ndarray, log_transmat: np.ndarray, lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`get_viterbi` on a encoded, unphased emission: each clone's best path and its log score, summed over contigs in
    order.

    Args:
        encoded: the `EncodedEmission`, `(n_clones, n_classes, K)` tables.
        log_startprob: `(K,)` the log prior at a contig's first bin.
        log_transmat: `(K, K)`.
        lengths: bins per contig, every clone's.

    Returns:
        `(n_clones, n_bins)` paths and `(n_clones,)` log scores.
    """
    tables = (encoded.rdr, encoded.rdr_index, encoded.baf, encoded.baf_index)
    starts, uniform = get_contig_offsets(lengths), is_uniform_switch(log_transmat)
    paths, scores = get_viterbi(*tables, log_startprob, log_transmat, starts, uniform)
    return paths, np.cumsum(scores, axis=1)[:, -1]


def get_log_gamma(log_alpha: np.ndarray, log_beta: np.ndarray) -> np.ndarray:
    """The log posterior of each state, normalized per clone and bin.

    Args:
        log_alpha: `(n_clones, n_bins, S)`.
        log_beta: `(n_clones, n_bins, S)`.

    Returns:
        `(n_clones, n_bins, S)` log posteriors.

    Raises:
        RuntimeError: where no state can emit a bin.
    """
    log_gamma = log_alpha + log_beta
    if np.any(np.all(np.isneginf(log_gamma), axis=-1)):
        raise RuntimeError("No state can emit some bins: the posterior is undefined.")
    return log_gamma - scipy.special.logsumexp(log_gamma, axis=-1, keepdims=True)


def get_llf(log_alpha: np.ndarray, starts: np.ndarray) -> float:
    """The log-likelihood: the log-sum of each chain's last forward column, summed over clones and contigs.

    Args:
        log_alpha: `(n_clones, n_bins, S)`.
        starts: `get_contig_offsets`.

    Returns:
        The log-likelihood, summed clone-major.
    """
    return float(np.sum(scipy.special.logsumexp(log_alpha[:, starts[1:] - 1], axis=-1).reshape(-1)))


def get_forward_backward(
    encoded: EncodedEmission,
    log_startprob: np.ndarray,
    log_transmat: np.ndarray,
    lengths: np.ndarray,
    log_sitewise_transmat: np.ndarray | None,
    penalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """`get_log_alpha` and `get_log_beta` on a encoded emission; phased tables `(..., 2, K)` read as `2K` rows.

    Args:
        encoded: the `EncodedEmission`.
        log_startprob: `(K,)`.
        log_transmat: `(K, K)`.
        lengths: bins per contig, every clone's.
        log_sitewise_transmat: `(n_bins,)` log phase switch, or None where unphased.
        penalize: `hmm.penalize_phase_only_on_same_cnv`.

    Returns:
        `(log_alpha, log_beta)`, `(n_clones, n_bins, S)` each.
    """
    n_clones, n_bins = encoded.rdr_index.shape
    rdr = np.reshape(encoded.rdr, (n_clones, encoded.rdr.shape[1], -1))
    baf = np.reshape(encoded.baf, (n_clones, encoded.baf.shape[1], -1))
    tables = (rdr, encoded.rdr_index, baf, encoded.baf_index)
    starts = get_contig_offsets(lengths)
    stay, switch = get_phase_steps(log_sitewise_transmat, n_bins)
    log_alpha = get_log_alpha(*tables, log_startprob, log_transmat, starts, stay, switch, penalize)
    return log_alpha, get_log_beta(*tables, log_transmat, starts, stay, switch, penalize)
