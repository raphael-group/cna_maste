"""Frozen copies of the `(A, B)` decoder's code before T- #124, verbatim: D1 `hmm_ab_decoder_utils.get_clone_column`,
`hmm_ab_decoder.get_clone_emission_loglikes`; D2 `hmm_utils.viterbi_uniform_switch`; D3
`hmm_ab_decoder.hmm_ab_decoder_potts_problem` (its chain annotated `CloneChain`)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit

from cnamaste.ab_decode.model import CloneChain
from cnamaste.emission import (
    CnaEmissionConfounders,
    get_decoded_emission,
    get_encoded,
    get_log_emission,
    get_minor_allele_count,
    get_state_major,
)
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.potts import PottsProblem
from cnamaste.spatio_genomic_counts import PseudobulkCounts


def get_clone_column(counts: PseudobulkCounts, clone: int, *, params: str, fold: bool) -> PseudobulkCounts:
    """Clone `clone`'s `(n_obs, 1)` counts, a channel not in `params` None, with their encodings, the minor count's where
    `fold`, for the coded lattice.

    Args:
        counts: `(n_clones, n_obs)` the clones' pseudobulks.
        clone: the column.
        params: the channels scored.
        fold: encode the minor count.

    Returns:
        The clone's `PseudobulkCounts`.
    """
    column = slice(clone, clone + 1)
    x_nb, base = counts.umis[column], counts.normal_baseline_sized[column]
    x_bb, total = counts.alt_snp_umis[column], counts.total_snp_umis[column]
    b_count = get_minor_allele_count(x_bb, total) if fold else x_bb
    encoded = get_encoded(PseudobulkCounts(x_nb, base, b_count, total, None, {}), common_zero_depth=False)
    has_rdr, has_baf = "m" in params, "p" in params
    return PseudobulkCounts(
        x_nb if has_rdr else None,
        base if has_rdr else None,
        x_bb if has_baf else None,
        total if has_baf else None,
        None,
        encoded.encodings,
    )


def get_clone_emission_loglikes(
    clone_counts: dict[int, PseudobulkCounts],
    parameters: CnaHmmParameters,
    clone_confounders: CnaEmissionConfounders,
    prior: np.ndarray,
    *,
    params: str,
) -> dict[int, np.ndarray]:
    """Each tumour clone's `(K, n_obs)` coded-lattice loglike, `eqn:nb_emission` plus `eqn:bb_emission`, less the
    `eta_s` prior and the terms the counts alone set, the same in every pair (`data_terms=False`).

    Args:
        clone_counts: each tumour clone's `PseudobulkCounts` with its encodings.
        parameters: the lattice's, with the dispersions.
        clone_confounders: `(n_clones, 1)` `theta_c` and `log Z_c`.
        prior: `(K,)` `eta_s` per pair off the ploidy-neutral pair.
        params: the channels scored, `m` read depth and `p` B-allele fraction.

    Returns:
        The loglike, by clone.
    """
    out = {}
    for c, counts in clone_counts.items():
        confounders = CnaEmissionConfounders(*(np.asarray(v)[c : c + 1] for v in clone_confounders))
        rdr, baf = get_decoded_emission(
            get_log_emission(
                counts, parameters, confounders, layout="lattice", encoding="coded", phased=False, data_terms=False
            )
        )
        # NB the coded lattice scores both channels: the decode keeps its own.
        rdr, baf = (rdr if "m" in params else 0.0), (baf if "p" in params else 0.0)
        out[c] = get_state_major(rdr + baf)[:, :, 0] - prior[:, None]
    return out


@njit(cache=True)
def viterbi_uniform_switch(emission, log_stay, log_switch, log_start, lengths):
    """The Viterbi path of a log emission, per segment of `lengths`, and its summed log score, where the transition
    stays (`log_stay`) or switches to any other state (`log_switch`).

    The best predecessor is then the same for every target bar itself, so a step is O(K) rather than O(K^2): the best
    and second-best incoming values, the second standing in where the best is the target itself. `_viterbi_sum`'s
    trick (`cnamaste.sandbox.decoder.integer_copy`) without its running sum.

    Args: `emission` (K, n_obs) log emission; `log_start` (K,); `lengths` segment lengths.
    Returns: the `(n_obs,)` path and its summed log score.
    """
    K, n = emission.shape
    path = np.empty(n, dtype=np.int64)
    back = np.empty((n, K), dtype=np.int16)
    V, new = np.empty(K), np.empty(K)
    first, total = 0, 0.0
    for le in lengths:
        for g in range(first, first + le):
            if g == first:
                for s in range(K):
                    V[s] = log_start[s] + emission[s, g]
                    back[g, s] = -1
                continue
            b1, a1, b2 = -np.inf, 0, -np.inf
            for s in range(K):
                if V[s] > b1:
                    b2, b1, a1 = b1, V[s], s
                elif V[s] > b2:
                    b2 = V[s]
            for s in range(K):
                stay = V[s] + log_stay
                sw = (b2 if a1 == s else b1) + log_switch
                if stay >= sw:
                    new[s], back[g, s] = stay + emission[s, g], s
                else:
                    prev = a1
                    if a1 == s:
                        prev, best = 0, -np.inf
                        for j in range(K):
                            if j != s and V[j] > best:
                                best, prev = V[j], j
                    new[s], back[g, s] = sw + emission[s, g], prev
            for s in range(K):
                V[s] = new[s]
        s = 0
        for k in range(1, K):
            if V[k] > V[s]:
                s = k
        total += V[s]
        for g in range(first + le - 1, first - 1, -1):
            path[g] = s
            if g > first:
                s = back[g, s]
        first += le
    return path, total


def hmm_ab_decoder_potts_problem(
    chain: CloneChain, emission_loglikes: dict[int, np.ndarray], clones: Sequence[int]
) -> PottsProblem:
    """`eqn:ab_coupling` as one Potts field: `C x n_obs` nodes, `K` labels, every edge a reward for agreeing.

    A clone's step pays `log_switch` everywhere plus `log_stay - log_switch` where it stays; the agreement prior pays
    a constant plus `clone_coupling` per agreeing pair. Both rewards are non-negative, so the field is metric Potts and
    every expansion move is a minimum cut.

    Args:
        chain: the `CloneGenomicChain`.
        emission_loglikes: the `(K, n_obs)` log score, by clone.
        clones: the ids, in node order.

    Returns:
        The `PottsProblem`, its nodes `c * n_obs + g`.
    """
    n_bins = emission_loglikes[clones[0]].shape[1]
    starts = np.concatenate([[0], np.cumsum(chain.lengths)[:-1]]).astype(np.int64)
    first_bin = np.zeros(n_bins, dtype=bool)
    first_bin[starts] = True
    # NB the unary is a cost, so the loglike is negated; a segment's first bin carries log_start.
    unary = np.concatenate(
        [-(emission_loglikes[c].T + np.where(first_bin[:, None], chain.log_start[None, :], 0.0)) for c in clones]
    )
    bins = np.arange(n_bins)
    inside = bins[:-1][~first_bin[1:]]
    chain_first = [i * n_bins + inside for i in range(len(clones))]
    chain_second = [i * n_bins + inside + 1 for i in range(len(clones))]
    pair_first, pair_second = [], []
    for i in range(len(clones)):
        for k in range(i + 1, len(clones)):
            pair_first.append(i * n_bins + bins)
            pair_second.append(k * n_bins + bins)
    coupling = np.concatenate(
        [
            np.full(sum(len(a) for a in chain_first), chain.log_stay - chain.log_switch),
            np.full(sum(len(a) for a in pair_first), chain.clone_coupling),
        ]
    )
    return PottsProblem(
        unary=unary,
        first=np.concatenate(chain_first + pair_first),
        second=np.concatenate(chain_second + pair_second),
        coupling=coupling,
    )
