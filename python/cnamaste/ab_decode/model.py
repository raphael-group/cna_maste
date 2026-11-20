"""The `(A, B)` decoder's types, chain and small helpers, so `ab_decode.decode` shows its core logic.

The reference is `cna-maste-paper`, section `sec:ab_decoder`: `eqn:ab_admixture` the normal admixture, `eqn:ab_hmm` the
scored copy path, `eqn:ab_coupling` the clone agreement prior.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from cnamaste.ab_decode.lattice import get_ab_is_ploidy_neutral
from cnamaste.config import get_contig_starts
from cnamaste.emission import CnaEmissionConfounders, get_encoded, get_log_emission, get_minor_allele_count
from cnamaste.hmm.baum_welch import HmmTransitions
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import get_log_transmat
from cnamaste.spatio_genomic_counts import PseudobulkCounts

PLOIDY_BAND = 0.5
"""How far a copy path's mean `A + B` may sit from the ploidy under `genome_wide_ploidy_constraint`, `eqn:ab_hmm`'s
`|T_bar_m - P| <= 1/2`."""


def get_ploidy_band() -> float:
    """Return the ploidy band, 0.5."""
    return PLOIDY_BAND


class DecodeStatus(enum.Enum):
    """How a tumour clone's decode ended: `AbDecodeResult.status`, by clone."""

    OK = "ok"
    """Decoded as asked."""
    NO_PATH_IN_PLOIDY_BAND = "no_path_in_ploidy_band"
    """No price brings the clone's mean `A + B` into the band: its free copy path, scored -inf."""
    NORMAL_FRACTION_UNMOVED = "normal_fraction_unmoved"
    """The clone's fitted `theta_c` is its start, to `NORMAL_FRACTION_XATOL`: the fit did not move it."""


class CloneChain(NamedTuple):
    """One clone's copy-path chain over the lattice, `eqn:ab_hmm`, as every solver decodes it.

    Attributes:
        transitions: `hmm_utils.get_log_transmat(K, stay_prob)`, unphased, and the segments' bin counts; no step
            crosses a segment boundary.
        log_start: `(K,)` the log prior at a segment's first bin, `get_contig_start_prior`.
        clone_coupling: `eta_c` of `eqn:ab_coupling`, >= 0.
    """

    transitions: HmmTransitions
    log_start: np.ndarray
    clone_coupling: float

    @property
    def K(self) -> int:
        """The lattice's pairs."""
        return int(self.transitions.log_transmat.shape[0])

    @property
    def log_stay(self) -> float:
        """`log stay_prob`, a step that keeps its pair."""
        return float(self.transitions.log_transmat[0, 0])

    @property
    def log_switch(self) -> float:
        """`log((1 - stay_prob) / (K - 1))`, a step to any other pair."""
        return float(self.transitions.log_transmat[0, 1])

    @property
    def lengths(self) -> np.ndarray:
        """The segments' bin counts."""
        return self.transitions.lengths


class AbDecodeResult(NamedTuple):
    """The `(A, B)` decode, `sec:ab_decoder`.

    Attributes:
        copies: `(n_obs, n_clones, 2)` integer `(A, B)` per bin and clone, the normal clone (1, 1).
        normal_fractions: `(n_clones,)` each clone's `theta_c`, `eqn:ab_admixture`.
        log_mu_shift: `(n_clones,)` each clone's `log Z_c`, `eqn:nb_emission`, for the decode returned.
        stay_prob: the stay probability decoded at.
        score: the joint `eqn:ab_coupling` loglike of the decode returned, less the terms the counts alone set.
        clone_scores: each tumour clone's own `eqn:ab_hmm` loglike, by clone, less the terms the counts alone set.
        ploidy: each tumour clone's assumed ploidy, by name.
        status: each tumour clone's `DecodeStatus`.
    """

    copies: np.ndarray
    normal_fractions: np.ndarray
    log_mu_shift: np.ndarray
    stay_prob: float
    score: float
    clone_scores: dict[int, float]
    ploidy: dict[int, str]
    status: dict[int, DecodeStatus]


def get_contig_start_prior(ab_lattice: np.ndarray, stay_prob: float, ploidy: int, start_prior_type: str) -> np.ndarray:
    """Each pair's log prior at a contig's first bin, `eqn:ab_hmm`'s first transition.

    `anchored` enters the contig from the neutral pair as a bin is entered from its neighbour: `log stay_prob` to stay
    there and `log((1 - stay_prob) / (K - 1))` to any other pair. `uniform` is `log(1 / K)`.

    Args:
        ab_lattice: `(K, 2)` the pairs.
        stay_prob: the stay probability.
        ploidy: P.
        start_prior_type: one of `CONTIG_STARTS`.

    Returns:
        The `(K,)` log prior.

    Raises:
        ValueError: on `start_prior_type` not one of `CONTIG_STARTS`.
    """
    K = ab_lattice.shape[0]
    if start_prior_type == "uniform":
        return np.full(K, -np.log(K))
    if start_prior_type != "anchored":
        raise ValueError(f"contig_start {start_prior_type!r} is not one of {get_contig_starts()}.")
    home = get_ab_is_ploidy_neutral(ab_lattice, ploidy)
    return np.where(home, np.log(stay_prob / home.sum()), np.log((1.0 - stay_prob) / (K - 1)))


def get_clone_chain(ab_lattice: np.ndarray, lengths: Any, config: Any, *, ploidy: int) -> CloneChain:
    """The copy-path chain every tumour clone is decoded on.

    Args:
        ab_lattice: `(K, 2)` the pairs.
        lengths: the segments' bin counts.
        config: the run's `YAMLConfig`: `ab_decode.t`, `contig_start` and `clone_coupling`.
        ploidy: P.

    Returns:
        The `CloneChain`.
    """
    K = ab_lattice.shape[0]
    ab_decode = config.ab_decode
    stay_prob = float(ab_decode.t)
    transitions = HmmTransitions(get_log_transmat(K, stay_prob), None, np.asarray(lengths, dtype=np.int64))
    return CloneChain(
        transitions=transitions,
        log_start=get_contig_start_prior(ab_lattice, stay_prob, ploidy, ab_decode.contig_start),
        clone_coupling=float(ab_decode.clone_coupling),
    )


def get_clone_rows(counts: PseudobulkCounts, index: Any) -> PseudobulkCounts:
    """`counts` indexed clone-major, a channel None kept None; no encoding.

    Args:
        counts: `(n_clones, n_obs)` the clones' pseudobulks.
        index: the rows, or `(rows, bins)`, as numpy indexes.

    Returns:
        The indexed `PseudobulkCounts`.
    """
    umis, base, alt, total = (None if v is None else np.asarray(v)[index] for v in counts[:4])
    return PseudobulkCounts(umis, base, alt, total, None, {})


def get_coded_rows(counts: PseudobulkCounts, clones: Sequence[int], *, fold: bool) -> PseudobulkCounts:
    """The clones' rows of `counts`, the B count the minor count where `fold`, `rdr` and `baf` encoded.

    Args:
        counts: `(n_clones, n_obs)` the clones' pseudobulks.
        clones: the rows kept, in order.
        fold: encode the minor count.

    Returns:
        The `(len(clones), n_obs)` encoded `PseudobulkCounts`.
    """
    if fold:
        counts = counts._replace(alt_snp_umis=get_minor_allele_count(counts.alt_snp_umis, counts.total_snp_umis))
    rows = get_clone_rows(counts, np.asarray(clones, dtype=np.int64))
    return get_encoded(rows, common_zero_depth=False)


def get_copy_path_emission_loglike(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    clone_confounders: CnaEmissionConfounders,
    copy_path: np.ndarray,
    *,
    fold: bool,
    data_terms: bool,
) -> float:
    """A clone's summed `eqn:nb_emission` and `eqn:bb_emission` loglike along `copy_path`, at its `theta_c` and
    `log Z_c`.

    Args:
        counts: the clone's `PseudobulkCounts`, a channel None not scored.
        parameters: the lattice's, `ab_lattice.get_lattice_parameters`.
        clone_confounders: the clone's `theta_c` and `log Z_c`.
        copy_path: the state per bin.
        fold: score the minor count.
        data_terms: keep the terms the counts alone set; False where only differences in `theta_c` are read.

    Returns:
        The summed loglike.
    """
    if fold and counts.alt_snp_umis is not None:
        counts = counts._replace(alt_snp_umis=get_minor_allele_count(counts.alt_snp_umis, counts.total_snp_umis))
    rdr, baf = get_log_emission(
        counts, parameters, clone_confounders, layout="path", phased=False, data_terms=data_terms, path=copy_path
    )
    return float(np.sum(rdr)) + float(np.sum(baf))


def get_copy_path_loglike(emission_loglike: np.ndarray, chain: CloneChain, copy_path: np.ndarray) -> float:
    """`copy_path`'s summed `eqn:ab_hmm` loglike on `chain`: emission, start and transitions.

    Args:
        emission_loglike: `(K, n_obs)` the clone's log score per pair and bin.
        chain: the `CloneChain`.
        copy_path: the state per bin.

    Returns:
        The loglike: the emission along `copy_path`, `log_start` per segment, `log_stay` or `log_switch` per step.
    """
    value, first = 0.0, 0
    for le in chain.lengths:
        seg = copy_path[first : first + le]
        stays = int(np.count_nonzero(seg[1:] == seg[:-1]))
        value += float(chain.log_start[seg[0]] + emission_loglike[seg, np.arange(first, first + le)].sum())
        value += stays * chain.log_stay + (le - 1 - stays) * chain.log_switch
        first += le
    return value


def get_clone_agreement_prior(
    copy_paths: dict[int, np.ndarray], clone: int, n_states: int, n_obs: int, clone_coupling: float
) -> np.ndarray:
    """Clone `clone`'s `eqn:ab_coupling` term with every other clone held: `-clone_coupling` per other clone whose
    state differs, per pair and bin.

    Args:
        copy_paths: the state per bin, by clone.
        clone: the clone decoded.
        n_states: the lattice's pairs.
        n_obs: the bins.
        clone_coupling: `eta_c`.

    Returns:
        The `(n_states, n_obs)` prior, 0 where `clone_coupling` is 0.
    """
    others = [d for d in copy_paths if d != clone]
    if not others or clone_coupling <= 0.0:
        return np.zeros((n_states, n_obs))
    agree = np.zeros((n_states, n_obs))
    bins = np.arange(n_obs)
    for d in others:
        agree[copy_paths[d], bins] += 1.0
    return -clone_coupling * (len(others) - agree)


def get_all_clone_agreement_prior(copy_paths: dict[int, np.ndarray], clone_coupling: float) -> float:
    """`eqn:ab_coupling`'s prior: `-clone_coupling` per bin and pair of clones whose states differ.

    Args:
        copy_paths: the state per bin, by clone.
        clone_coupling: `eta_c`.

    Returns:
        The summed prior.
    """
    clones = sorted(copy_paths)
    if clone_coupling <= 0.0 or len(clones) < 2:
        return 0.0
    total = 0
    for i, c in enumerate(clones):
        for d in clones[i + 1 :]:
            total += int(np.count_nonzero(copy_paths[c] != copy_paths[d]))
    return -clone_coupling * total


def in_ploidy_band(copy_path: np.ndarray, totals: np.ndarray, ploidy: int) -> bool:
    """Whether `copy_path`'s mean `A + B` sits within `PLOIDY_BAND` of the ploidy, `eqn:ab_hmm`.

    Args:
        copy_path: the state per bin.
        totals: `(K,)` each pair's `A + B`.
        ploidy: P.

    Returns:
        Whether the mean is in the band.
    """
    return ploidy - PLOIDY_BAND <= float(totals[copy_path].mean()) <= ploidy + PLOIDY_BAND


def get_log_mu_shift_start(fit_log_mu_shift: float | None, normal_fraction: float, ploidy: int) -> float:
    """A tumour clone's first `log Z_c`: the fit's carried to the ploidy, `k / (1 + (k - 1) theta)` at `k = P / 2`, or
    the ploidy's own `theta + (1 - theta) k` where the fit carries none; at diploid the fit's, 0 where none.

    Args:
        fit_log_mu_shift: the fit's `log Z_c`, or None.
        normal_fraction: `theta_c`.
        ploidy: P.

    Returns:
        The `log Z_c`.
    """
    k = ploidy / 2.0
    if k == 1.0:
        return 0.0 if fit_log_mu_shift is None else float(fit_log_mu_shift)
    if fit_log_mu_shift is None:
        return float(np.log(normal_fraction + (1.0 - normal_fraction) * k))
    return float(fit_log_mu_shift + np.log(k / (1.0 + (k - 1.0) * normal_fraction)))


def get_clone_status(own: dict[int, float]) -> dict[int, DecodeStatus]:
    """Each clone's `DecodeStatus` from its own loglike: `NO_PATH_IN_PLOIDY_BAND` at -inf, else `OK`.

    Args:
        own: each clone's own loglike.

    Returns:
        The status, by clone.
    """
    return {c: DecodeStatus.NO_PATH_IN_PLOIDY_BAND if v == -np.inf else DecodeStatus.OK for c, v in own.items()}


def get_observed_baf_profiles(counts: PseudobulkCounts) -> np.ndarray:
    """Each clone's observed B fraction per bin, `hmm_utils.get_normal_clone`'s profile.

    Args:
        counts: `(n_clones, n_obs)` the clones' pseudobulks.

    Returns:
        The `(n_obs, n_clones)` B fraction, NaN where a bin has no allele count.
    """
    total = np.asarray(counts.total_snp_umis, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            total > 0, np.asarray(counts.alt_snp_umis, dtype=np.float64).T / np.maximum(total, 1e-12), np.nan
        )


def get_merged_ab_decoded_clones(seglevel: Any, agreement: float) -> np.ndarray:
    """Each clone's integer clone: clones whose `(A, B)` agree on at least `agreement` of the bins are one.

    In id order, a clone joins the first earlier group it agrees with, and names a group (its own id) otherwise.

    Args:
        seglevel: with `clone{c} A` and `clone{c} B` columns, as `cnv_<ploidy>_seglevel.tsv`.
        agreement: the share of bins, `ab_decode.int_clone_agreement`.

    Returns:
        Each clone's group id.
    """
    n = sum(1 for c in seglevel.columns if str(c).startswith("clone") and str(c).endswith(" A"))
    named: list[tuple[int, np.ndarray]] = []
    merged = np.arange(n)
    for clone in range(n):
        profile = seglevel[[f"clone{clone} A", f"clone{clone} B"]].to_numpy(dtype=np.int64)
        match = next(
            (name for name, other in named if float(np.mean(np.all(profile == other, axis=1))) >= agreement), None
        )
        if match is None:
            named.append((clone, profile))
            match = clone
        merged[clone] = match
    return merged
