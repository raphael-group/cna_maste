"""Operations on a clone labelling: the floor merge (`get_merged_labelling`), the normal-first order
(`get_reindexed_labelling`), and the neutral state's pin (`get_neutral_pinned`)."""

from typing import Any

import numpy as np

from cnamaste.config import start_time
from cnamaste.emission import CnaEmissionConfounders
from cnamaste.hmm_nophasing import get_hmm_baf_eps, get_neutral_state, get_path_baf_profiles
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import get_normal_clone
from cnamaste.logger import get_logger
from cnamaste.spatio_genomic_counts import SpatioGenomicCounts

logger = get_logger(__name__, start_time=start_time)


def get_merged_labelling(
    labelling: np.ndarray, total_snp_umis: Any, min_spots: int, min_snp_umis: float
) -> tuple[np.ndarray, list[list[int]]]:
    """Each clone under `min_spots` spots or `min_snp_umis` SNP UMIs joins the passing clone with most SNP UMIs.

    Args:
        labelling: `(n_spots,)` each spot's clone.
        total_snp_umis: the spots' SNP UMIs, `(n_spots, n_bins)`.
        min_spots: `hmrf.min_spots_per_clone`.
        min_snp_umis: the SNP UMI floor.

    Returns:
        The merged labelling, each group's index, and the groups, each a passing clone and the clones it took in;
        a single clone keeps its labelling.

    Raises:
        RuntimeError: where every clone fails a floor.
    """
    clones = np.unique(labelling)
    if len(clones) == 1:
        return labelling, [[labelling[0]]]

    snp_umis = {c: SpatioGenomicCounts.of(total_snp_umis).spots(labelling == c).sum() for c in clones}
    few_spots = [c for c in clones if np.sum(labelling == c) < min_spots]
    few_snp_umis = [c for c in clones if snp_umis[c] < min_snp_umis]
    logger.info(f"Found {len(few_spots)} clones with < {min_spots} spots: {few_spots}")
    logger.info(f"Found {len(few_snp_umis)} clones with < {min_snp_umis:_} SNP UMIs: {few_snp_umis}")

    failed = [c for c in clones if c in few_spots or c in few_snp_umis]
    logger.info(f"Found {len(failed)} new clones failing thresholds on min. spots or min. SNP umis.")
    passed = [c for c in clones if c not in failed]
    if len(passed) == 0:
        logger.error("All clones failed min. spots or min. SNP UMIs thresholds; cannot proceed with merging.")
        raise RuntimeError()

    groups = [[c] for c in passed]
    for c in failed:
        best = int(np.argmax([snp_umis[p] for p in passed]))
        logger.warning(f"Assigning failed clone {c} to clone {[passed[best]]} (with largest SNP UMIs).")
        groups[best].append(c)

    group_of = {c: g for g, group in enumerate(groups) for c in group}
    return np.array([group_of[c] for c in labelling]), groups


def get_reindexed_labelling(
    labelling: np.ndarray, paths: np.ndarray, p_binom: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The clones renumbered: the normal clone 0 (`get_normal_clone`), the rest by spot count, fewest first.

    Args:
        labelling: `(n_spots,)` each spot's clone, 0..N-1.
        paths: `(n_bins, n_clones)` each clone's decoded state.
        p_binom: `(n_states,)` the states' B-allele fraction.

    Returns:
        The renumbered labelling, and the old clone of each new one.
    """
    normal = get_normal_clone(get_path_baf_profiles(paths, p_binom), get_hmm_baf_eps())
    clones, n_spots = np.unique(labelling, return_counts=True)
    rest = clones != normal
    order = np.concatenate(([normal], clones[rest][np.argsort(n_spots[rest])])).astype(int)
    logger.info(f"Remapping clone index: {normal} (normal) to 0, otherwise sorted by spot count.")

    new_of = np.zeros(np.max(clones) + 1, dtype=int)
    new_of[order] = np.arange(len(order))
    return new_of[labelling], order


def get_neutral_pinned(
    parameters: CnaHmmParameters, confounders: CnaEmissionConfounders, paths: np.ndarray, fitted_baf: bool
) -> tuple[CnaHmmParameters, CnaEmissionConfounders]:
    """`log_mu`'s scale fixed under `hmrf.apply_logmu_shift`: the neutral state at `mu = 1`, each `log Z_c` moved
    by the same amount, so no likelihood or path moves; unchanged where every `log Z_c` is 0.

    Args:
        parameters: the fit's.
        confounders: the fit's.
        paths: `(n_bins, n_clones)` each clone's decoded state.
        fitted_baf: whether the neutral state reads the fitted `p_binom`.

    Returns:
        The parameters and confounders, pinned.
    """
    shift = np.asarray(confounders.log_mu_shift, dtype=np.float64)
    if not np.any(shift):
        return parameters, confounders
    neutral = get_neutral_state(paths, parameters.p_binom if fitted_baf else None)
    pin = float(np.asarray(parameters.log_mu).reshape(-1)[neutral])
    logger.info(f"logmu_shift: state {neutral} pinned to mu = 1, from log mu = {pin:.6f}.")
    return parameters._replace(log_mu=np.asarray(parameters.log_mu) - pin), confounders._replace(
        log_mu_shift=shift - pin
    )
