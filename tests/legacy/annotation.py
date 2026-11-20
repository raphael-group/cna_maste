"""A frozen copy of `cnamaste.annotation.load_clone_labels` as PR- #101 held it, verbatim, whose baseline T- #84's
`normal_baseline_sized` replaces; `tests/legacy/test_legacy.py` holds the new code byte-equal to it."""

import numpy as np
import pandas as pd

from cnamaste.annotation import logger
from cnamaste.config import get_global_config
from legacy.pseudobulk import merge_pseudobulk_by_index_mix
from legacy.spatio_genomic_counts import SpatioGenomicPairCounts


def load_clone_labels(single_X=None, config=None):
    if config is None:
        config = get_global_config()

    logger.warning(f"Assuming known clone labels={config.annotation.clone_label}")

    clone_id = (
        pd.read_csv(config.annotation.clone_label, sep="\t", index_col=0)["labels"]
        .str.replace("clone_", "")
        .str.replace("normal", "-1")
        .astype(int)
        .to_numpy()
    )
    clone_id += 1

    initial_clone_index_baf = [np.where(clone_id == xx)[0] for xx in np.unique(clone_id)]

    if single_X is None:
        return initial_clone_index_baf, None

    known_rdr_normal = SpatioGenomicPairCounts.of(single_X).rdr.spot_sums(clone_id == 0)
    known_rdr_normal = known_rdr_normal / np.sum(known_rdr_normal)

    # TODO
    # bidx_inconfident = np.where(
    #     known_rdr_normal < config.quality.min_normal_count_perbin
    # )[0]
    # known_rdr_normal[bidx_inconfident] = 0

    spots_coverage = SpatioGenomicPairCounts.of(single_X).rdr.sum(axis=0)

    known_single_base_nb_mean = known_rdr_normal.reshape(-1, 1) @ spots_coverage.reshape(1, -1)

    _, known_base_nb_mean, _, _ = merge_pseudobulk_by_index_mix(
        single_X,
        known_single_base_nb_mean,
        np.zeros_like(known_single_base_nb_mean),
        initial_clone_index_baf,
        None,
    )

    return initial_clone_index_baf, known_base_nb_mean
