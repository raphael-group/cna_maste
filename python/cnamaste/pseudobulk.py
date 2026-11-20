from typing import Any

import numpy as np

from cnamaste.config import start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def log_pseudobulk(pseudobulk: Any, spots_per_label: list[int]) -> None:
    """Log each label's pseudobulk: its median BAF, UMIs per spot, median RDR and UMI percentiles.

    Args:
        pseudobulk: the `(n_labels, n_bins)` `spatio_genomic_counts.PseudobulkCounts`.
        spots_per_label: each label's spots.
    """
    X, base_nb_mean, total_bb_RD = pseudobulk.X, pseudobulk.normal_baseline_sized.T, pseudobulk.total_snp_umis.T
    for k, n_spots in enumerate(spots_per_label):
        if n_spots == 0:
            logger.warning(f"Clone {k} has no cells, skipping")

    for k, n_spots in enumerate(spots_per_label):
        percentiles = [50, 75, 90, 95, 99, 100]

        # TODO may be NAN if insufficient spots to aggregate at least one snp-covering umi for a segment.
        bafs = X[:, 1, k] / total_bb_RD[:, k]

        valid_rdr = base_nb_mean[:, k] > 0

        # NB base_nb_mean is initially non-defined.
        if np.any(valid_rdr):
            rdrs = X[:, 0, k] / base_nb_mean[:, k]
        else:
            rdrs = np.nan * np.ones_like(X[:, 0, k])

        logger.info(f"Found median (finite) baf={np.median(bafs[np.isfinite(bafs)]):.3f} for clone {k}.")
        logger.info(
            f"Found {n_spots} spots, mean umis per spot={np.sum(X[:, 0, k]) / n_spots:.3f} and mean snp-covering umis per spot={np.sum(total_bb_RD[:, k]) / n_spots:.3f} for clone {k}"
        )

        if np.any(valid_rdr):
            if not np.isclose(
                np.nansum(X[:, 0, k]),
                np.nansum(base_nb_mean[:, k]),
                rtol=1e-5,
                atol=1e-6,
            ):
                logger.warning(
                    f"Expected consistency between normal baseline normalization total umi for the clone, {np.nansum(X[:, 0, k])} != {np.sum(base_nb_mean[:, k])}"
                )

            logger.info(
                f"Found median umis={np.median(X[:, 0, k])} and median RDR={np.median(rdrs[valid_rdr]):.3f} for clone {k} with {100.0 * np.mean(valid_rdr > 0.0):.3f}% valid."
            )
            logger.info(f"Found umi percentiles=\n{np.percentile(X[:, 0, k], percentiles)}\nfor\n{percentiles} [%].")

    logger.info(f"Merged single_X to pseudobulk of shape {X.shape[2]}.")
