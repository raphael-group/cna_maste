# ruff: noqa
"""A frozen copy of `cnamaste.count_encoder.CountEncoder` as PR- #103 held it, verbatim, which T- #84's encodings on
`SpatioGenomicLabelledMultiCounts` and `PseudobulkCounts` replace; `tests/legacy/test_legacy.py` holds the new code
byte-equal to it."""

import numpy as np

from cnamaste.config import start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


class CountEncoder:
    """Each spot's `(count, total)` pairs as their unique rows and, per observation, the index of its row.

    A value computed once per unique pair is decoded to every observation by an index gather.

    Args: `obs_count`, `total_count` (n_obs, n_spots) counts; `common_zero_depth` collapses zero-total pairs to (0, 0).
    """

    def __init__(self, obs_count, total_count, common_zero_depth):
        self.n_obs = obs_count.shape[0]
        self.n_spots = obs_count.shape[1]

        # NB per spot: the unique pairs' counts and totals, and each observation's pair index.
        self.unique_obs, self.unique_total, self.inverse = self.construct_unique_encoding(
            obs_count, total_count, common_zero_depth
        )

    def decode(self, values, spot, axis):
        """Gather `values` over the spot's unique pairs to every observation, along `axis`.

        Args: `values` with `n_unique` entries along `axis`; `spot` the spot index; `axis` the unique-pair axis.
        Returns: `values` with `n_obs` entries along `axis`.
        """
        n_unique = self.unique_obs[spot].shape[0]
        if values.shape[axis] != n_unique:
            raise ValueError(f"values {values.shape} have no axis {axis} of {n_unique} unique pairs.")
        return np.take(values, self.inverse[spot], axis=axis)

    @property
    def compression_rate(self):
        total_original_elements = self.n_obs * self.n_spots

        if total_original_elements == 0:
            return 0.0

        total_compressed_elements = sum(u.shape[0] for u in self.unique_obs)

        return 1.0 - total_compressed_elements / total_original_elements

    @staticmethod
    def prepared(obs_count, total_count, common_zero_depth):
        """`(counts, totals)` as encoded: a zero total's count set to 0 where `common_zero_depth`; otherwise as given.

        Non-integer pairs are not rounded; the M step's gradient scores these (`hmm_mstep_grad`).

        Args: `obs_count`, `total_count` (n_obs, n_spots) counts; `common_zero_depth` zeroes counts at zero total.
        Returns: copies `(counts, totals)`, (n_obs, n_spots) each.
        """
        counts = np.array(obs_count, copy=True)
        totals = np.array(total_count, copy=True)

        # NB all entries with 0 total counts collapse into a single (0, 0) state.
        if common_zero_depth:
            zero_mask = totals == 0
            counts[zero_mask] = 0
            totals[zero_mask] = 0

        return counts, totals

    @staticmethod
    def construct_unique_encoding(obs_count, total_count, common_zero_depth):
        """Each spot's unique `(count, total)` pairs, as `np.unique` orders them, and each observation's pair index.

        Args: `obs_count`, `total_count` (n_obs, n_spots) counts; `common_zero_depth` as `prepared`.
        Returns: per spot, the unique counts, the unique totals and the `int32` inverse.
        """
        counts, totals = CountEncoder.prepared(obs_count, total_count, common_zero_depth)
        unique_obs, unique_total, inverses = [], [], []

        for s in range(counts.shape[1]):
            pairs, inverse = np.unique(np.vstack([counts[:, s], totals[:, s]]).T, axis=0, return_inverse=True)
            unique_obs.append(pairs[:, 0])
            unique_total.append(pairs[:, 1])
            # NB 4 bytes an observation, where the one-hot CSR held 20 and a sparse matmul per decode.
            inverses.append(inverse.reshape(-1).astype(np.int32))

        return unique_obs, unique_total, inverses
