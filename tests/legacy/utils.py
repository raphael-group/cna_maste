"""A frozen copy of `cnamaste.utils.write_binned_data` as PR- #101 held it, verbatim, which T- #84's count containers
replace; `tests/legacy/test_legacy.py` holds the new code byte-equal to it."""

import numpy as np

from cnamaste.results import write_npz
from legacy.spatio_genomic_counts import npz_entries


def write_binned_data(directory, lengths, barcodes, **counts):
    """`binned_data.npz`: the per-spot bins the pages draw from, as CalicoST's holds them, so a page is redrawn after
    the run on another grouping of the spots (`spatio_genomic_counts.from_npz` reads it).

    Each of `counts` goes as `spatio_genomic_counts.npz_entries` writes it, sparse where it is sparse; the file is
    deflated where `single_X` is dense, and its CSR entries stored otherwise (`write_npz`).

    Args: `directory` the run's, None to write nothing; `lengths` bins per contig; `barcodes` the spots'; `counts`
    name -> `SpatioGenomicPairCounts`, `SpatioGenomicCounts` or array: `single_X`, `single_base_nb_mean`,
    `single_total_bb_RD`.
    """
    entries = {k: v for name, value in counts.items() for k, v in npz_entries(name, value).items()}
    write_npz(
        directory,
        "binned_data.npz",
        compress="single_X" in entries,
        lengths=np.asarray(lengths),
        barcodes=np.asarray(barcodes, dtype=str),
        **entries,
    )
