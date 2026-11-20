from pathlib import Path

import numpy as np
import pandas as pd

from cnamaste.config import get_global_config, start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def load_celltype(celltype_file: str | Path) -> pd.DataFrame:
    """Read the cell type per spot from a tab-separated annotation file with `barcode` and `cell_type` columns.

    Args:
        celltype_file: the annotation file, e.g. `spot_annotation.tsv`, plain or `.gz`.

    Returns:
        Frame of barcode, cell_type, one row per spot, in file order.

    Raises:
        ValueError: where either column is missing.
    """
    df_celltype = pd.read_csv(celltype_file, sep="\t", usecols=["barcode", "cell_type"], dtype=str)[
        ["barcode", "cell_type"]
    ]

    spots_per_celltype = df_celltype["cell_type"].value_counts(dropna=False)

    logger.info(
        f"Read {len(df_celltype):_} spots of {len(spots_per_celltype)} cell types from {celltype_file}:\n"
        f"{spots_per_celltype.to_string()}"
    )

    return df_celltype


def load_clone_labels(counts=None, config=None):
    """Read the known clone labels; with `counts`, also each clone's pseudobulk baseline, sized by the normal clone.

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`, or None.
        config: the run's.

    Returns:
        The spots per clone, and the `(n_obs, n_clones)` baseline or None.
    """
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

    if counts is None:
        return initial_clone_index_baf, None

    # NB clone 0 the normal spots; no bin zeroed under quality.min_normal_count_perbin.
    clone = np.searchsorted(np.unique(clone_id), clone_id)
    known = counts.with_labelling("normal_spot", clone_id == 0).with_labelling("clone", clone)
    known_base_nb_mean = known.get_pseudobulk("clone").normal_baseline_sized.T

    return initial_clone_index_baf, known_base_nb_mean
