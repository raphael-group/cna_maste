import functools
import os
import re

import numpy as np
import pandas as pd
import polars as pl
import pyranges as pr

from cnamaste.config import get_global_config, start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


cancer_gene_patterns = (
    r"^MT-",  # Mitochondrial genes
    r"^S100A6$",  # Exact match for S100A6
    r"^B2M$",  # Exact match for B2M
    r"^HLA-B$",  # Exact match for HLA-B
    r"^TPT1$",  # Exact match for TPT1
    r"^FTL$",  # Exact match for FTL
    r"^FTH1$",  # Exact match for FTH1
    r"^RPL",  # Ribosomal protein L genes
    r"^RPS",  # Ribosomal protein S genes
)

pl.Config.set_tbl_cols(-1)


def exp_cancer_gene(gene_name):
    return any(re.match(pattern, gene_name) for pattern in cancer_gene_patterns)


def get_reference_genes(hgtable_file):
    config = get_global_config()

    # TODO HACK use legacy only for now.
    if True or config.run.legacy:
        logger.info_once(f"Assuming legacy gene annotation={hgtable_file}.")

        # NB read gene info and keep only chr1-chr22 and genes appearing in adata
        #    name2  chrom  cdsStart    cdsEnd
        df_hgtable = pd.read_csv(hgtable_file, header=0, index_col=0, sep="\t")
        df_hgtable = df_hgtable[df_hgtable.chrom.isin([f"chr{i}" for i in range(1, 23)])]
        df_hgtable = df_hgtable.rename(
            columns={
                "chrom": "Chromosome",
                "cdsStart": "Start",
                "cdsEnd": "End",
                "name2": "gene",
            }
        )
    else:
        # TODO rename_attr=True
        df_hgtable = pr.read_gtf(config.references.annotation_file, full=True)
        df_hgtable = df_hgtable.query("Feature == 'gene'").drop_duplicates("gene_id", keep="first")
        df_hgtable = df_hgtable[["Chromosome", "Start", "End", "gene_name", "gene_id"]]

        # NB drop .5 suffix to gene_id.
        df_hgtable["gene_id"] = df_hgtable["gene_id"].str.replace(r"\.\d+$", "", regex=True)
        df_hgtable = df_hgtable[df_hgtable.Chromosome.isin([f"chr{i}" for i in range(1, 23)])]
        df_hgtable = df_hgtable.rename(columns={"gene_name": "gene"})

    logger.info(f"Read reference genes:\n{df_hgtable}")

    df_gene = pd.DataFrame(
        {
            "CHR": [int(x[3:]) for x in df_hgtable.Chromosome.to_numpy()],
            "START": df_hgtable.Start.to_numpy(),
            "END": df_hgtable.End.to_numpy(),
            "snp_id": None,
            "gene": df_hgtable.gene.to_numpy(),
            "is_interval": True,
        }
    )

    return df_gene


AUTOSOMES = tuple(f"chr{i}" for i in range(1, 23))
"""The genetic map's contigs kept, `chrN` read as the integer N."""


def get_reference_recomb_rates(geneticmap_file):
    """The genetic map's autosomes (`_genetic_map`), read once per file and run, and re-read where the file changed.

    The table is shared and read-only: a caller reads it and does not write to it.

    Args:
        geneticmap_file: the genetic map.

    Returns:
        The shared table.
    """

    stat = os.stat(geneticmap_file)
    return _genetic_map(str(geneticmap_file), stat.st_size, stat.st_mtime_ns)


@functools.lru_cache(maxsize=1)
def _genetic_map(geneticmap_file, size, mtime_ns):
    """The genetic map's autosomes, contigs as integers, rows sorted by (contig, position), logical order.

    Index labels are the rows' positions in the file. Bitwise the per-row form it replaces (`isin`, `str.replace`,
    `sort_values(["chrom", "pos"])`) on any map: same `pandas` parser, contigs read as a
    category, rows ordered by a stable integer `lexsort`. pyarrow's parser is not used: it rounds differently.

    Args:
        geneticmap_file: the map.
        size: its stat, the cache key.
        mtime_ns: its stat, the cache key.

    Returns:
        The sorted table.
    """

    raw = pd.read_csv(geneticmap_file, header=0, sep="\t", dtype={"chrom": "category"})
    number = np.array(
        [AUTOSOMES.index(n) + 1 if n in AUTOSOMES else 0 for n in raw.chrom.cat.categories], dtype=np.int64
    )
    contig = number[raw.chrom.cat.codes.to_numpy()]

    kept = np.flatnonzero(contig > 0)
    pos = raw.pos.to_numpy()
    order = kept[np.lexsort((pos[kept], contig[kept]))]

    df = pd.DataFrame(
        {name: contig[order] if name == "chrom" else raw[name].to_numpy()[order] for name in raw.columns},
        index=pd.Index(order),
    )

    logger.info_once(f"Read reference recombination rates from {geneticmap_file}:\n{df.head()}")

    return df
