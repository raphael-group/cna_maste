"""Genomic segmentations as labellings of the input gene rows and the corresponding hierarchy a run creates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

DROPPED = -1
"""The label of a gene a segmentation does not keep."""


def get_dropped() -> int:
    """Return the dropped, -1."""
    return DROPPED


def get_contig_lengths(contig: Any) -> np.ndarray:
    """Rows per contig, in row order: the lengths of the runs of `contig`.

    Args:
        contig: each row's contig, a contig's rows one run.

    Returns:
        The `(n_contigs,)` int64 lengths, empty for no row.
    """
    contig = np.asarray(contig)
    if not contig.size:
        return np.zeros(0, dtype=np.int64)
    edges = np.concatenate(([0], np.flatnonzero(np.diff(contig)) + 1, [contig.size]))
    return np.diff(edges).astype(np.int64)


# TODO test all usage
@dataclass(frozen=True, eq=False)
class Genes:
    """The root: `df_gene_snp`'s gene rows, sorted by contig and position.

    Args:
        contig: per-gene arrays.
        start: per-gene arrays.
        end: per-gene arrays.
        name: per-gene arrays.
        row: per-gene arrays.
        key: per-gene arrays.
    """

    contig: np.ndarray
    start: np.ndarray
    end: np.ndarray
    name: np.ndarray
    """Each gene's name, its `gene` in the table."""
    row: np.ndarray
    """Each gene's row in the table it was read from."""
    key: np.ndarray
    """Each gene's index label in that table: what a later table is matched on."""

    @classmethod
    def from_table(cls, table: Any) -> Genes:
        """Read the rows with `is_interval` set, as `form_gene_snp_table` marks genes.

        Args:
            table: `df_gene_snp`.

        Returns:
            The `Genes`.
        """
        genes = np.asarray(table["is_interval"], dtype=bool)
        row = np.flatnonzero(genes)
        contig = np.asarray(table["CHR"], dtype=np.int64)[row]

        if row.size and np.any(np.diff(contig) < 0):
            msg = "gene rows must be sorted by contig"
            raise ValueError(msg)

        # TODO NB on row and key.
        return cls(
            contig=contig,
            start=np.asarray(table["START"], dtype=np.int64)[row],
            end=np.asarray(table["END"], dtype=np.int64)[row],
            name=np.asarray(table["gene"], dtype=object)[row],
            row=row,
            key=np.asarray(table.index)[row],
        )

    @property
    def n_rows(self) -> int:
        return int(self.row.size)


class Rows(NamedTuple):
    """A root that is not genes: the rows of a level, e.g. a count matrix's columns or a segmentation's segments."""

    n_rows: int


# TODO test all usage
# TODO GeneSegmentation rename
@dataclass(frozen=True, eq=False)
class Segmentation:
    """One segmentation: a label per row of its root, `DROPPED` where it keeps none; one label per row.

    Args:
        root: the genes, or the rows of a level.
        label: per-row labels.
        ids: the segment ids.
        name: the level.
    """

    root: Genes | Rows
    label: np.ndarray
    ids: np.ndarray
    """Each segment's id in the table it was read from, in label order."""
    name: str

    # -- construction --------------------------------------------------------

    @classmethod
    def from_table(cls, table: Any, key: str, genes: Genes | None = None) -> Segmentation:
        """Read `table[key]` at the gene rows, labels in id order.

        Id order is the array order (`groupby(key, sort=True)`), so it must be genomic order too.
        An id carried only by SNP rows has no gene and is refused.

        Args:
            table: with column `key`.
            genes: the root.

        Returns:
            The `Segmentation`.
        """
        genes = Genes.from_table(table) if genes is None else genes
        column = pd.Series(np.asarray(table[key]))
        is_gene = np.asarray(table["is_interval"], dtype=bool)

        # NB the table's gene rows located among the root's by index label: a
        #    later table may have dropped rows, never added a gene.
        where = pd.Index(genes.key).get_indexer(np.asarray(table.index)[is_gene])
        if np.any(where < 0):
            msg = f"{key}: the table holds genes the root does not"
            raise ValueError(msg)

        every = column.dropna().unique()

        # NB the known-range path's -1 is not a segment; refused by name.
        if np.issubdtype(np.asarray(every).dtype, np.number) and np.any(np.asarray(every) < 0):
            msg = (
                f"{key}: a negative id marks rows no segment covers (the known-range path's -1), which is not a segment"
            )
            raise ValueError(msg)
        at_genes = column[is_gene].reset_index(drop=True)
        present = at_genes.notna().to_numpy()

        ids, inverse = np.unique(at_genes[present].to_numpy(), return_inverse=True)

        if ids.size != every.size:
            msg = f"{key}: {every.size - ids.size} segments hold no gene"
            raise ValueError(msg)

        label = np.full(genes.n_rows, DROPPED, dtype=np.int64)
        label[where[present]] = inverse

        return cls.of(genes, label, ids=ids, name=key)

    @classmethod
    def of(cls, genes: Genes, label: Any, *, ids: Any = None, name: str) -> Segmentation:
        """Build a labelling of `genes`, checked: contiguous, one contig each, genomic order.

        Args:
            genes: the root.
            label: per-gene labels.
            ids: the segment ids.
            name: the level.

        Returns:
            The `Segmentation`.
        """
        label = np.asarray(label, dtype=np.int64)
        kept = label[label != DROPPED]

        if kept.size and (np.any(np.diff(kept) < 0) or np.any(np.diff(kept) > 1)):
            msg = f"{name}: labels are not contiguous runs in genomic order"
            raise ValueError(msg)

        if kept.size and kept[0] != 0:
            msg = f"{name}: labels must start at 0"
            raise ValueError(msg)

        n = int(kept[-1]) + 1 if kept.size else 0
        position = np.flatnonzero(label != DROPPED)
        first = position[np.searchsorted(kept, np.arange(n), side="left")]
        last = position[np.searchsorted(kept, np.arange(n), side="right") - 1]

        if np.any(genes.contig[first] != genes.contig[last]):
            msg = f"{name}: a segment spans two contigs"
            raise ValueError(msg)

        # NB a dropped gene may fall inside a segment: its extent runs from first to last kept gene.
        return cls(
            root=genes,
            label=label,
            ids=np.arange(n) if ids is None else np.asarray(ids),
            name=name,
        )

    @classmethod
    def from_terms(cls, n_rows: int, rows: Any, segments: Any, n_segments: int, *, name: str) -> Segmentation:
        """Build a labelling of `n_rows` rows from (row, segment) terms, each row in one segment at most.

        Args:
            n_rows: the root's rows.
            rows: each term's row.
            segments: each term's segment.
            n_segments: the segments.
            name: the level.

        Returns:
            The `Segmentation`, its root `Rows(n_rows)`.

        Raises:
            ValueError: where a row takes a second label.
        """
        rows, segments = np.asarray(rows, dtype=np.int64), np.asarray(segments, dtype=np.int64)
        if np.unique(rows).size != rows.size:
            msg = f"{name}: {rows.size - np.unique(rows).size} rows take a second label"
            raise ValueError(msg)
        label = np.full(n_rows, DROPPED, dtype=np.int64)
        label[rows] = segments
        return cls(Rows(n_rows), label, np.arange(n_segments), name)

    @property
    def genes(self) -> Genes:
        """The root, where it is genes.

        Returns:
            The `Genes`.

        Raises:
            TypeError: where the root is the rows of a level.
        """
        if not isinstance(self.root, Genes):
            msg = f"{self.name}: its root is not genes"
            raise TypeError(msg)
        return self.root

    @property
    def terms(self) -> tuple[np.ndarray, np.ndarray]:
        """The kept rows and each one's segment.

        Returns:
            The rows and the segments.
        """
        rows = np.flatnonzero(self.label != DROPPED)
        return rows, self.label[rows]

    def coarsen(self, parent: Any, *, name: str) -> Segmentation:
        """Coarsen: `parent[k]` is segment `k`'s new label, `DROPPED` drops it.

        Args:
            parent: per-segment new labels.
            name: the level.

        Returns:
            The coarser `Segmentation`.
        """
        parent = np.asarray(parent, dtype=np.int64)
        label = np.where(self.label == DROPPED, DROPPED, parent[self.label])
        return Segmentation.of(self.genes, label, name=name)

    def select(self, keep: Any, *, name: str) -> Segmentation:
        """Drop some segments; survivors keep their genes.

        Args:
            keep: per-segment mask.
            name: the level.

        Returns:
            The `Segmentation`.
        """
        keep = np.asarray(keep, dtype=bool)
        renumber = np.where(keep, np.cumsum(keep) - 1, DROPPED)
        selected = self.coarsen(renumber, name=name)
        return Segmentation(self.root, selected.label, self.ids[keep], name)

    @property
    def n_segments(self) -> int:
        return int(self.ids.size)

    @property
    def first(self) -> np.ndarray:
        """Each segment's first gene.

        Returns:
            The gene indices.
        """
        kept = np.flatnonzero(self.label != DROPPED)
        order = np.searchsorted(self.label[kept], np.arange(self.n_segments), "left")
        return kept[order]

    @property
    def last(self) -> np.ndarray:
        """Each segment's last gene.

        Returns:
            The gene indices.
        """
        kept = np.flatnonzero(self.label != DROPPED)
        order = np.searchsorted(self.label[kept], np.arange(self.n_segments), "right")
        return kept[order - 1]

    @property
    def contig(self) -> np.ndarray:
        values: np.ndarray = self.genes.contig[self.first]
        return values

    @property
    def start(self) -> np.ndarray:
        values: np.ndarray = self.genes.start[self.first]
        return values

    @property
    def end(self) -> np.ndarray:
        values: np.ndarray = self.genes.end[self.last]
        return values

    @property
    def length(self) -> np.ndarray:
        """Each segment's extent in base pairs, `end - start`.

        Returns:
            The lengths.
        """
        length: np.ndarray = self.end - self.start
        return length

    @property
    def gene(self) -> np.ndarray:
        """Each segment's first gene, by its index label in the table the root was read from.

        Returns:
            The labels.
        """
        gene: np.ndarray = self.genes.key[self.first]
        return gene

    @property
    def lengths(self) -> np.ndarray:
        """Segments per contig, in contig order: the grid a lattice restarts on.

        Returns:
            The counts.
        """
        return get_contig_lengths(self.contig)

    @property
    def boundary(self) -> np.ndarray:
        """True at each contig's last segment, which has no successor in its contig.

        Returns:
            The boolean mask.
        """
        contig = self.contig
        last = np.ones(self.n_segments, dtype=bool)
        last[:-1] = contig[1:] != contig[:-1]
        return last

    def refines(self, other: Segmentation) -> bool:
        """Whether every segment here lies inside one of `other`'s, over the genes both keep.

        Args:
            other: a `Segmentation` of the same root.

        Returns:
            The bool.
        """
        both = (self.label != DROPPED) & (other.label != DROPPED)
        fine, coarse = self.label[both], other.label[both]
        mapped = np.full(self.n_segments, DROPPED, dtype=np.int64)
        mapped[fine] = coarse
        return bool(np.all(mapped[fine] == coarse))

    def aggregate(self, values: Any, ufunc: Any = np.add) -> np.ndarray:
        """Reduce a per-gene array over each segment's genes.

        Args:
            values: per gene.
            ufunc: the reduction.

        Returns:
            The per-segment array.
        """
        values = np.asarray(values)

        if values.shape[0] != self.root.n_rows:
            msg = f"{self.name}: aggregate takes one entry per gene"
            raise ValueError(msg)

        kept = np.flatnonzero(self.label != DROPPED)
        offsets = np.searchsorted(self.label[kept], np.arange(self.n_segments), "left")
        reduced: np.ndarray = ufunc.reduceat(values[kept], offsets, axis=0)
        return reduced

    def broadcast(self, values: Any, fill: Any = np.nan) -> np.ndarray:
        """Read a per-segment array at each gene; a dropped gene gets `fill`.

        Args:
            values: per segment.
            fill: for dropped genes.

        Returns:
            The per-gene array.
        """
        values = np.asarray(values)

        if values.shape[0] != self.n_segments:
            msg = f"{self.name}: broadcast takes one entry per segment"
            raise ValueError(msg)

        dtype = np.result_type(values.dtype, np.asarray(fill).dtype)
        out = np.full((self.root.n_rows, *values.shape[1:]), fill, dtype=dtype)
        kept = self.label != DROPPED
        out[kept] = values[self.label[kept]]
        return out


# TODO rename RUN_GENOMIC_SEGMENTATIONS
LEVELS = (
    "reference_genes",
    "initial_blocks",
    "phasing_segments",
    "phased_bins",
    "normal_filtered_bins",
    "normal_rebinned_bins",
)


def get_levels() -> tuple[str, ...]:
    """Return the run's segmentation levels, `LEVELS`, in the order it makes them.

    Returns:
        The level names.
    """
    return LEVELS


# TODO test all usage
class TrackedSegmentation:
    """The run's segmentations of one root: a column of labels per level, one row per gene, `DROPPED` where none.

    `update` appends a level under its name; equal labels in a column share that level's segment, and each column's
    `lengths` is that stage's.

    Args:
        table: `df_gene_snp`, whose gene rows are the root.
    """

    def __init__(self, table: Any) -> None:
        self.genes = Genes.from_table(table)
        self.levels: dict[str, Segmentation] = {}
        self.update(np.arange(self.genes.n_rows), "reference_genes")

    def update(self, labels: Any, name: str) -> Segmentation:
        """Append `labels` as level `name`, checked by `Segmentation.of`; a name already recorded is refused.

        Args:
            labels: one per gene, or a `Segmentation` of this root.
            name: the level.

        Returns:
            The `Segmentation`.
        """
        if name in self.levels:
            msg = f"{name}: already a level"
            raise ValueError(msg)
        label = labels.label if isinstance(labels, Segmentation) else labels
        level = Segmentation.of(self.genes, label, name=name)
        self.levels[name] = level
        return level

    def update_from_table(self, table: Any, key: str, name: str) -> Segmentation:
        """Append `table[key]` read at the gene rows (`Segmentation.from_table`) as level `name`.

        Args:
            table: with column `key`.
            name: the level.

        Returns:
            The `Segmentation`.
        """
        return self.update(Segmentation.from_table(table, key, genes=self.genes), name)

    def __getitem__(self, name: str) -> Segmentation:
        return self.levels[name]

    @property
    def names(self) -> list[str]:
        return list(self.levels)

    @property
    def table(self) -> np.ndarray:
        """The `(genes, levels)` int64 labels, levels in the order made.

        Returns:
            The array.
        """
        return np.stack([s.label for s in self.levels.values()], axis=1)

    def frame(self) -> pd.DataFrame:
        """The genes' contig, start, end and name, then one column per level.

        Returns:
            The data frame.
        """
        out = pd.DataFrame(
            {"CHR": self.genes.contig, "START": self.genes.start, "END": self.genes.end, "gene": self.genes.name}
        )
        for name, level in self.levels.items():
            out[name] = level.label
        return out

    def get_snps(self, table: Any) -> pd.DataFrame:
        """The SNP rows of `table`, the table this was built from, each with its gene and its block's first gene.

        A SNP's gene is the nearest gene row before it named as its `gene`; its block's gene, the first gene row of
        its `block_id`. Either is `DROPPED` where there is none.

        Args:
            table: `df_gene_snp` with `snp_id`, `gene` and `block_id`.

        Returns:
            `CHR`, `POS` and `snp_id`, then `gene` and `block_gene` as indices into the genes, in table order.
        """
        position = np.arange(len(table))
        genes = pd.DataFrame(
            {"position": position[self.genes.row], "name": np.asarray(table["gene"])[self.genes.row]}
        ).assign(gene=np.arange(self.genes.n_rows))
        is_snp = pd.notna(table["snp_id"]).to_numpy()
        snps = pd.DataFrame(
            {
                "CHR": np.asarray(table["CHR"], dtype=np.int64)[is_snp],
                "POS": np.asarray(table["START"], dtype=np.int64)[is_snp],
                "snp_id": np.asarray(table["snp_id"])[is_snp],
                "position": position[is_snp],
                "name": np.asarray(table["gene"])[is_snp],
                "block_id": np.asarray(table["block_id"])[is_snp],
            }
        )
        named = pd.merge_asof(
            snps.dropna(subset=["name"]), genes, on="position", by="name", allow_exact_matches=False
        ).set_index("position")["gene"]
        gene = named.reindex(snps["position"]).fillna(DROPPED).to_numpy(np.int64)
        block_of_gene = np.asarray(table["block_id"])[self.genes.row]
        first = pd.Series(np.arange(self.genes.n_rows)).groupby(block_of_gene).first()
        block_gene = first.reindex(snps["block_id"]).fillna(DROPPED).to_numpy(np.int64)
        return snps[["CHR", "POS", "snp_id"]].assign(gene=gene, block_gene=block_gene)

    def get_snp_frame(self, snps: pd.DataFrame, phase: Any) -> pd.DataFrame:
        """The SNPs' contig, position, id and gene name, one column per level as `frame` has them, then `phase`.

        A SNP takes its gene's name and label at `reference_genes`, and its block's gene's label and `phase` at every
        other level; no name, or `DROPPED`, where it has no such gene.

        Args:
            snps: `get_snps`' frame.
            phase: per gene, `DROPPED` where none.

        Returns:
            The data frame, one row per SNP.
        """
        out = snps[["CHR", "POS", "snp_id"]].copy()
        gene = snps["gene"].to_numpy()
        out["gene"] = np.where(gene >= 0, self.genes.name[gene], None)
        for name, level in self.levels.items():
            gene = snps["gene" if name == "reference_genes" else "block_gene"].to_numpy()
            out[name] = np.where(gene >= 0, level.label[gene], DROPPED)
        block_gene = snps["block_gene"].to_numpy()
        out["phase"] = np.where(block_gene >= 0, np.asarray(phase)[block_gene], DROPPED)
        return out


# TODO test all usage
# TODO why separate?
def phasing_segments(blocks: Segmentation, refined_lengths: Any) -> np.ndarray:
    """Label `blocks`' genes by `initial_phase_given_partition`'s segments.

    `refined_lengths` counts blocks per phase segment in order: block `k` lies in segment `j` for the `j`-th run.

    Args:
        blocks: the block `Segmentation`.
        refined_lengths: blocks per phase segment.

    Returns:
        The phase-segment `Segmentation`.
    """
    parent = np.repeat(np.arange(len(refined_lengths)), np.asarray(refined_lengths, dtype=np.int64))
    if parent.size != blocks.n_segments:
        msg = f"phase segments cover {parent.size} blocks of {blocks.n_segments}"
        raise ValueError(msg)
    return blocks.coarsen(parent, name="phasing_segments").label
