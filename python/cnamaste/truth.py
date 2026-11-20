"""The truth for a simulated sample: its true clones, their spot coordinates and allele-specific copies."""

from __future__ import annotations

import dataclasses
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cnamaste.io import read_numpy, read_text_table


@dataclass(frozen=True)
class Truth:
    """A CalicoST sample's planted clones and copies; a `cnamaste.sim` draw's tree, phase and slices too
    (`read_sim_truth`), its counts read on demand.

    Args:
        barcodes: each documented at its attribute.
        coords: each documented at its attribute.
        labels: each documented at its attribute.
        clones: each documented at its attribute.
        profile: each documented at its attribute.
        sample_ids: each documented at its attribute.
        tree: each documented at its attribute.
        phase: each documented at its attribute.
        sample: each documented at its attribute.
    """

    barcodes: np.ndarray
    """Barcode per spot, as `truth_clone_labels.tsv` orders them."""
    coords: np.ndarray
    """`(n, 2)` float `x, y` per barcode."""
    labels: np.ndarray
    """Planted clone per barcode: `normal` 0, `clone_k` `k + 1`, as `remap_clone_num` numbers them."""
    clones: tuple[str, ...]
    profile: pd.DataFrame
    """`chr start end`, then `<clone>_A_copy`, `<clone>_B_copy` per segment."""
    sample_ids: np.ndarray | None = None
    """Slice per barcode, a `cnamaste.sim` draw's; None otherwise."""
    tree: pd.DataFrame | None = None
    """`truth_tree.tsv`: per node its `parent`, then per event `node parent chr start end A B`, in draw order."""
    phase: np.ndarray | None = None
    """`truth_phase.npy`: per SNP, True where the written A and B counts are exchanged."""
    sample: Path | None = None
    """The draw's directory, its counts read from it on demand."""

    @property
    def n_clones(self) -> int:
        return len(self.clones)

    def get_simulation(self) -> dict[str, Any]:
        """The draw's `Simulation`, as `manifest.json` records it.

        Returns:
            The record.
        """
        assert self.sample is not None
        record: dict[str, Any] = json.loads((self.sample / "manifest.json").read_text())["simulation"]
        return record

    def get_umis(self) -> sp.csr_matrix:
        """Each barcode's gene UMIs, the slices stacked in the truth's barcode order.

        Returns:
            `(n_barcodes, n_genes)` counts, the genes the simulation's baseline orders.

        Raises:
            ValueError: where a slice's barcodes are not the truth's.
        """
        assert self.sample is not None and self.sample_ids is not None
        found = [
            anndata.read_h5ad(self.sample / i / "filtered_feature_bc_matrix.h5ad")
            for i in dict.fromkeys(self.sample_ids)
        ]
        if not np.array_equal(np.concatenate([a.obs_names.to_numpy() for a in found]).astype(str), self.barcodes):
            raise ValueError(f"{self.sample}: the slices' barcodes are not truth_clone_labels.tsv's")
        return sp.vstack([sp.csr_matrix(a.X) for a in found], format="csr")

    def get_snp_counts(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Each barcode's SNP UMIs per planted haplotype: the written A and B counts exchanged back where `phase`.

        Returns:
            `(n_barcodes, n_snps)` A and B counts.
        """
        assert self.sample is not None and self.phase is not None
        written = [sp.load_npz(self.sample / "snp" / f"cell_snp_{x}allele.npz").tocsr() for x in "AB"]
        switched = sp.diags(self.phase.astype(np.int64))
        kept = sp.diags(1 - self.phase.astype(np.int64))
        return (
            sp.csr_matrix(written[0] @ kept + written[1] @ switched),
            sp.csr_matrix(written[1] @ kept + written[0] @ switched),
        )

    def get_snp_ids(self) -> np.ndarray:
        """The SNPs' ids, `<chr>_<pos>_<ref>_<alt>`.

        Returns:
            `(n_snps,)` str ids.
        """
        assert self.sample is not None
        snp_ids = read_numpy(self.sample / "snp" / "unique_snp_ids.npy", allow_pickle=True)  # type: ignore[no-untyped-call]
        return np.asarray(snp_ids).astype(str)

    def copies_at(self, chromosome: np.ndarray, position: np.ndarray) -> np.ndarray:
        """True `(A, B)` at each position; `-1` where no segment covers it.

        Args:
            chromosome: (n,) loci.
            position: (n,) loci.

        Returns:
            `(n, n_clones, 2)` copies.
        """
        chrom = self.profile["chr"].astype(str).str.removeprefix("chr").to_numpy()
        starts, ends = self.profile["start"].to_numpy(), self.profile["end"].to_numpy()
        found = np.full(position.size, -1)
        query = np.asarray(chromosome).astype(str)
        for row in range(len(self.profile)):
            found[(query == chrom[row]) & (position >= starts[row]) & (position < ends[row])] = row
        copies = np.full((position.size, self.n_clones, 2), -1, dtype=np.int64)
        kept = found >= 0
        for label, clone in enumerate(self.clones):
            for allele, column in enumerate(("A", "B")):
                copies[kept, label, allele] = self.profile[f"{clone}_{column}_copy"].to_numpy()[found[kept]]
        return copies


def sample_dir(root: Path, sample_id: str) -> Path:
    """The directory holding `sample_id`'s truth files under a study's `root`.

    Args:
        root: the study directory.
        sample_id: the sample.

    Returns:
        `root / "simulated_data_related" / sample_id`.
    """
    return Path(root) / "simulated_data_related" / sample_id


def _truth_table(sample: Path | Mapping[str, bytes], name: str, **kwargs: Any) -> pd.DataFrame:
    """The truth table `name` (tab-separated) from a sample directory, plain or `.gz`, or from its bytes by name.

    Args:
        sample: a directory or name -> bytes.
        name: the truth file.
        kwargs: to `pandas.read_csv`.

    Returns:
        The frame.
    """
    if isinstance(sample, Path):
        return read_text_table(sample / name, sep="\t", **kwargs)
    return pd.read_csv(io.BytesIO(sample[name]), sep="\t", **kwargs)


def read_truth(sample: Path | Mapping[str, bytes]) -> Truth:
    """`sample`'s truth files: the planted clone labels, spot coordinates and allele-specific copies.

    Args:
        sample: a directory, or the truth files' bytes by name.

    Returns:
        The `Truth`.
    """
    table = _truth_table(sample, "truth_clone_labels.tsv", index_col=0).reset_index()
    table.columns = ["barcode", "clone", "x", "y", *table.columns[4:]]
    others = sorted((n for n in table["clone"].unique() if n != "normal"), key=lambda n: int(n.removeprefix("clone_")))
    clones = ("normal", *others)
    index = {c: i for i, c in enumerate(clones)}
    return Truth(
        barcodes=table["barcode"].to_numpy().astype(str),
        coords=table[["x", "y"]].to_numpy(dtype=np.float64),
        labels=table["clone"].map(index).to_numpy(dtype=np.int64),
        clones=clones,
        profile=_truth_table(sample, "truth_acn_profile.tsv"),
    )


def read_sim_truth(sample: Path) -> Truth:
    """A `cnamaste.sim` draw's truth (`sim.write.write_realization`): `read_truth`'s, with each barcode's slice, the
    tree, the phase and the directory its counts are read from.

    Args:
        sample: the draw's directory.

    Returns:
        The `Truth`.
    """
    truth = read_truth(sample)
    table = _truth_table(sample, "truth_clone_labels.tsv", index_col=0)
    return dataclasses.replace(
        truth,
        sample_ids=table["sample_id"].to_numpy().astype(str),
        tree=_truth_table(sample, "truth_tree.tsv", dtype={"chr": str}),
        phase=np.asarray(read_numpy(sample / "truth_phase.npy"), dtype=bool),  # type: ignore[no-untyped-call]
        sample=sample,
    )


def at_truth(barcodes: np.ndarray, labels: np.ndarray, truth: Truth) -> np.ndarray:
    """`labels` (in the run's spot order) read at the truth's barcodes; -1 where the run dropped a spot.

    Args:
        barcodes: the run's spots.
        labels: per spot.
        truth: the planted truth.

    Returns:
        One label per truth barcode.
    """
    return pd.Series(labels, index=np.asarray(barcodes).astype(str)).reindex(truth.barcodes, fill_value=-1).to_numpy()
