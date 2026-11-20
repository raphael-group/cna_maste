"""A perfect sample drawn from a fixture's planted truth: every spot purely its clone, every gene at one baseline
expression, every SNP heterozygous, and the phase switched at the rate the HMM assumes.

Structure: the source (`Source`, `get_source`), the loci and their copies (`get_snp_loci`, `get_gene_loci`,
`get_clone_copies`), the expectations (`get_expected_umis`, `get_b_allele_fractions`, `get_switch_probs`), the draws
(`draw_umis`, `draw_phase`, `draw_allele_counts`, `draw_sample`), and the store (`write_sample`, on `sim.write`), which
names the sample by its hash (`sims.get_fixture_hash`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple, cast

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cnamaste.config import YAMLConfig
from cnamaste.io import read_numpy
from cnamaste.qa.sims import get_project, get_resources
from cnamaste.recomb import get_sitewise_transmat
from cnamaste.reference import get_reference_genes
from cnamaste.sim.write import get_gzipped_bytes, get_made_dir, get_named_dir, get_npy, write_sim_npz
from cnamaste.truth import Truth, read_truth

COPIED = (
    "barcodes.txt.gz",
    "unique_snp_ids.npy.gz",
    "spatial/tissue_positions_list.csv.gz",
    "truth_clone_labels.tsv.gz",
    "truth_acn_profile.tsv.gz",
)
"""The source's files a perfect sample holds byte for byte."""


def get_copied() -> tuple[str, ...]:
    """Return the copied, `COPIED`."""
    return COPIED


class Perfect(NamedTuple):
    """A perfect sample's pins.

    Args:
        source: the fixture hash it is drawn from.
        sample: its own hash at `run_sim_gen`'s defaults (seed 0, depth factor 100).
    """

    source: str
    sample: str


PERFECT = {"perfect_easy": Perfect("2d4ce9a9", "f11658d6"), "perfect_hard": Perfect("8797710b", "bd4c8102")}
"""A perfect sample's short name -> its pins: CalicoST easy and hard."""


def get_perfect() -> dict[str, Perfect]:
    """Return the perfect, `PERFECT`."""
    return PERFECT


CONFIG = get_project() / "zenodo_sim_config.yaml"
"""The shipped configuration, whose `phasing` parameters set the planted switch rate."""


def get_config() -> Path:
    """Return the config, `CONFIG`."""
    return CONFIG


class Source(NamedTuple):
    """The fixture a perfect sample is drawn from.

    Args:
        directory: the fixture's sample.
        truth: its planted truth.
        adata: its counts, whose `obs` and `var` the sample keeps.
        snp_ids: its SNPs.
        clone_of_spot: each spot's index into `truth.clones`, in `adata`'s order.
    """

    directory: Path
    truth: Truth
    adata: anndata.AnnData
    snp_ids: np.ndarray
    clone_of_spot: np.ndarray


class Sample(NamedTuple):
    """A drawn perfect sample.

    Args:
        umis: (spots, genes) UMIs.
        a_counts: (spots, SNPs) allele counts.
        b_counts: (spots, SNPs) allele counts.
        phase: (SNPs,) 0 where a SNP's B count is its clone's B haplotype, 1 where its A.
        switch_probs: (SNPs,) the probability of a switch between each SNP and the next.
        mu0: the source's mean UMIs per gene and spot.
        snp_depth: its mean reads per SNP and spot, each before the depth factor.
    """

    umis: sp.csr_matrix
    a_counts: sp.csr_matrix
    b_counts: sp.csr_matrix
    phase: np.ndarray
    switch_probs: np.ndarray
    mu0: float
    snp_depth: float


def get_source(directory: Path) -> Source:
    """A fixture's sample, read as a perfect sample needs it.

    Args:
        directory: the fixture's sample, e.g. `sims.fixture("2d4ce9a9")`.

    Returns:
        The `Source`.
    """
    truth = read_truth(directory)
    adata = anndata.read_h5ad(directory / "filtered_feature_bc_matrix.h5ad")
    index = dict(zip(truth.barcodes, truth.labels))
    clone_of_spot = np.array([index[b] for b in adata.obs_names], dtype=np.int64)
    snp_ids = read_numpy(directory / "unique_snp_ids.npy", allow_pickle=True)  # type: ignore[no-untyped-call]
    return Source(directory, truth, adata, snp_ids, clone_of_spot)


def get_snp_loci(snp_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each SNP's contig and position, from its id `<chr>_<pos>_<ref>_<alt>`.

    Args:
        snp_ids: (SNPs,) ids.

    Returns:
        (SNPs,) int contigs and (SNPs,) int positions.
    """
    parts = [str(s).split("_") for s in snp_ids]
    return np.array([int(p[0]) for p in parts]), np.array([int(p[1]) for p in parts])


def get_gene_loci(genes: pd.Index, hgtable_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Each gene's autosome and start, `reference.get_reference_genes`' first row of its name; -1 where it has none.

    Args:
        genes: (genes,) names.
        hgtable_file: the GRCh38 gene table.

    Returns:
        (genes,) int contigs and (genes,) int starts.
    """
    genes_table = get_reference_genes(hgtable_file)  # type: ignore[no-untyped-call]
    table = genes_table.drop_duplicates("gene").set_index("gene")
    found = table.reindex(genes.astype(str))
    return found["CHR"].fillna(-1).to_numpy(np.int64), found["START"].fillna(-1).to_numpy(np.int64)


def get_clone_copies(truth: Truth, chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """True (A, B) per locus and clone, `Truth.copies_at`; (1, 1) where no segment covers a locus.

    Args:
        truth: the planted truth.
        chrom: (n,) loci.
        pos: (n,) loci.

    Returns:
        (n_clones, n, 2) copies.
    """
    copies = truth.copies_at(chrom, pos)
    # NB an unplaced gene (no autosome in the gene table) is held at the diploid baseline.
    copies[copies < 0] = 1
    return copies.transpose(1, 0, 2)


def get_expected_umis(mu0: float, depth_factor: float, copies: np.ndarray) -> np.ndarray:
    """Expected UMIs per clone and gene, `mu0 * depth_factor * (A + B) / 2`.

    Args:
        mu0: the source's mean UMIs per gene and spot.
        depth_factor: its multiple.
        copies: (n_clones, genes, 2).

    Returns:
        (n_clones, genes) expectations.
    """
    return mu0 * depth_factor * copies.sum(axis=-1) / 2.0


def get_b_allele_fractions(copies: np.ndarray) -> np.ndarray:
    """The B-allele fraction per clone and SNP, `B / (A + B)`; 1/2 where A + B is 0, which draws no reads.

    Args:
        copies: (n_clones, SNPs, 2).

    Returns:
        (n_clones, SNPs) fractions.
    """
    total = copies.sum(axis=-1)
    return np.where(total > 0, copies[..., 1] / np.maximum(total, 1), 0.5)


def get_switch_probs(chrom: np.ndarray, pos: np.ndarray, geneticmap_file: Path, config: YAMLConfig) -> np.ndarray:
    """The HMM's phase switch probability between each SNP and the next: `recomb.get_sitewise_transmat` with each
    SNP its own block, at `config.phasing`'s `nu` and `logphase_shift` (`min_prob` read from the global config).

    Args:
        chrom: (SNPs,) loci in logical order.
        pos: (SNPs,) loci in logical order.
        geneticmap_file: the GRCh38 genetic map.
        config: the run's.

    Returns:
        (SNPs,) probabilities, the last the floor (no next SNP).
    """
    snps = pd.DataFrame({"CHR": chrom, "START": pos, "END": pos, "snp": np.arange(len(pos))})
    phasing = config.phasing  # type: ignore[attr-defined]
    log_transmat = get_sitewise_transmat("snp", snps, geneticmap_file, phasing.nu, phasing.logphase_shift)  # type: ignore[no-untyped-call]
    return np.asarray(np.exp(log_transmat))


def draw_umis(expected: np.ndarray, clone_of_spot: np.ndarray, rng: np.random.Generator) -> sp.csr_matrix:
    """Poisson UMIs per spot and gene at its clone's expectation, spot by spot in order.

    Args:
        expected: (n_clones, genes).
        clone_of_spot: (spots,) clone index.
        rng: the draw's generator.

    Returns:
        (spots, genes) int64 UMIs.
    """
    rows = [sp.csr_matrix(rng.poisson(expected[clone_of_spot[s : s + 256]])) for s in range(0, len(clone_of_spot), 256)]
    return sp.vstack(rows, format="csr").astype(np.int64)


def draw_phase(switch_probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The planted phase per SNP: the first 0 or 1 at random, then flipped after SNP i with `switch_probs[i]`.

    Args:
        switch_probs: (SNPs,) per `get_switch_probs`.
        rng: the draw's generator.

    Returns:
        (SNPs,) int8 phase.
    """
    first = rng.integers(2)
    flips = rng.random(len(switch_probs) - 1) < switch_probs[:-1]
    return ((first + np.concatenate([[0], np.cumsum(flips)])) % 2).astype(np.int8)


def draw_allele_counts(
    snp_depth: float, fractions: np.ndarray, phase: np.ndarray, clone_of_spot: np.ndarray, rng: np.random.Generator
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Allele counts per spot and SNP: Poisson depth at `snp_depth`, then the B count binomial at the clone's B-allele
    fraction, or at its A-allele fraction where the phase is 1.

    Args:
        snp_depth: the expected reads per spot and SNP.
        fractions: (n_clones, SNPs) B-allele fractions.
        phase: (SNPs,) per `draw_phase`.
        clone_of_spot: (spots,) clone index.
        rng: the draw's generator.

    Returns:
        (spots, SNPs) int64 A and B counts.
    """
    phased = np.where(phase == 1, 1.0 - fractions, fractions)
    depth = rng.poisson(snp_depth, size=(len(clone_of_spot), len(phase)))
    b = rng.binomial(depth, phased[clone_of_spot])
    return sp.csr_matrix(depth - b, dtype=np.int64), sp.csr_matrix(b, dtype=np.int64)


def get_mean_depths(source: Source) -> tuple[float, float]:
    """The source's mean UMIs per gene and spot, and mean SNP reads (A + B) per SNP and spot.

    Args:
        source: the fixture.

    Returns:
        `(mu0, snp_depth)`.
    """
    a = read_snp_counts(source.directory, "A")
    b = read_snp_counts(source.directory, "B")
    return float(cast(sp.csr_matrix, source.adata.X).mean()), float((a.sum() + b.sum()) / np.prod(a.shape))


def read_snp_counts(directory: Path, allele: str) -> sp.csr_matrix:
    """A sample's `cell_snp_<allele>allele.npz`.

    Args:
        directory: the sample.
        allele: A or B.

    Returns:
        (spots, SNPs) counts.
    """
    return sp.load_npz(directory / f"cell_snp_{allele}allele.npz").tocsr()


def draw_sample(
    source: Source,
    depth_factor: float,
    seed: int,
    grch38: Path,
    config: YAMLConfig,
) -> Sample:
    """A perfect sample of `source`: UMIs, phase and allele counts, each from its own stream of `seed`.

    Args:
        source: the fixture.
        depth_factor: the multiple of its depths.
        seed: the draw's.
        grch38: the GRCh38 resources.
        config: the run's, for the phasing parameters.

    Returns:
        The `Sample`.
    """
    mu0, snp_depth = get_mean_depths(source)
    umi_rng, phase_rng, allele_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))

    gene_chrom, gene_pos = get_gene_loci(source.adata.var_names, grch38 / get_resources()["hgtable_file"])
    gene_copies = get_clone_copies(source.truth, gene_chrom, gene_pos)
    umis = draw_umis(get_expected_umis(mu0, depth_factor, gene_copies), source.clone_of_spot, umi_rng)

    snp_chrom, snp_pos = get_snp_loci(source.snp_ids)
    switch_probs = get_switch_probs(snp_chrom, snp_pos, grch38 / get_resources()["geneticmap_file"], config)
    phase = draw_phase(switch_probs, phase_rng)

    fractions = get_b_allele_fractions(get_clone_copies(source.truth, snp_chrom, snp_pos))
    a, b = draw_allele_counts(depth_factor * snp_depth, fractions, phase, source.clone_of_spot, allele_rng)
    return Sample(umis, a, b, phase, switch_probs, mu0, snp_depth)


def get_commit() -> str:
    """The checkout's commit, `git rev-parse HEAD` in `PROJECT`.

    Returns:
        The sha, or `unknown` outside a git checkout.
    """
    found = subprocess.run(["git", "rev-parse", "HEAD"], cwd=get_project(), capture_output=True, text=True, check=False)
    return found.stdout.strip() or "unknown"


def write_sample(source: Source, sample: Sample, out: Path, name: str, manifest: dict[str, object]) -> Path:
    """`sample` in the source's layout under `out/<name>_<hash>/`, with the copied truth, the planted phase and its
    switches, a `sample_sheet.tsv` that `run_cnamaste` reads (absolute paths) and a `manifest.json`, staged then renamed (`sim.write.get_named_dir`).

    Args:
        source: the fixture.
        sample: the draw.
        out: the parent directory, existing.
        name: the sample's short name.
        manifest: the draw's parameters, to which the hash is added.

    Returns:
        The sample's directory.

    Raises:
        RuntimeError: where `out` is missing, `out/<name>_staging/` or the sample's directory exists.
    """
    staging = get_made_dir(out, f"{name}_staging", exist_ok=False)
    get_made_dir(staging, "spatial", exist_ok=False)
    for copied in COPIED:
        shutil.copyfile(source.directory / copied, staging / copied)

    adata = anndata.AnnData(X=sample.umis, obs=source.adata.obs, var=source.adata.var)
    adata.write_h5ad(staging / "filtered_feature_bc_matrix.h5ad", compression="gzip")
    write_sim_npz(staging / "cell_snp_Aallele.npz", sample.a_counts, None, False)
    write_sim_npz(staging / "cell_snp_Ballele.npz", sample.b_counts, None, False)

    (staging / "truth_phase.npy.gz").write_bytes(get_gzipped_bytes(get_npy(sample.phase)))
    switched = np.flatnonzero(np.diff(sample.phase)) + 1
    switches = pd.DataFrame({"snp_index": switched, "snp_id": source.snp_ids[switched]})
    (staging / "truth_phase_switches.tsv.gz").write_bytes(
        get_gzipped_bytes(switches.to_csv(sep="\t", index=False).encode())
    )

    directory, fixture_hash = get_named_dir(staging, name)

    # NB absolute, so a run reads the sample from any working directory.
    located = str(directory.resolve())
    sheet = pd.DataFrame({"bam": ["NULL"], "sample_id": [name], "spaceranger_dir": [located], "snp_dir": [located]})
    sheet.to_csv(directory / "sample_sheet.tsv", sep="\t", index=False)
    manifest = {**manifest, "switches": len(switched), "hash": fixture_hash}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory
