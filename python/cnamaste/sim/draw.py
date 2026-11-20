"""Realizations of a simulation: one planted truth (clone count, layout, tree), then per realization its counts and
phase, and under `[sample] vary = "truth"` its own truth after the first.

`SeedSequence(seed)` spawns the tree's, the layout's and the barcodes' generators; realization `k` draws its counts and
phase from the first two children of `SeedSequence([seed, 1], spawn_key=(k,))`, and its truth from the next two, so
it is independent of how many are asked for. `realize` and `stream` hold nothing between realizations.
"""

from __future__ import annotations

from collections.abc import Container, Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cnamaste.io import read_numpy
from cnamaste.qa.sims import get_resources, grch38
from cnamaste.sim.laws import (
    draw_ab_counts,
    draw_snp_reads,
    draw_spot_totals,
    draw_umis,
    get_allele_share,
    get_depth_factors,
    get_snp_mixture,
    read_coverage,
)
from cnamaste.sim.layout import draw_barcodes, draw_clone_labels, draw_sample_ids, get_array
from cnamaste.sim.manifest import Simulation, get_simulation, get_tumour_clones
from cnamaste.sim.perfect import get_snp_loci
from cnamaste.sim.phase import draw_switched, get_switch_probabilities
from cnamaste.sim.tree import CloneTree, draw_tree, draw_tree_clone_count, get_clone_copies, get_truth_profile
from cnamaste.sim.write import write_realization


class Planted(NamedTuple):
    """A simulation's planted truth, shared by its realizations.

    Args:
        clones: normal, then the tumour clones.
        tree: the clone tree.
        profile: `truth_acn_profile.tsv`.
        labels: per slice, each spot's index into `clones`.
        sample_ids: per slice, its id.
        barcodes: per slice, `{barcode}_{sample_id}`.
        array_row: each spot's array row, every slice alike.
        array_col: each spot's array column, every slice alike.
        genes: the genes, `normal_baseline`'s order.
        snp_ids: the SNPs.
        switch_probabilities: per SNP, a switch from the SNP before it.
        gene_weights: (genes, clones) each gene's expected share, unnormalized, `lambda d`.
        library: (clones,) each clone's library factor, `sum lambda d`.
        share: (clones, SNPs) the haplotype-A share.
    """

    clones: tuple[str, ...]
    tree: CloneTree
    profile: pd.DataFrame
    labels: list[np.ndarray]
    sample_ids: list[str]
    barcodes: list[np.ndarray]
    array_row: np.ndarray
    array_col: np.ndarray
    genes: np.ndarray
    snp_ids: np.ndarray
    switch_probabilities: np.ndarray
    gene_weights: np.ndarray
    library: np.ndarray
    share: np.ndarray


class Realization(NamedTuple):
    """One realization's draws.

    Args:
        k: its index.
        planted: the truth it shares.
        umis: per slice, (spots, genes) UMIs.
        a_counts: (spots of every slice, SNPs) the written A counts.
        b_counts: (spots of every slice, SNPs) the written B counts.
        switched: (SNPs,) True where A and B are exchanged.
    """

    k: int
    planted: Planted
    umis: list[sp.csr_matrix]
    a_counts: sp.csr_matrix
    b_counts: sp.csr_matrix
    switched: np.ndarray


def get_snp_ids(simulation: Simulation) -> np.ndarray:
    """The simulation's SNP ids, `<chr>_<pos>_<ref>_<alt>`.

    Args:
        simulation: the simulation.

    Returns:
        (SNPs,) str ids.
    """
    snp_ids = read_numpy(simulation.snps, allow_pickle=True)  # type: ignore[no-untyped-call]
    return np.asarray(snp_ids).astype(str)


def get_recomb_switch_probabilities(simulation: Simulation, geneticmap_file: Path) -> np.ndarray:
    """The simulation's switch probabilities, `phase.get_switch_probabilities` on its SNPs; 0 without switch errors.

    Args:
        simulation: the simulation.
        geneticmap_file: the GRCh38 genetic map.

    Returns:
        (SNPs,) probabilities.
    """
    chromosome, position = get_snp_loci(get_snp_ids(simulation))
    if not simulation.switch_errors:
        return np.zeros(position.size)
    return get_switch_probabilities(chromosome, position, geneticmap_file, simulation.nu, simulation.unit)


def realize(
    simulation: Simulation, switch_probabilities: np.ndarray, wanted: Container[int] | None
) -> Iterator[Realization]:
    """Each realization in turn, those in `wanted` where given: the truth drawn once (again per realization after
    the first under `vary = "truth"`), then each realization's counts and phase.

    Args:
        simulation: the simulation.
        switch_probabilities: (SNPs,) per SNP, a phase switch from the SNP before it.
        wanted: the realizations to draw; None for every one.

    Yields:
        The realizations, k = 0, 1, ...
    """
    tree_rng, layout_rng, id_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(simulation.seed).spawn(3))
    model = simulation.model
    coverage = read_coverage(simulation.coverage)
    baseline = pd.read_csv(simulation.baseline, sep="\t", comment="#")
    lam = baseline["lambda"].to_numpy()
    gene_chrom = baseline["chrom"].astype(str).str.removeprefix("chr").to_numpy()
    gene_pos = ((baseline["cdsStart"] + baseline["cdsEnd"]) // 2).to_numpy()
    snp_ids = get_snp_ids(simulation)
    snp_chrom, snp_pos = get_snp_loci(snp_ids)
    snp_law = get_snp_mixture(coverage, model.snp_dispersion, snp_ids.size)

    array_row, array_col, points = get_array(*simulation.array)
    length, suffix, n_bytes = simulation.barcodes
    whitelist = draw_barcodes(array_row.size, length, suffix, id_rng)
    sample_ids = draw_sample_ids(len(simulation.slices), n_bytes, id_rng)

    def draw_planted(tree_rng: np.random.Generator, layout_rng: np.random.Generator) -> Planted:
        resolved = simulation._replace(
            cna=simulation.cna._replace(n_clones=draw_tree_clone_count(simulation, tree_rng))
        )
        spots, placed = draw_clone_labels(resolved, points, layout_rng)
        # NB an unplaced clone leaves the truth; the others are renumbered `clone_0, ...` in order.
        renumber = np.array([*(placed.index(c) if c in placed else -1 for c in get_tumour_clones(resolved)), -1])
        resolved = resolved._replace(cna=resolved.cna._replace(n_clones=len(placed)))
        clones = ("normal", *get_tumour_clones(resolved))
        tree = draw_tree(resolved, tree_rng)
        normal_frac = np.array([0.0, *[model.normal_frac] * (len(clones) - 1)])
        depth_factors = get_depth_factors(get_clone_copies(tree, clones, gene_chrom, gene_pos), normal_frac)
        return Planted(
            clones=clones,
            tree=tree,
            profile=get_truth_profile(tree, clones, simulation.chromosome_lengths),
            labels=[renumber[s] + 1 for s in spots],
            sample_ids=sample_ids,
            barcodes=[np.array([f"{b}_{i}" for b in whitelist]) for i in sample_ids],
            array_row=array_row,
            array_col=array_col,
            genes=baseline["gene"].to_numpy(),
            snp_ids=snp_ids,
            switch_probabilities=switch_probabilities,
            gene_weights=lam[:, None] * depth_factors,
            library=lam @ depth_factors,
            share=get_allele_share(get_clone_copies(tree, clones, snp_chrom, snp_pos), normal_frac, model.admixture),
        )

    shared = draw_planted(tree_rng, layout_rng)
    for k in range(simulation.realizations):
        if wanted is not None and k not in wanted:
            continue
        # NB spawn(4)'s first two children are spawn(2)'s: the counts and phase are the same under either `vary`.
        streams = np.random.SeedSequence([simulation.seed, 1], spawn_key=(k,)).spawn(
            4 if simulation.vary == "truth" else 2
        )
        planted = (
            draw_planted(*(np.random.default_rng(s) for s in streams[2:]))
            if simulation.vary == "truth" and k > 0
            else shared
        )
        rng = np.random.default_rng(streams[0])
        switched = draw_switched(switch_probabilities, snp_chrom, np.random.default_rng(streams[1]))
        umis, a_counts, b_counts = [], [], []
        for labels in planted.labels:
            totals = np.rint(draw_spot_totals(coverage.spot_umi, labels.size, rng) * planted.library[labels])
            umis.append(
                draw_umis(
                    model.counts_sampler, totals, planted.gene_weights, labels, model.dirichlet_concentration, rng
                )
            )
            reads = draw_snp_reads(snp_law, labels.size, snp_ids.size, rng)
            a, b = draw_ab_counts(reads, planted.share, model.bb_overdispersion, switched, labels, rng)
            a_counts.append(a)
            b_counts.append(b)
        yield Realization(
            k=k,
            planted=planted,
            umis=umis,
            a_counts=sp.vstack(a_counts, format="csr").astype(np.int64),
            b_counts=sp.vstack(b_counts, format="csr").astype(np.int64),
            switched=switched,
        )


def stream(
    simulation: str | Simulation, wanted: Container[int] | None, out: Path | None
) -> Iterator[tuple[Realization, Path | None]]:
    """`realize` at recomb's switch probabilities on CalicoST's GRCh38 genetic map, each realization written under
    `out` where given; nothing is held between realizations.

    Args:
        simulation: the simulation, or its name.
        wanted: the realizations to draw; None for every one.
        out: the parent directory of each written `<name>_<hash>/`; None writes nothing.

    Yields:
        Each realization and its written directory, None where `out` is None.

    Raises:
        ValueError: where no GRCh38 resources are found.
    """
    simulation = get_simulation(simulation) if isinstance(simulation, str) else simulation
    references = grch38()
    if references is None:
        raise ValueError("no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
    switches = get_recomb_switch_probabilities(simulation, references / get_resources()["geneticmap_file"])
    for realization in realize(simulation, switches, wanted):
        yield realization, None if out is None else write_realization(realization, simulation, out)
