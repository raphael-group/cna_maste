"""`cnamaste.sim`: dev_tree easy r0's and study15 r0's pins, and the laws the draw states.

The pins hold every stage at once; the others hold what a pin cannot say is right: an explicit event plants its
expression ratio and BAF, a realization does not depend on the count drawn, a lost haplotype stays lost down the
tree, a lognormal length law stated by its mean has that mean, and a SNP entry is its NB's quantile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

from cnamaste.qa.sims import get_fixture_hash, get_resources, grch38
from cnamaste.sim.draw import Realization, get_recomb_switch_probabilities, realize, stream
from cnamaste.sim.laws import draw_event_length, draw_zero_truncated, get_zero_probability
from cnamaste.sim.manifest import Event, Length, get_simulation
from cnamaste.sim.perfect import get_snp_loci
from cnamaste.sim.tree import draw_tree, get_clone_copies, get_explicit_tree, get_truth_profile
from cnamaste.sim.write import write_realization

REFERENCES = grch38()


@pytest.mark.slow
@pytest.mark.skipif(REFERENCES is None, reason="no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
def test_dev_tree_easy_r0_is_its_pin(tmp_path: Path) -> None:
    """recomb's switches draw the fixture's phase: its 8 of 13,482 probabilities 2.8e-14 off move no switch."""
    assert REFERENCES is not None
    simulation = get_simulation("dev_tree_easy")
    switches = get_recomb_switch_probabilities(simulation, REFERENCES / get_resources()["geneticmap_file"])
    realization = next(realize(simulation, switches, None))
    assert get_fixture_hash(write_realization(realization, simulation, tmp_path)) == "e4bbe03f"


@pytest.mark.slow
@pytest.mark.skipif(REFERENCES is None, reason="no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
def test_study15_r0_is_its_first_draw(tmp_path: Path) -> None:
    """A uniform clone count, Normal radii, `unplaced = "drop"`, a tree drawn per edge: r0 is its first draw's hash."""
    realization, directory = next(stream("study15", None, tmp_path))
    assert directory is not None and get_fixture_hash(directory) == "784e4ebf"
    assert realization.planted.clones[1:] == ("clone_0", "clone_1", "clone_2", "clone_3")


@pytest.fixture(scope="module")
def explicit_easy() -> list[Realization]:
    """explicit_easy's three realizations, each with its own rectangular layout."""
    return [realization for realization, _ in stream("explicit_easy", None, None)]


@pytest.mark.slow
@pytest.mark.skipif(REFERENCES is None, reason="no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
def test_explicit_events_plant_their_expression_ratio_and_baf(explicit_easy: list[Realization]) -> None:
    """Pseudobulk by true clone, each altered segment's UMI share over the clone's (1, 1) genes, against normal's, is
    `(A + B) / 2` to 10% (3 SE of the Dirichlet spread at kappa 100 on 625 spots); its B share of SNP reads is
    `B / (A + B)` to 0.03 (4 binomial SE at 3,300 reads), with no switch errors and no admixture."""
    planted = explicit_easy[0].planted
    labels, umis = planted.labels[0], explicit_easy[0].umis[0]
    a_counts, b_counts = explicit_easy[0].a_counts, explicit_easy[0].b_counts
    genes = pd.read_csv(get_simulation("explicit_easy").baseline, sep="\t", comment="#")
    gene_chrom = genes["chrom"].str.removeprefix("chr").to_numpy()
    gene_pos = ((genes["cdsStart"] + genes["cdsEnd"]) // 2).to_numpy()
    snp_chrom, snp_pos = get_snp_loci(planted.snp_ids)
    profile = planted.profile
    checked = 0
    for clone in (1, 2):
        name = planted.clones[clone]
        states = profile[[f"{name}_A_copy", f"{name}_B_copy"]].to_numpy()
        on = [(gene_chrom == str(r.chr)) & (gene_pos >= r.start) & (gene_pos < r.end) for r in profile.itertuples()]
        flat = np.any([g for g, (a, b) in zip(on, states, strict=True) if (a, b) == (1, 1)], axis=0)
        for row, genes_on, (a, b) in zip(profile.itertuples(), on, states, strict=True):
            if (a, b) == (1, 1):
                continue
            shares = [umis[labels == k][:, genes_on].sum() / umis[labels == k][:, flat].sum() for k in (clone, 0)]
            snps = (snp_chrom.astype(str) == str(row.chr)) & (snp_pos >= row.start) & (snp_pos < row.end)
            b_reads = b_counts[labels == clone][:, snps].sum()
            reads = b_reads + a_counts[labels == clone][:, snps].sum()
            assert abs(shares[0] / shares[1] / ((a + b) / 2) - 1) < 0.1
            assert abs(b_reads / reads - b / (a + b)) < 0.03
            checked += 1
    assert checked == 9


@pytest.mark.slow
@pytest.mark.skipif(REFERENCES is None, reason="no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
def test_a_realization_does_not_depend_on_the_count_drawn(explicit_easy: list[Realization]) -> None:
    """Realization 2 drawn alone is realization 2 of three: its own layout (`vary = "truth"`), counts and phase."""
    alone, _ = next(stream("explicit_easy", {2}, None))
    full = explicit_easy[2]
    assert alone.k == 2
    assert not np.array_equal(full.planted.labels[0], explicit_easy[0].planted.labels[0])
    assert np.array_equal(alone.planted.labels[0], full.planted.labels[0])
    for drawn, again in (
        (alone.umis[0], full.umis[0]),
        (alone.a_counts, full.a_counts),
        (alone.b_counts, full.b_counts),
    ):
        assert (drawn != again).nnz == 0


def test_an_explicit_tree_refuses_a_regained_haplotype() -> None:
    """Under irreversible LOH, clone_0 regaining the B its founder lost on chr3 is refused; reversible, it is drawn."""
    simulation = get_simulation("explicit_easy")
    events = (("founder", Event("3", 0, 10**7, 1, 0)), ("clone_0", Event("3", 0, 5 * 10**6, 1, 1)))
    lost = simulation._replace(cna=simulation.cna._replace(events=events))
    with pytest.raises(ValueError, match="inadmissible"):
        get_explicit_tree(lost)
    regained = get_explicit_tree(lost._replace(cna=lost.cna._replace(loh="reversible")))
    assert regained.edge_events["clone_0"] == (events[1][1],)


def test_irreversible_loh_never_regains_a_lost_haplotype() -> None:
    """Over 200 dev_tree easy trees, a haplotype at 0 in a node is at 0 in each child, on every segment."""
    simulation = get_simulation("dev_tree_easy")
    lost = 0
    for seed in range(200):
        tree = draw_tree(simulation, np.random.default_rng(seed))
        nodes = tuple(tree.parent)
        profile = get_truth_profile(tree, nodes, simulation.chromosome_lengths)
        middle = ((profile["start"] + profile["end"]) // 2).to_numpy()
        copies = get_clone_copies(tree, nodes, profile["chr"].astype(str).to_numpy(), middle)
        for child, parent in tree.parent.items():
            if parent is not None:
                above, below = copies[:, nodes.index(parent)], copies[:, nodes.index(child)]
                assert np.all(below[above == 0] == 0)
                lost += int(np.sum(above == 0))
    assert lost > 0


def test_a_lognormal_length_stated_by_its_mean_has_that_mean() -> None:
    """`median = mean exp(-sigma^2 / 2)`: 40,000 lengths at dev_tree easy's sigma average 5e7 to 4 standard errors."""
    length = Length("lognormal", None, 5e7, None, 0.541, 1.0)
    rng = np.random.default_rng(0)
    drawn = np.array([draw_event_length(length, rng) for _ in range(40_000)], dtype=np.float64)
    assert abs(drawn.mean() - 5e7) < 4 * drawn.std() / np.sqrt(drawn.size)


@pytest.mark.parametrize("dispersion", [0.0, 0.25, 1.0])
def test_a_zero_truncated_draw_is_the_nb_quantile(dispersion: float) -> None:
    """Each draw given `k >= 1` is scipy's NB (Poisson at 0) quantile of its level above `P(0)`, at means 0.05-40."""
    centres = np.array([0.05, 1.0, 40.0])
    zeros = get_zero_probability(centres, dispersion)
    rng = np.random.default_rng(0)
    pick, level = rng.random(3_000), rng.random(3_000)
    drawn = draw_zero_truncated(np.array([1 / 3, 2 / 3, 1.0]), centres, zeros, dispersion, pick, level)
    component = np.searchsorted([1 / 3, 2 / 3, 1.0], pick, side="right")
    mu = centres[component]
    u = zeros[component] + level * (1.0 - zeros[component])
    if dispersion > 0:
        expected = nbinom.ppf(u, 1.0 / dispersion, 1.0 / (1.0 + dispersion * mu))
    else:
        expected = poisson.ppf(u, mu)
    assert np.array_equal(drawn, np.maximum(expected, 1))
