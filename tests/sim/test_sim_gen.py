"""`cnamaste.sim.perfect`'s perfect samples of CalicoST easy (`2d4ce9a9`) and hard (`8797710b`), judged against the
expectations they are drawn at.

Each tolerance is 4 sigma of the draw's own distribution (Poisson, binomial), so a row fails where the generator
departs from its stated model, not on its noise. The draws are on subsets of each source's spots and genes, seconds each.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import anndata
import numpy as np
import pytest

from cnamaste.config import YAMLConfig, set_global_config
from cnamaste.io import read_numpy
from cnamaste.qa.sims import fixture, get_fixture_hash, get_resources, grch38
from cnamaste.sim import perfect
from cnamaste.sim.write import get_made_dir

REFERENCES = grch38()
pytestmark = pytest.mark.skipif(REFERENCES is None, reason="no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")


@pytest.fixture(scope="module")
def config() -> YAMLConfig:
    found = YAMLConfig.from_file(perfect.get_config())
    set_global_config(found)
    return found


@pytest.fixture(scope="module", params=sorted(perfect.get_perfect()))
def name(request: pytest.FixtureRequest) -> str:
    """Each perfect sample's short name."""
    return str(request.param)


@pytest.fixture(scope="module")
def source(name: str) -> perfect.Source:
    return perfect.get_source(fixture(perfect.get_perfect()[name].source))


def get_regions(source: perfect.Source, chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Each locus's row of `truth_acn_profile`, -1 where none covers it."""
    profile = source.truth.profile
    found = np.full(len(pos), -1)
    for row, (c, start, end) in enumerate(profile[["chr", "start", "end"]].itertuples(index=False)):
        found[(chrom == int(c)) & (pos >= start) & (pos < end)] = row
    return found


def get_subset(source: perfect.Source, n_spots: int) -> perfect.Source:
    """`source` cut to its first `n_spots` spots of each clone and every 20th gene."""
    spots = np.concatenate([np.flatnonzero(source.clone_of_spot == c)[:n_spots] for c in range(source.truth.n_clones)])
    genes = np.arange(0, source.adata.n_vars, 20)
    adata = anndata.AnnData(
        X=source.adata.X[spots][:, genes], obs=source.adata.obs.iloc[spots], var=source.adata.var.iloc[genes]
    )
    return source._replace(adata=adata, clone_of_spot=source.clone_of_spot[spots])


def test_the_fixture_hash_is_the_source_s(name: str, tmp_path: Path) -> None:
    """`get_fixture_hash` of the source's eight sample files is its name's hash; fails where the convention drifts."""
    pinned = perfect.get_perfect()[name].source
    for copied in (
        *perfect.get_copied(),
        "filtered_feature_bc_matrix.h5ad",
        "cell_snp_Aallele.npz",
        "cell_snp_Ballele.npz",
    ):
        (tmp_path / copied).parent.mkdir(exist_ok=True)
        shutil.copyfile(fixture(pinned) / copied, tmp_path / copied)
    assert get_fixture_hash(tmp_path) == pinned


def test_each_perfect_hash_is_its_own() -> None:
    """The pinned perfect hashes, which `test_sim_perfect` asserts of each full draw, differ from every source's and
    each other's: a draw that copied its source's counts, or two that coincide, fails."""
    pins = [h for pinned in perfect.get_perfect().values() for h in pinned]
    assert len(set(pins)) == len(pins)


def test_a_seed_names_its_sample_and_copies_its_source_s_truth(
    name: str, source: perfect.Source, config: YAMLConfig, tmp_path: Path
) -> None:
    """The same seed writes the same hash, another seed another; the truth files decode to the source's bytes."""
    subset = get_subset(source, 20)
    written = [
        perfect.write_sample(
            subset,
            perfect.draw_sample(subset, 1.0, seed, REFERENCES, config),
            get_made_dir(tmp_path, str(i), exist_ok=False),
            name,
            {},
        )
        for i, seed in enumerate((0, 0, 1))
    ]
    hashes = [w.name.removeprefix(f"{name}_") for w in written]
    assert hashes[0] == hashes[1] != hashes[2]
    for truth in ("truth_clone_labels.tsv.gz", "truth_acn_profile.tsv.gz"):
        assert gzip.decompress((written[0] / truth).read_bytes()) == gzip.decompress(
            (source.directory / truth).read_bytes()
        )


def test_the_umis_are_drawn_at_their_clone_s_copies(source: perfect.Source) -> None:
    """Each clone and region's mean UMIs per spot and gene is `mu0 * depth_factor * (A + B) / 2` within 4 sigma."""
    subset = get_subset(source, 100)
    mu0, _ = perfect.get_mean_depths(source)
    chrom, pos = perfect.get_gene_loci(subset.adata.var_names, REFERENCES / get_resources()["hgtable_file"])
    copies = perfect.get_clone_copies(subset.truth, chrom, pos)
    umis = perfect.draw_umis(
        perfect.get_expected_umis(mu0, 100.0, copies), subset.clone_of_spot, np.random.default_rng(0)
    )
    regions = get_regions(subset, chrom, pos)
    for clone in range(subset.truth.n_clones):
        spots = umis[subset.clone_of_spot == clone].toarray()
        for region in np.unique(regions[regions >= 0]):
            genes = regions == region
            expected = mu0 * 100.0 * copies[clone, genes].sum(axis=-1)[0] / 2.0
            drawn = spots[:, genes]
            assert abs(drawn.mean() - expected) < 4.0 * np.sqrt(max(expected, 1e-12) / drawn.size)


def test_the_b_allele_fraction_is_planted_up_to_the_phase(source: perfect.Source, config: YAMLConfig) -> None:
    """Each clone and region's phase-corrected B fraction is `B / (A + B)` within 4 sigma, exactly where LOH."""
    clone_of_spot = get_subset(source, 100).clone_of_spot
    chrom, pos = perfect.get_snp_loci(source.snp_ids)
    fractions = perfect.get_b_allele_fractions(perfect.get_clone_copies(source.truth, chrom, pos))
    switch_probs = perfect.get_switch_probs(chrom, pos, REFERENCES / get_resources()["geneticmap_file"], config)
    phase = perfect.draw_phase(switch_probs, np.random.default_rng(1))
    a, b = perfect.draw_allele_counts(3.0, fractions, phase, clone_of_spot, np.random.default_rng(2))
    planted_b = np.where(phase == 1, a.toarray(), b.toarray())
    depth = (a + b).toarray()
    regions = get_regions(source, chrom, pos)
    for clone in range(source.truth.n_clones):
        spots = clone_of_spot == clone
        for region in np.unique(regions):
            snps = regions == region
            f, n = fractions[clone, snps][0], depth[spots][:, snps].sum()
            drawn = planted_b[spots][:, snps].sum() / n
            assert abs(drawn - f) <= 4.0 * np.sqrt(f * (1.0 - f) / n)


def test_the_switches_follow_the_switch_model(config: YAMLConfig) -> None:
    """The switch count is `sum p` within 4 sigma, and the switches' mean `p` is `sum p^2 / sum p` within 4 sigma,
    above the mean over all SNP pairs: the switches fall where the model's probabilities are highest. The switch
    model reads only the SNPs, which every source shares byte for byte, so one draw stands for each."""
    sources = [fixture(pinned.source) / "unique_snp_ids.npy.gz" for pinned in perfect.get_perfect().values()]
    assert len({gzip.decompress(path.read_bytes()) for path in sources}) == 1
    chrom, pos = perfect.get_snp_loci(read_numpy(sources[0].with_suffix(""), allow_pickle=True))
    p = perfect.get_switch_probs(chrom, pos, REFERENCES / get_resources()["geneticmap_file"], config)[:-1]
    phase = perfect.draw_phase(np.append(p, 0.0), np.random.default_rng(3))
    switched = np.diff(phase) != 0
    assert abs(switched.sum() - p.sum()) < 4.0 * np.sqrt(p.sum())
    at_switches = p[switched]
    expected = (p**2).sum() / p.sum()
    assert abs(at_switches.mean() - expected) < 4.0 * at_switches.std() / np.sqrt(switched.sum())
    assert at_switches.mean() > p.mean()
