"""`cnamaste.sim.plot` on dev_tree easy r0 (`e4bbe03f`), drawn fresh: both truth pages reproduce byte for byte, the
committed fixture's are the same bytes, and the pseudobulk by true clone the page draws reads the planted `(A, B)` at
the clone's normal fraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cnamaste.qa.sims import fixture, grch38
from cnamaste.scripts.run_sim_plot import main
from cnamaste.sim.draw import stream
from cnamaste.sim.plot import (
    get_binned_counts,
    get_binned_profile,
    get_bins,
    get_normal_fractions,
    get_planted_tracks,
    get_truth_pages,
    write_truth_pages,
)
from cnamaste.truth import Truth, read_sim_truth

pytestmark = pytest.mark.skipif(grch38() is None, reason="no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")


@pytest.fixture(scope="module")
def dev_tree_easy(tmp_path_factory: pytest.TempPathFactory) -> Truth:
    """dev_tree easy r0's truth, drawn by `cnamaste.sim` (`e4bbe03f`)."""
    _, directory = next(stream("dev_tree_easy", {0}, tmp_path_factory.mktemp("sim")))
    assert directory is not None and directory.name == "dev_tree_easy_e4bbe03f"
    return read_sim_truth(directory)


@pytest.mark.slow
@pytest.mark.very_slow
def test_run_sim_plot_on_the_committed_fixture_draws_the_fresh_draw_s_pages(
    dev_tree_easy: Truth, tmp_path: Path
) -> None:
    """`run_sim_plot sim/dev_tree_easy_e4bbe03f`, reading its committed `manifest.json`, writes the fresh draw's pages
    byte for byte (`6f25bc80`, `e1eaa4fd`): no creation date, nothing drawn from state or from where the draw lies."""
    fresh = write_truth_pages(dev_tree_easy, tmp_path / "fresh")
    assert [p.name for p in fresh] == list(get_truth_pages().values())
    assert main([str(fixture("e4bbe03f")), "--into", str(tmp_path / "committed")]) == 0
    for page in fresh:
        assert (tmp_path / "committed" / page.name).read_bytes() == page.read_bytes()


@pytest.mark.slow
def test_the_pseudobulk_by_true_clone_reads_the_planted_profile(dev_tree_easy: Truth) -> None:
    """Per tumour clone and planted `(A, B)` other than `(1, 1)`, its 1 Mb bins pooled: BAF within 0.02 of
    `eqn:ab_admixture`'s at normal fraction 0.14 (0.0076 measured), RDR within 10% of its depth over `Z_c` (8.2%
    measured, 13 bins of clone III's (0, 2)). Unmixed, (1, 0)'s BAF is 0.07 off; diploid, its RDR 43%."""
    bins = get_bins(dev_tree_easy.get_simulation()["chromosome_lengths"])
    profile = get_binned_profile(dev_tree_easy, bins)
    counts = get_binned_counts(dev_tree_easy, bins).with_labelling("clone", dev_tree_easy.labels)
    pseudobulk = counts.get_pseudobulk("clone")
    rdr, baf, _ = get_planted_tracks(profile, pseudobulk, get_normal_fractions(dev_tree_easy))
    checked = 0
    for c in range(1, dev_tree_easy.n_clones):
        pairs = profile[[f"clone{c} A", f"clone{c} B"]].to_numpy()
        for state in np.unique(pairs, axis=0):
            on = (pairs == state).all(axis=1) & (pseudobulk.total_snp_umis[c] > 0)
            if tuple(state) == (1, 1):
                continue
            expected_umis = np.sum(pseudobulk.normal_baseline_sized[c][on] * rdr[c][on])
            assert abs(np.sum(pseudobulk.umis[c][on]) / expected_umis - 1) < 0.1
            b_share = np.sum(pseudobulk.alt_snp_umis[c][on]) / np.sum(pseudobulk.total_snp_umis[c][on])
            assert abs(b_share - baf[c][on][0]) < 0.02
            checked += 1
    assert checked == 7
