"""`cnamaste.sim.study` on `hd_tree` (T- #190): a Felsenstein tree plants exactly `cnas` CNAs, one or more on each
edge below normal; the study order is a permutation of its 45 samples; its first sample is pinned and read by
`run_cnamaste` as Visium HD's square lattice with 8 neighbours, its normal clone CNA-free and on its slice; and that
sample streams through cnamaste, scored, its truth pages written, and CalicoST, to its first BAF-only inference."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from cnamaste import external
from cnamaste.config import YAMLConfig
from cnamaste.io import get_spatial_positions
from cnamaste.platforms.visium import physical_coords, reads_visium_hd
from cnamaste.qa.sim_runs import default_solver
from cnamaste.qa.sim_study import drop_sample_counts, stream_study_sample
from cnamaste.qa.sims import get_fixture_hash, grch38
from cnamaste.results import run_log_name
from cnamaste.sim.manifest import get_simulation
from cnamaste.sim.study import draw_study_sample, get_sample_name, get_study_order, get_study_simulations
from cnamaste.sim.tree import draw_edge_counts, draw_tree, draw_tree_clone_count
from cnamaste.spatial import construct_multislice_lattice_adjacency, validate_expected_adjacency
from cnamaste.truth import read_truth

REFERENCES = grch38()

RING = {(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)} - {(0, 0)}
"""A square bin's 4 edge and 4 corner neighbours, in array `(row, col)` offsets."""


def test_a_study_tree_plants_exactly_n_cnas_on_every_edge_below_normal() -> None:
    """Over 50 truths of each of the 9 simulations: `cnas` CNAs in all, at least one on each of the `2k - 1` edges
    below normal, none on normal, no `founder`; every clone count `k` of 2-4 with `2k - 1 <= cnas` is drawn, and no
    edge-count draw reaches the cap (it raises)."""
    for simulation in get_study_simulations("hd_tree"):
        n, counts = simulation.cna.cnas, set()
        assert n is not None
        for seed in range(50):
            rng = np.random.default_rng(seed)
            clones = draw_tree_clone_count(simulation, rng)
            tree = draw_tree(simulation._replace(cna=simulation.cna._replace(n_clones=clones)), rng)
            below = [node for node in tree.parent if node != "normal"]
            counts.add(clones)
            assert len(below) == 2 * clones - 1 <= n and "founder" not in tree.parent
            assert all(len(tree.edge_events[node]) >= 1 for node in below) and tree.edge_events["normal"] == ()
            assert sum(len(tree.edge_events[node]) for node in below) == n
        assert counts == {k for k in (2, 3, 4) if 2 * k - 1 <= n}


def test_edge_counts_are_one_plus_poisson_given_their_total() -> None:
    """At n = 9 CNAs on m = 5 edges, 10,000 draws of `1 + Poisson(lambda)` given the total: each total is 9 and each
    edge's extras follow Binomial(4, 1/5), the marginal of the uniform multinomial iid Poissons given their total
    are, to 0.02 per value (4 binomial SE at 10,000 draws)."""
    rng = np.random.default_rng(0)
    drawn = np.array([draw_edge_counts(9, 5, rng) for _ in range(10_000)])
    assert np.all(drawn.sum(axis=1) == 9) and drawn.min() == 1
    expected = binom.pmf(np.arange(5), 4, 0.2)
    for edge in drawn.T:
        assert np.allclose(np.bincount(edge - 1, minlength=5) / edge.size, expected, rtol=0.0, atol=0.02)


def test_the_study_order_is_a_seeded_permutation_of_its_45_samples() -> None:
    """45 distinct `(cnas, length, realization)` samples, the same order on each call, not the grid's."""
    names = [get_sample_name(s, k) for s, k in get_study_order("hd_tree")]
    grid = [get_sample_name(s, k) for s in get_study_simulations("hd_tree") for k in range(s.realizations)]
    assert len(grid) == 45 and sorted(names) == sorted(grid) and names != grid
    assert names == [get_sample_name(s, k) for s, k in get_study_order("hd_tree")]


@pytest.mark.slow
@pytest.mark.skipif(REFERENCES is None, reason="no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
def test_the_first_sample_is_its_pin_and_reads_as_visium_hd(tmp_path: Path) -> None:
    """The first sample in study order is `hd_tree_cnas6_length20mb_r1_67fc47de`. Its slice reads as Visium HD: the
    `square` adjacency at `coordination_num` 8 passes `validate_visiumhd_adjacency` and joins every interior bin to
    exactly its 8-ring of array offsets. `normal` is (1, 1) everywhere and holds bins of the slice. Its `config.yaml`
    passes the configuration's checks with the hd_tree overlay: `int_clone_agreement` 1.0, copy caps 8,
    `t` 0.999, `nonbalance_min_baf_dist` 0.1, `npart_phasing` 5 as 5 x 5 rectangles, one start candidate,
    `spatial_weight` 1.1, square at 8 neighbours. The stream's `drop_sample_counts` deletes exactly its six count
    files, its truth still read."""
    simulation, k = get_study_order("hd_tree")[0]
    realization, directory = draw_study_sample(simulation, k, tmp_path)
    assert directory.name == "hd_tree_cnas6_length20mb_r1_67fc47de" and get_fixture_hash(directory) == "67fc47de"
    config = YAMLConfig.from_file(directory / "config.yaml")
    config.issue_warnings()  # type: ignore[no-untyped-call]
    read = config.to_dict()
    assert (read["ab_decode"] | read["phasing"] | read["hmrf"]).items() >= {
        "int_clone_agreement": 1.0,
        "max_total_copy": 8,
        "max_allele_copy": 8,
        "npart_phasing": 5,
        "clone_init_type": "rectangle",
        "n_start_candidates": 1,
        "t": 0.999,
        "nonbalance_min_baf_dist": 0.1,
        "spatial_weight": 1.1,
        "lattice_type": "square",
        "coordination_num": 8,
    }.items()
    planted = realization.planted
    assert (planted.profile[["normal_A_copy", "normal_B_copy"]] == 1).all(axis=None)
    assert all(np.any(labels == 0) for labels in planted.labels)

    spaceranger_dir = directory / planted.sample_ids[0]
    assert reads_visium_hd(spaceranger_dir)  # type: ignore[no-untyped-call]
    positions = get_spatial_positions(spaceranger_dir)  # type: ignore[no-untyped-call]
    coords = positions[["x", "y"]].to_numpy(np.float64)
    slices = np.zeros(len(coords), dtype=int)
    physical = physical_coords(coords, slices, [spaceranger_dir], lattice="square")  # type: ignore[no-untyped-call]
    lattice = construct_multislice_lattice_adjacency(slices, ["s"], physical, None, "square", 8)  # type: ignore[no-untyped-call]
    adjacency = lattice.tocsr()
    assert validate_expected_adjacency(physical, adjacency, "square", 8) > 0.9  # type: ignore[no-untyped-call]
    index = positions[["array_row", "array_col"]].to_numpy(np.int64)
    _, rows, columns = simulation.array
    interior = np.flatnonzero(np.all((index > 0) & (index < np.array([rows, columns]) - 1), axis=1))
    for spot in interior:
        found = {
            tuple(index[j] - index[spot])
            for j in adjacency.indices[adjacency.indptr[spot] : adjacency.indptr[spot + 1]]
        }
        assert found == RING
    assert interior.size == (rows - 2) * (columns - 2)

    # NB the stream's drop: exactly the count files go; the truth stays readable.
    before = {p for p in directory.rglob("*") if p.is_file()}
    dropped = drop_sample_counts(directory)
    assert sorted(p.relative_to(directory).as_posix() for p in dropped) == [
        f"{planted.sample_ids[0]}/filtered_feature_bc_matrix.h5ad",
        f"{planted.sample_ids[0]}/spatial/tissue_positions.csv",
        "snp/barcodes.txt",
        "snp/cell_snp_Aallele.npz",
        "snp/cell_snp_Ballele.npz",
        "snp/unique_snp_ids.npy",
    ]
    assert {p for p in directory.rglob("*") if p.is_file()} == before - set(dropped)
    assert np.array_equal(read_truth(directory).labels, planted.labels[0])


@pytest.mark.cache
@pytest.mark.skipif(REFERENCES is None, reason="no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
@pytest.mark.skipif(not external.available(), reason="no OpenGM build: $CNAMASTE_OPENGM_HOME")
def test_a_small_stream_fills_the_compile_caches(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """`pytest -m cache` before `run_sim_study`: explicit_easy's two clones and five CNAs on a 30 x 30 hd_tree slice
    streamed through cnamaste, so numba's and JAX's on-disk caches hold the stream's kernels and the first streamed
    sample is not timed with their compilation. Run only where `-m` asks for `cache`."""
    if re.search(r"(?<!not )\bcache\b", str(request.config.getoption("markexpr"))) is None:
        pytest.skip("fills the compile caches: pytest -m cache")
    simulation = get_study_order("hd_tree")[0][0]
    small = simulation._replace(name="hd_tree_cache", array=("square", 30, 30), cna=get_simulation("explicit_easy").cna)
    record = stream_study_sample(small, 0, tmp_path, ("cnamaste",), default_solver, stop_calicost=True)
    assert all(math.isfinite(v) for v in record["cnamaste"].values() if isinstance(v, float))


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.skipif(REFERENCES is None, reason="no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
@pytest.mark.skipif(not external.available(), reason="no OpenGM build: $CNAMASTE_OPENGM_HOME")
def test_a_stream_of_one_runs_cnamaste_and_writes_the_truth_pages(tmp_path: Path) -> None:
    """The first sample through `stream_study_sample`, cnamaste alone: it completes on the square lattice with 8
    neighbours (`validate_visiumhd_adjacency`'s log line), its run scores finite, both truth pages are written and the
    counts are kept. Metrics, seconds and peak MB are recorded under `qa/stream/` and printed, not asserted."""
    simulation, k = get_study_order("hd_tree")[0]
    record = stream_study_sample(simulation, k, tmp_path, ("cnamaste",), default_solver, stop_calicost=True)
    print(json.dumps(record, default=str))
    cnamaste, sample = record["cnamaste"], tmp_path / "simulated_data_related" / record["sample_id"]
    run_dir = Path(cnamaste["run_dir"])
    assert "4 edges and 4 corners" in (run_dir / run_log_name(run_dir.name.removeprefix("run_"))).read_text()
    assert all(math.isfinite(v) for v in cnamaste.values() if isinstance(v, float))
    assert sorted(record["truth_pages"]) == ["qa/truth_combined.pdf", "qa/truth_combined_spatial.pdf"]
    assert all((sample / page).is_file() for page in record["truth_pages"]) and record["dropped"] == []


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.skip(
    reason="CalicoST is parked on this host (T- #193): 800 s and 10.8 GB to its first inference at 10,000 bins"
)
def test_a_stream_of_one_starts_calicost(tmp_path: Path) -> None:
    """The first sample through `stream_study_sample`, CalicoST alone: it reads the inputs, builds its adjacency and
    starts its first BAF-only inference, where it is stopped; its neighbour counts are recorded and printed."""
    simulation, k = get_study_order("hd_tree")[0]
    record = stream_study_sample(simulation, k, tmp_path, ("calicost",), default_solver, stop_calicost=True)
    print(json.dumps(record, default=str))
    calicost = record["calicost"]
    parsed = tmp_path / "nomixing_calicost_related" / record["sample_id"] / "parsed_inputs"
    assert calicost["started_inference"] and (parsed / "table_meta.csv.gz").is_file()
    assert math.isfinite(calicost["smooth_neighbours"]) and math.isfinite(calicost["adjacency_neighbours"])
