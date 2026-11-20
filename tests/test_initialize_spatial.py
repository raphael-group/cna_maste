"""The initial labellings, each `CLONE_INIT_TYPES`: `rectangle` (`get_rectangle_labelling`, `get_block_labelling`) and
`median_bisection` (`get_bisection_labelling`, `get_slice_bisection_labelling`), at the `phasing` and `rdr` stages of
`get_initial_labelling`.

Referees: `rectangle`'s indices pinned to 5fbdd73's (`sandbox.spatial`) on the two-slice draws; each part connected in its slice's lattice adjacency (`construct_multislice_lattice_adjacency`), on a
synthetic Visium array and Visium HD grid and on the two-slice draws dev_tree easy (`e4bbe03f`, triangular) and
dev_tree easy square (`bdff091f`, square); the part sizes `size_bounds` derives from delta; the largest-remainder
allocation; and a slice's parts unmoved by another slice's positions. It costs 3 s.
"""

from __future__ import annotations

import hashlib
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score

from cnamaste.config import config_defaults, get_clone_init_types
from cnamaste.initialize.spatial import (
    SpotPositions,
    get_baf_distinction,
    get_best_start,
    get_bisection_labelling,
    get_block_labelling,
    get_initial_labelling,
    get_parts_per_slice,
    get_slice_bisection_labelling,
)
from cnamaste.platforms.visium import visium_physical
from cnamaste.spatial import construct_multislice_lattice_adjacency
from cnamaste.spatio_genomic_counts import SpatioGenomicCounts
from test_lattice_adjacency import get_two_slices, two_slice_positions

SEEDS = range(10)
JITTER = config_defaults["hmrf"]["bisection_quantile_jitter"]

RECTANGLE_PINS = {
    "e4bbe03f": ("a4337955", {2: "6c3dc203", 3: "8d5757dc", 4: "715baa33"}),
    "bdff091f": ("84579cb5", {2: "ca8595c3", 3: "11e05560", 4: "8c9b0899"}),
}
"""Draw -> sha256[:8] of 5fbdd73's phasing start (`initialize_clones`, 2 x 2 per slice), and of its RDR split of the
first slice (`get_block_labelling`, `random_state` 0) by n_clones."""


def index_digest(labelling: np.ndarray, n_clones: int) -> str:
    """sha256[:8] of each of `n_clones` clones' spot indices, int64, in order."""
    clone_index = [np.flatnonzero(labelling == clone) for clone in range(n_clones)]
    return hashlib.sha256(b"".join(np.asarray(i, np.int64).tobytes() + b"|" for i in clone_index)).hexdigest()[:8]


def get_phasing_labelling(draw: str, clone_init_type: str) -> np.ndarray:
    """`get_initial_labelling` at `phasing` on the two-slice `draw`, seed 0."""
    config = run_config(clone_init_type, "rectangle", 2)
    rng = np.random.default_rng(0)
    return get_initial_labelling(None, SpotPositions(*two_slice_positions(draw)), config, stage="phasing", rng=rng)


def run_config(clone_init_type: str, rdr_clone_init_type: str, n_clones_rdr: int) -> SimpleNamespace:
    """The configuration keys the clone starts read, the rest at their defaults."""
    hmrf = {**config_defaults["hmrf"], "rdr_clone_init_type": rdr_clone_init_type, "n_clones_rdr": n_clones_rdr}
    return SimpleNamespace(
        phasing=SimpleNamespace(**{**config_defaults["phasing"], "clone_init_type": clone_init_type}),
        hmrf=SimpleNamespace(**hmrf),
        hmm=SimpleNamespace(**config_defaults["hmm"]),
    )


def synthetic_slice(lattice: str) -> np.ndarray:
    """A 40 x 30 slice's physical positions, um: a Visium array (`array_row + array_col` even) or 8 um bins."""
    if lattice == "triangular":
        index = np.array([(r, c) for r in range(40) for c in range(60) if (r + c) % 2 == 0], dtype=np.float64)
        return visium_physical(index)
    return np.array([(8.0 * c, -8.0 * r) for r in range(40) for c in range(30)])


def size_bounds(n_spots: int, n_parts: int) -> tuple[int, int]:
    """The fewest and most spots a part can hold when each cut of k parts takes k1 / k +- delta of its piece."""
    if n_parts == 1:
        return n_spots, n_spots
    k_lower = n_parts // 2
    lower = int(np.floor((k_lower / n_parts - JITTER) * n_spots))
    upper = int(np.ceil((k_lower / n_parts + JITTER) * n_spots))
    lows, highs = zip(
        size_bounds(lower, k_lower),
        size_bounds(n_spots - upper, n_parts - k_lower),
        size_bounds(upper, k_lower),
        size_bounds(n_spots - lower, n_parts - k_lower),
    )
    return min(lows), max(highs)


def assert_connected_parts(adjacency: object, parts: np.ndarray) -> None:
    """Each part is one connected component of the adjacency restricted to it."""
    for part in np.unique(parts):
        index = np.flatnonzero(parts == part)
        n_components, _ = connected_components(adjacency[index][:, index], directed=False)
        assert n_components == 1, (part, n_components)


@pytest.mark.parametrize("lattice", ["triangular", "square"])
@pytest.mark.parametrize("n_parts", [2, 3, 4, 9])
def test_bisection_cuts_connected_parts_within_the_delta_bound(lattice: str, n_parts: int) -> None:
    """On a synthetic array, over 10 seeds: exactly `n_parts` parts, labelled 0..n_parts - 1, each connected in the
    lattice adjacency and of a size within `size_bounds`. Fails where a cut leaves a part empty, split, or of a size the
    jitter does not admit."""
    physical = synthetic_slice(lattice)
    adjacency = construct_multislice_lattice_adjacency(np.zeros(len(physical), int), ["s"], physical, None, lattice)
    lower, upper = size_bounds(len(physical), n_parts)
    for seed in SEEDS:
        parts = get_bisection_labelling(physical, n_parts, np.random.default_rng(seed), JITTER)
        sizes = np.bincount(parts)
        assert len(sizes) == n_parts and parts.shape == (len(physical),)
        assert lower <= sizes.min() and sizes.max() <= upper, (seed, sizes.tolist(), lower, upper)
        assert_connected_parts(adjacency, parts)


def test_bisection_is_deterministic_per_seed_and_varies_across_seeds() -> None:
    """The same seed gives the same parts; 10 seeds give pairwise ARI below 0.9; NumPy's global state is untouched.
    Fails where the cut reads the global generator, or a seed does not move it."""
    physical = synthetic_slice("triangular")
    state = np.random.get_state()
    drawn = [get_bisection_labelling(physical, 9, np.random.default_rng(seed), JITTER) for seed in SEEDS]
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    assert np.array_equal(drawn[0], get_bisection_labelling(physical, 9, np.random.default_rng(0), JITTER))
    for first, second in itertools.combinations(drawn, 2):
        assert adjusted_rand_score(first, second) < 0.9


def test_parts_are_allocated_by_largest_remainder() -> None:
    """Quotas 2.5, 0.0125 and 1.4875 of 4 parts give 2, 1, 1: the empty quota is raised to 1, from the slice furthest
    above its quota; quotas of 6 over 300 and 100 spots give 4.5, 1.5 -> 5, 1 (ties to the first). Fails where a slice
    gets no part, or the total moves."""
    assert get_parts_per_slice([800, 4, 476], 4).tolist() == [2, 1, 1]
    assert get_parts_per_slice([300, 100], 6).tolist() == [5, 1]
    with pytest.raises(ValueError, match="cannot give each"):
        get_parts_per_slice([10, 10, 10], 2)


@pytest.mark.parametrize("draw", list(get_two_slices()))
def test_slice_bisection_on_the_two_slice_draws(draw: str) -> None:
    """The phasing start, `phasing.npart_phasing`^2 x 2 clones: none empty, each slice's share its largest-remainder
    quota and connected in its lattice; reflecting the first slice's positions leaves the second's parts unchanged.
    Fails where a slice's start reads another's positions, or a clone starts empty."""
    _, physical, sample_ids = two_slice_positions(draw)
    n_clones = config_defaults["phasing"]["npart_phasing"] ** 2 * 2
    clones = get_slice_bisection_labelling(physical, sample_ids, n_clones, np.random.default_rng(0), JITTER)
    assert np.bincount(clones, minlength=n_clones).min() > 0 and clones.max() == n_clones - 1
    parts_per_slice = [len(np.unique(clones[sample_ids == s])) for s in (0, 1)]
    assert parts_per_slice == get_parts_per_slice(np.bincount(sample_ids), n_clones).tolist()
    adjacency = construct_multislice_lattice_adjacency(sample_ids, ["a", "b"], physical, None, get_two_slices()[draw])
    assert_connected_parts(adjacency, clones)
    reflected = physical.copy()
    reflected[sample_ids == 0, 0] *= -1.0
    moved = get_slice_bisection_labelling(reflected, sample_ids, n_clones, np.random.default_rng(0), JITTER)
    assert np.array_equal(moved[sample_ids == 1], clones[sample_ids == 1])
    assert not np.array_equal(moved[sample_ids == 0], clones[sample_ids == 0])


@pytest.mark.parametrize("draw", list(get_two_slices()))
def test_rectangle_reproduces_the_old_starts(draw: str) -> None:
    """`rectangle` at both sites: the phasing start and the RDR split of the first slice at 2, 3 and 4 clones are
    5fbdd73's index for index (`RECTANGLE_PINS`); `median_bisection` selected instead gives its 2 x 4 clones. Fails
    where a default run would move from cleanup's, or the key does not select the start."""
    coords, _, sample_ids = two_slice_positions(draw)
    phasing_pin, rdr_pins = RECTANGLE_PINS[draw]
    assert index_digest(get_phasing_labelling(draw, "rectangle"), 8) == phasing_pin
    index = np.flatnonzero(sample_ids == 0)
    for n_clones, pin in rdr_pins.items():
        assert index_digest(get_block_labelling(coords[index], n_clones, 0), n_clones) == pin, n_clones
    assert np.unique(get_phasing_labelling(draw, "median_bisection")).size == 8


@pytest.mark.parametrize("rdr_clone_init_type", get_clone_init_types())
def test_rdr_refinement_splits_each_baf_clone_above_the_snp_umi_floor(rdr_clone_init_type: str) -> None:
    """Two BAF clones on a synthetic array, 3 segments: the first with 59 SNP UMIs, below 20 x 3, keeps 1 RDR clone;
    the second, at 60, is cut into `hmrf.n_clones_rdr` 3. Every RDR clone lies in its BAF clone and the labels are
    0..3; NumPy's global state is untouched. Fails where the floor is misread, an RDR clone crosses its BAF clone, or
    a start reads the global generator."""
    physical = synthetic_slice("square")
    baf = (physical[:, 0] > np.median(physical[:, 0])).astype(np.int64)
    n_obs = 3
    snp_umis = np.zeros((n_obs, len(physical)))
    snp_umis[0, np.flatnonzero(baf == 0)[0]] = 59.0
    snp_umis[0, np.flatnonzero(baf == 1)[0]] = 60.0
    counts = SimpleNamespace(
        labellings={"clone": baf}, total_snp_umis=SpatioGenomicCounts(snp_umis.T), n_segments=n_obs
    )
    state = np.random.get_state()
    rdr = get_initial_labelling(
        counts,
        SpotPositions(physical, physical, np.zeros(len(baf), dtype=int)),
        run_config("rectangle", rdr_clone_init_type, 3),
        stage="rdr",
        rng=np.random.default_rng(0),
    )
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    assert np.array_equal(np.unique(rdr), np.arange(4))
    assert set(rdr[baf == 0]) == {0} and set(rdr[baf == 1]) == {1, 2, 3}


def test_the_phasing_start_keeps_the_candidate_of_the_strongest_baf_distinction() -> None:
    """Two planted clones of 200 spots, 4 segments, 20 SNP UMIs per spot and segment: the second at BAF 0.2 on two
    segments, both 0.5 elsewhere, the written phase flipped on segments 1 and 3; deadband 0.1. A candidate aligned
    with the planted clones scores above one cutting each in half and is kept; with no CNA every candidate scores 0
    (noise about 1/2 is within the band); a clone under 20 x 4 SNP UMIs weighs nothing. Fails where the BAF is not
    folded (the flips cancel it), the band or the floor are misread, or the lesser is kept."""
    rng = np.random.default_rng(0)
    planted = np.repeat([0, 1], 200)
    total = np.full((400, 4), 20)
    halves = np.tile([0, 1], 200)
    for shift, kept in ((0.2, planted), (0.5, halves)):
        alt = rng.binomial(total, np.array([[0.5] * 4, [shift, shift, 0.5, 0.5]])[planted])
        alt[:, [1, 3]] = total[:, [1, 3]] - alt[:, [1, 3]]
        counts = SimpleNamespace(alt_snp_umis=SpatioGenomicCounts(alt), total_snp_umis=SpatioGenomicCounts(total))
        assert np.array_equal(get_best_start(counts, [halves, planted], 0.1), kept)
        scores = [
            get_baf_distinction(
                np.array([alt[labels == c].sum(axis=0) for c in (0, 1)]),
                np.array([total[labels == c].sum(axis=0) for c in (0, 1)]),
                80,
                0.1,
            )
            for labels in (planted, halves)
        ]
        assert scores[0] > 0.01 > scores[1] if shift < 0.5 else scores == [0.0, 0.0]
        # NB a second clone of 79 or 81 SNP UMIs at BAF 0 on segment 0: under the floor it weighs nothing.
        alt_0, total_0 = alt[planted == 0].sum(axis=0), total[planted == 0].sum(axis=0)
        for depth, scored in ((79, False), (81, True)):
            few = get_baf_distinction(np.array([alt_0, [0, 0, 0, 0]]), np.array([total_0, [depth, 0, 0, 0]]), 80, 0.1)
            assert (few > 1e-3) == scored
