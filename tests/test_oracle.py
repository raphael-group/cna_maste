"""The oracle tier: each inference step run alone on CalicoST easy (`2d4ce9a9`) and hard (`8797710b`), every other
step's answer the planted truth, scored on its own part (`cnamaste.qa.oracle`).

Inputs: the capture's recorded `08_rdr/run_hmrf` call (counts, start labelling, adjacency, configuration) and
`09_outputs/decode_integer_copies`' configuration; no pipeline run but O4's. Referee: `truth.read_truth`.

- O1 HMM: the truth clones' `run_hmrf` under `hmrf.fixed_assignment`, from `get_initial_parameters` on them; per
  tumour clone, the ARI of its decoded state path against its planted AB pair per bin, and breakpoint recall.
- O2 label solver: the field of the planted AB pairs (`qa.oracle.get_oracle_fit`, O1's dispersions and `log Z_c`)
  solved by `hmrf.get_hmrf_labelling`, per solver and start; clone ARI, and energy against the planted labelling's.
- O3 (A, B) decoder: `decode_integer_copies` on the truth clones' pseudobulks and O1's fit; per tumour clone
  `exact_altered`, `sensitivity`, `specificity` (`qa.metrics`), the fitted normal fraction reported (planted 0).
- O4 ablation: `run_cnamaste` with the planted clones as `annotation.clone_label`, free and held
  (`hmrf.fixed_assignment`); the panels reported beside the capture's full run. No hook plants the copy states.

Floors: the first measured value rounded down to 2 decimals. O1-O3 37 s serial, O4 about 1.3 min per run; 215 s on 4
workers (`oracle`, off unless asked for by -m).
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

import cnamaste.scripts.run_cnamaste as rc
from cnamaste import external
from cnamaste.config import set_global_config
from cnamaste.emission import CnaEmissionConfounders
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmrf import field_argmax, get_hmrf_labelling, run_hmrf
from cnamaste.hmrf_fit import HmrfFit
from cnamaste.potts import potts_energy
from cnamaste.qa.capture import Capture, get_entry_point_seed, get_output_dir, get_overrides, get_template
from cnamaste.qa.metrics import PANEL_METRICS, exact_altered, metrics, read_estimate, sensitivity, specificity
from cnamaste.qa.oracle import (
    get_breakpoint_recall,
    get_clone_estimate,
    get_clone_label_frame,
    get_oracle_field,
    get_oracle_fit,
    get_seglevel,
    get_start_on_truth,
    get_truth_bin_copies,
    get_truth_labelling,
)
from cnamaste.qa.sim_runs import run_dir_of
from cnamaste.qa.sims import get_capture_path, get_fixtures, get_project
from cnamaste.spatio_genomic_counts import PseudobulkCounts
from cnamaste.truth import Truth, read_truth

pytestmark = pytest.mark.oracle

BREAKPOINT_TOLERANCE = 1
"""A planted breakpoint is recalled by a decoded one within this many bins."""

ENERGY_RTOL = 1e-9
"""A solver's energy is under the planted labelling's plus this fraction of its magnitude (1.4e-3 at easy)."""

# NB measured at d2268d1 (cleanup2 aede44b + this PR); per tumour clone 1..3. CalicoST easy (`2d4ce9a9`), hard (`8797710b`).
O1_FLOORS = {
    "2d4ce9a9": {"ari": (0.53, 0.82, 0.67), "breakpoint_recall": (0.66, 1.0, 0.6)},
    "8797710b": {"ari": (0.55, 0.17, 0.46), "breakpoint_recall": (0.81, 0.87, 1.0)},
}

# NB measured at d2268d1: clone ARI per (solver, start).
O2_FLOORS = {
    "2d4ce9a9": {
        ("opengm_expansion", "usual"): 0.94,
        ("opengm_expansion", "field_argmax"): 0.94,
        ("opengm_icm", "usual"): 0.95,
        ("opengm_icm", "field_argmax"): 0.93,
        ("icm", "usual"): 0.95,
        ("icm", "field_argmax"): 0.92,
    },
    "8797710b": {
        ("opengm_expansion", "usual"): 0.92,
        ("opengm_expansion", "field_argmax"): 0.92,
        ("opengm_icm", "usual"): 0.95,
        ("opengm_icm", "field_argmax"): 0.66,
        ("icm", "usual"): 0.95,
        ("icm", "field_argmax"): 0.61,
    },
}

O2_LOCAL_MINIMA = {("8797710b", "opengm_icm", "field_argmax"), ("8797710b", "icm", "field_argmax")}
"""The rows whose solver stops above the planted labelling's energy (+200.8 and +302.5 measured): a local minimum."""

# NB measured at d2268d1; per tumour clone 1..3.
O3_FLOORS = {
    "2d4ce9a9": {
        "exact_altered": (0.33, 0.62, 0.91),
        "sensitivity": (0.63, 0.98, 0.97),
        "specificity": (0.99, 0.99, 0.99),
    },
    "8797710b": {
        "exact_altered": (0.43, 0.44, 0.6),
        "sensitivity": (0.92, 0.87, 0.94),
        "specificity": (0.99, 0.99, 0.99),
    },
}

O2_SOLVERS = ("opengm_expansion", "opengm_icm", "icm")
O2_STARTS = ("usual", "field_argmax")


class OracleInputs(NamedTuple):
    """A fixture's recorded RDR stage, its truth and O1's fit on the truth clones."""

    truth: Truth
    counts: Any
    start: np.ndarray
    adjacency: Any
    config: Any
    labelling: np.ndarray
    pseudobulk: PseudobulkCounts
    codes: np.ndarray
    fit: HmrfFit


@functools.cache
def get_oracle_inputs(sim_hash: str) -> OracleInputs:
    """The fixture's `08_rdr/run_hmrf` inputs, its truth on them and O1: `run_hmrf` held at the truth clones."""
    capture = Capture(get_capture_path(sim_hash))
    try:
        barcodes = capture.decode("09_outputs/get_assignment_frame_0")[1][0]
        initialize, _, init_kwargs = capture.decode("08_rdr/get_initial_parameters")
        _, (counts, start, adjacency, config), _ = capture.decode("08_rdr/run_hmrf")
    finally:
        capture.close()
    truth = read_truth(get_project() / "sim" / get_fixtures()[sim_hash])
    labelling = get_truth_labelling(truth, barcodes)
    held = copy.deepcopy(config)
    held.hmrf.fixed_assignment = True
    parameters = initialize(counts, labelling, held, **init_kwargs)
    fit = run_hmrf(counts, labelling, adjacency, held, parameters=parameters, stage="rdr")
    pseudobulk = counts.with_labelling("clone", labelling).get_pseudobulk("clone")
    return OracleInputs(
        truth, counts, start, adjacency, config, labelling, pseudobulk, get_truth_bin_copies(truth, counts), fit
    )


def test_o1_hmm_recovers_each_clone_s_copy_state_path(sim_hash: str) -> None:
    """Given the truth clones, each tumour clone's decoded state path partitions its bins as its planted AB pairs do
    (ARI) and places a breakpoint within one bin of each planted one, at or above `O1_FLOORS`."""
    inputs = get_oracle_inputs(sim_hash)
    path, contig = inputs.fit["pred_cnv"], np.asarray(inputs.counts.contig)
    tumour = range(1, inputs.truth.n_clones)
    ari = [adjusted_rand_score(inputs.codes[:, c], path[c]) for c in tumour]
    recall = [get_breakpoint_recall(inputs.codes[:, c], path[c], contig, BREAKPOINT_TOLERANCE) for c in tumour]
    print(f"{sim_hash} O1 ari {np.round(ari, 4).tolist()}, breakpoint_recall {np.round(recall, 4).tolist()}")
    assert np.all(np.array(ari) >= O1_FLOORS[sim_hash]["ari"])
    assert np.all(np.array(recall) >= O1_FLOORS[sim_hash]["breakpoint_recall"])


@pytest.mark.parametrize("start", O2_STARTS)
@pytest.mark.parametrize("solver", O2_SOLVERS)
def test_o2_label_solver_recovers_the_clones(sim_hash: str, solver: str, start: str) -> None:
    """Given the planted copy states, the solver's labelling scores clone ARI at or above `O2_FLOORS` and an energy
    under the planted labelling's plus `ENERGY_RTOL`, above it on `O2_LOCAL_MINIMA`, from the run's start mapped onto
    the truth clones or the field's argmax."""
    if solver.startswith("opengm") and not external.available():
        pytest.skip(str(external.unavailable()))
    inputs = get_oracle_inputs(sim_hash)
    oracle = get_oracle_fit(inputs.fit, inputs.codes, inputs.pseudobulk)
    field = get_oracle_field(inputs.counts, oracle)
    starts = {
        "usual": get_start_on_truth(inputs.start, inputs.labelling, inputs.truth.n_clones),
        "field_argmax": field_argmax(field),
    }
    config = copy.deepcopy(inputs.config)
    config.hmrf.label_solver = solver
    set_global_config(config)
    labels = get_hmrf_labelling(inputs.counts, oracle, inputs.adjacency, starts[start], config, merge=False, index=0)[0]
    spatial_weight = config.hmrf.spatial_weight
    planted = potts_energy(field, inputs.adjacency, inputs.labelling, spatial_weight)
    energy = potts_energy(field, inputs.adjacency, labels, spatial_weight)
    ari = adjusted_rand_score(inputs.labelling, labels)
    print(f"{sim_hash} O2 {solver} {start}: ari {ari:.4f}, energy - planted {energy - planted:.4f}")
    assert ari >= O2_FLOORS[sim_hash][(solver, start)]
    assert (energy < planted + ENERGY_RTOL * abs(planted)) == ((sim_hash, solver, start) not in O2_LOCAL_MINIMA)


def test_o3_decoder_recovers_each_clone_s_integer_copies(sim_hash: str) -> None:
    """Given the truth clones' pseudobulks and O1's fit, the decoded `(A, B)` score `exact_altered`, `sensitivity` and
    `specificity` per tumour clone at or above `O3_FLOORS`; the fitted normal fractions are reported, planted 0."""
    inputs = get_oracle_inputs(sim_hash)
    capture = Capture(get_capture_path(sim_hash))
    try:
        decode, args, _ = capture.decode("09_outputs/decode_integer_copies")
    finally:
        capture.close()
    config = args[4]
    n_clones = inputs.truth.n_clones
    # NB the normal clone 0 held at theta 1, as run_integer_copy_decoding holds it.
    start = np.where(np.arange(n_clones) == 0, 1.0, config.ab_decode.normal_fraction_start).reshape(-1, 1)
    alphas, taus = inputs.fit.parameters.alphas, inputs.fit.parameters.taus
    decoded = decode(
        inputs.pseudobulk,
        CnaHmmParameters(None, None, None, alphas, taus),
        CnaEmissionConfounders(start, inputs.fit.confounders.log_mu_shift),
        inputs.counts.lengths,
        config,
    )
    seglevel = get_seglevel(inputs.counts, decoded.copies)
    panels = {"exact_altered": exact_altered, "sensitivity": sensitivity, "specificity": specificity}
    found = {
        name: [panel(get_clone_estimate(inputs.truth, seglevel, c), inputs.truth) for c in range(1, n_clones)]
        for name, panel in panels.items()
    }
    print(
        f"{sim_hash} O3 { ({k: np.round(v, 4).tolist() for k, v in found.items()}) }, normal fractions "
        f"{np.round(decoded.normal_fractions[1:], 4).tolist()}"
    )
    for name, values in found.items():
        assert np.all(np.array(values) >= O3_FLOORS[sim_hash][name]), name


O4_RUNGS = {
    "known_clones": [],
    "known_clones_held": ["hmrf.fixed_assignment=true"],
}
"""Each ablation rung's overrides beyond `annotation.clone_label`."""


@pytest.mark.parametrize("rung", O4_RUNGS)
def test_o4_ablation_run_completes(sim_hash: str, rung: str, scratch: Path) -> None:
    """`run_cnamaste` with the planted clones as `annotation.clone_label` completes; its panels are reported beside
    the capture's full run, in the capture's scratch directory."""
    if not external.available():
        pytest.skip(str(external.unavailable()))
    truth = read_truth(get_project() / "sim" / get_fixtures()[sim_hash])
    capture = Capture(get_capture_path(sim_hash))
    try:
        barcodes = capture.decode("09_outputs/get_assignment_frame_0")[1][0]
        full = {key: float(capture.h5["metrics"].attrs[key]) for key in PANEL_METRICS}
    finally:
        capture.close()
    clone_label = scratch / "clone_label.tsv"
    get_clone_label_frame(truth, barcodes).to_csv(clone_label, sep="\t")
    over_rides = [
        *get_overrides(get_fixtures()[sim_hash]),
        f"annotation.clone_label={clone_label}",
        *O4_RUNGS[rung],
    ]
    np.random.seed(get_entry_point_seed())
    rc.run_cnamaste(str(get_template()), over_rides=over_rides)
    found = metrics(read_estimate(run_dir_of(scratch / get_output_dir())), truth)
    print(f"{sim_hash} O4 full { ({k: round(v, 4) for k, v in full.items()}) }")
    print(f"{sim_hash} O4 {rung} { ({k: round(v, 4) for k, v in found.items()}) }")
    assert np.isfinite(list(found.values())).all()
