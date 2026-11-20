//! `trws::trws` against enumeration of every labelling of small problems (T- #198): the bound never above the
//! minimum, the labelling's energy never below it, exact on trees, a certificate where bound and energy meet, a bound
//! that never falls from pass to pass, `max_iterations` a prefix of the longer run, and renaming the labels renaming
//! the result. The referee for agreement with OpenGM is `opengm_oracle.rs`.

mod common;

use cnamaste_expansion::trws::{trws, TrwsResult};
use common::{planted_lattice, Owned, SplitMix};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

/// Relative tolerance on an energy or a bound: sums of at most a few dozen terms, through a few dozen passes.
const RTOL: f64 = 1e-9;

fn config(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        failure_persistence: Some(Box::new(FileFailurePersistence::WithSource("regressions"))),
        ..ProptestConfig::default()
    }
}

/// A problem of 2 to 8 nodes and 2 to 4 labels, unary on [-2, 2), pairs on random distinct nodes, coupling on
/// (0, 2); `tree` joins each node to one before it, so the pairs form a tree.
fn problems(tree: bool) -> impl Strategy<Value = Owned> {
    (2usize..=8, 2usize..=4).prop_flat_map(move |(n, k)| {
        let pairs = if tree {
            proptest::collection::vec((0.0f64..1.0, 0.05f64..2.0), n - 1)
                .prop_map(move |draws| {
                    draws
                        .iter()
                        .enumerate()
                        .map(|(i, &(u, j))| ((u * (i + 1) as f64) as usize, i + 1, j))
                        .collect::<Vec<_>>()
                })
                .boxed()
        } else {
            proptest::collection::vec((0..n, 1..n, 0.05f64..2.0), 0..=2 * n)
                .prop_map(move |draws| {
                    draws
                        .into_iter()
                        .map(|(a, d, j)| {
                            let b = (a + d) % n;
                            (a.min(b), a.max(b), j)
                        })
                        .collect::<Vec<_>>()
                })
                .boxed()
        };
        (proptest::collection::vec(-2.0f64..2.0, n * k), pairs).prop_map(move |(unary, pairs)| Owned {
            n_states: k,
            unary,
            first: pairs.iter().map(|p| p.0).collect(),
            second: pairs.iter().map(|p| p.1).collect(),
            coupling: pairs.iter().map(|p| p.2).collect(),
        })
    })
}

fn solve(problem: &Owned, max_iterations: usize) -> TrwsResult {
    trws(&problem.view(), max_iterations, 1e-9)
}

/// The enumerated minimum of the Potts energy.
fn minimum(problem: &Owned) -> f64 {
    let total: f64 = problem.coupling.iter().sum();
    problem
        .enumerate()
        .iter()
        .map(|e| e.1 - total)
        .fold(f64::INFINITY, f64::min)
}

fn scale(x: f64) -> f64 {
    RTOL * x.abs().max(1.0)
}

proptest! {
    #![proptest_config(config(256))]

    /// Fails where the bound is not a lower bound, or the labelling reported is not the one its energy is of.
    #[test]
    fn the_bound_is_below_the_minimum_and_the_energy_above(problem in problems(false)) {
        let found = solve(&problem, 1000);
        let best = minimum(&problem);
        let energy = problem.view().potts_energy(&found.labels);
        prop_assert!(found.bound <= best + scale(best), "bound {} above minimum {}", found.bound, best);
        prop_assert!(energy >= best - scale(best), "energy {} below minimum {}", energy, best);
    }

    /// TRW-S is exact on a tree (Wainwright, Jaakkola and Willsky 2005; Kolmogorov 2006): fails where a chain's
    /// min-sum or the averaging leaves a gap there.
    #[test]
    fn on_a_tree_the_bound_and_the_energy_are_the_minimum(problem in problems(true)) {
        let found = solve(&problem, 1000);
        let best = minimum(&problem);
        prop_assert!((found.bound - best).abs() <= scale(best), "bound {} against {}", found.bound, best);
        prop_assert!((problem.view().potts_energy(&found.labels) - best).abs() <= scale(best));
    }

    /// Where bound and energy meet, the labelling is a minimum. Fails where either is misreported.
    #[test]
    fn a_closed_gap_certifies_a_minimum(problem in problems(false)) {
        let found = solve(&problem, 1000);
        let energy = problem.view().potts_energy(&found.labels);
        if (energy - found.bound).abs() <= scale(found.bound) {
            let best = minimum(&problem);
            prop_assert!((energy - best).abs() <= scale(best), "certified {} against minimum {}", energy, best);
        }
    }

    /// TRW-S's bound never falls from pass to pass (Kolmogorov 2006). Fails where a pass reads stale min-marginals.
    #[test]
    fn the_bound_never_falls(problem in problems(false)) {
        let found = solve(&problem, 1000);
        for pass in found.trace.windows(2) {
            prop_assert!(pass[1] >= pass[0] - scale(pass[0]), "{} after {}", pass[1], pass[0]);
        }
        prop_assert_eq!(*found.trace.last().unwrap(), found.bound);
    }

    /// Fails where `max_iterations` is exceeded or a shorter run is not the longer one's prefix.
    #[test]
    fn max_iterations_cuts_the_trace_to_a_prefix(problem in problems(false), cap in 1usize..6) {
        let (full, cut) = (solve(&problem, 1000), solve(&problem, cap));
        prop_assert!(cut.trace.len() <= cap + 1);
        prop_assert_eq!(&cut.trace[..], &full.trace[..cut.trace.len()]);
    }

    /// The labelling kept is the best rounded so far: its energy never rises as `max_iterations` grows, each run the
    /// prefix of the next. Fails where the last rounding replaces a better earlier one.
    #[test]
    fn the_best_labelling_so_far_is_kept(problem in problems(false)) {
        let passes = solve(&problem, 1000).trace.len();
        let energies: Vec<f64> = (1..passes.max(2))
            .map(|cap| problem.view().potts_energy(&solve(&problem, cap).labels))
            .collect();
        for pair in energies.windows(2) {
            prop_assert!(pair[1] <= pair[0], "{} after {}", pair[1], pair[0]);
        }
    }

    /// Permuting the labels permutes the result, bit for bit: every operation is per label but the minimum. Real
    /// values, so no two labels tie. Fails where a label's index enters a message or the rounding.
    #[test]
    fn renaming_the_labels_renames_the_result(problem in problems(false), seed in any::<u64>()) {
        let k = problem.n_states;
        let mut rng = SplitMix::new(seed);
        let mut order: Vec<usize> = (0..k).collect();
        for i in (1..k).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let mut renamed = problem.clone();
        for i in 0..problem.n_nodes() {
            for (x, &to) in order.iter().enumerate() {
                renamed.unary[i * k + to] = problem.unary[i * k + x];
            }
        }
        let (found, renamed_found) = (solve(&problem, 1000), solve(&renamed, 1000));
        prop_assert_eq!(&found.trace, &renamed_found.trace);
        for i in 0..problem.n_nodes() {
            prop_assert_eq!(order[found.labels[i]], renamed_found.labels[i]);
        }
    }
}

/// A planted 30 x 40 lattice of 4 clones: the bound rises pass to pass and meets the energy to the 1e-9 the run stops
/// at, from either side: once the gap closes, rounding puts the bound up to ~1e-13 relative above the energy, as
/// OpenGM's does on this problem (PR- #55 measured TRW-S's bound at its energy on planted lattices).
#[test]
fn a_planted_lattice_closes_its_gap() {
    let problem = planted_lattice(30, 40, 4, 1.0, 1.1, 3);
    let found = solve(&problem, 1000);
    let energy = problem.view().potts_energy(&found.labels);
    assert!(
        (energy - found.bound).abs() <= 1e-9 * found.bound.abs(),
        "gap {}",
        energy - found.bound
    );
    assert!(found.trace.windows(2).all(|w| w[1] >= w[0] - 1e-9 * w[0].abs()));
}

/// Nodes with no pair are chains of their own: the result is each one's unary argmin, the bound their sum.
#[test]
fn without_pairs_each_node_takes_its_unary_argmin() {
    let mut problem = planted_lattice(6, 7, 5, 0.5, 1.1, 11);
    problem.first.clear();
    problem.second.clear();
    problem.coupling.clear();
    let found = solve(&problem, 1000);
    assert_eq!(found.labels, common::argmin_labels(&problem));
    assert_eq!(found.bound, problem.view().potts_energy(&found.labels));
}
