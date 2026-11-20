//! `expansion::expand_with` against enumeration of every labelling of small problems, and its own stop rule, with
//! OpenGM's moves and T- #195's (`opengm_patch`): the result within twice the minimum (Boykov, Veksler and Zabih
//! 2001), reached by moves that never raise the energy, stopped after `n_states` moves that lower it by nothing; the
//! two energies differ by the summed coupling; renaming nodes or reordering pairs renames the result. T- #195's result
//! is a local minimum under every alpha move; OpenGM's is not, pinned on two nodes.

mod common;

use cnamaste_expansion::expansion::expand_with;
use common::{argmin_labels, planted_lattice, Owned};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

/// Relative tolerance on an energy: sums of at most a few dozen terms.
const RTOL: f64 = 1e-12;

fn config(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        failure_persistence: Some(Box::new(FileFailurePersistence::WithSource("regressions"))),
        ..ProptestConfig::default()
    }
}

/// A problem of 2 to 8 nodes and 2 to 4 labels, unary on [0, 4), pairs on random distinct nodes with coupling on
/// (0, 2); `integer` rounds both to halves, so energies are summed exactly and moves tie.
fn problems(integer: bool) -> impl Strategy<Value = Owned> {
    (2usize..=8, 2usize..=4)
        .prop_filter("at most 4**8 labellings", |(n, k)| k.pow(*n as u32) <= 1 << 16)
        .prop_flat_map(move |(n, k)| {
            let round = move |x: f64| if integer { (2.0 * x).round() / 2.0 } else { x };
            let pair = (0..n, 1..n, 0.01f64..2.0).prop_map(move |(a, d, j)| {
                let b = (a + d) % n;
                (a.min(b), a.max(b), round(j).max(0.5))
            });
            (
                proptest::collection::vec(0.0f64..4.0, n * k),
                proptest::collection::vec(pair, 0..=2 * n),
            )
                .prop_map(move |(unary, pairs)| Owned {
                    n_states: k,
                    unary: unary.into_iter().map(round).collect(),
                    first: pairs.iter().map(|p| p.0).collect(),
                    second: pairs.iter().map(|p| p.1).collect(),
                    coupling: pairs.iter().map(|p| p.2).collect(),
                })
        })
}

fn starts(problem: &Owned) -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(0..problem.n_states, problem.n_nodes())
}

fn problems_and_starts(integer: bool) -> impl Strategy<Value = (Owned, Vec<usize>)> {
    problems(integer).prop_flat_map(|p| {
        let s = starts(&p);
        (Just(p), s)
    })
}

fn expanded(problem: &Owned, start: &[usize], max_steps: usize, patch: bool) -> (Vec<usize>, usize) {
    let mut labels = start.to_vec();
    let steps = expand_with(&problem.view(), &mut labels, max_steps, patch);
    (labels, steps)
}

proptest! {
    #![proptest_config(config(256))]

    /// T- #195's moves: fails where a move is not the optimal alpha move, or the stop comes before convergence.
    /// OpenGM's fail it (`opengm_s_move_drops_a_pair_labelled_apart_off_alpha`).
    #[test]
    fn the_result_is_a_local_minimum_under_every_alpha_move((problem, start) in problems_and_starts(false)) {
        let (labels, _) = expanded(&problem, &start, 1000, true);
        let energy = problem.view().metric_energy(&labels);
        for (other, e) in problem.enumerate() {
            for alpha in 0..problem.n_states {
                let reachable = other.iter().zip(&labels).all(|(&y, &x)| y == x || y == alpha);
                prop_assert!(!reachable || e >= energy - RTOL * energy.abs().max(1.0),
                    "expanding {} reaches {} below {}", alpha, e, energy);
            }
        }
    }

    /// Unary non-negative and coupling positive: fails where the result is not within twice the minimum.
    #[test]
    fn the_result_is_within_twice_the_minimum((problem, start) in problems_and_starts(false), patch in any::<bool>()) {
        let (labels, _) = expanded(&problem, &start, 1000, patch);
        let best = problem.enumerate().iter().map(|e| e.1).fold(f64::INFINITY, f64::min);
        prop_assert!(problem.view().metric_energy(&labels) <= 2.0 * best + RTOL * best.abs().max(1.0));
    }

    /// Each move's energy against the last, and the stop: steps below `max_steps` end `n_states` moves that lower
    /// the energy by nothing, after one that lowered it. Fails where a move raises the energy or the count is off.
    #[test]
    fn moves_never_raise_the_energy_and_stop_after_n_states_unchanged(
        (problem, start) in problems_and_starts(true),
        patch in any::<bool>(),
    ) {
        let k = problem.n_states;
        let (_, steps) = expanded(&problem, &start, 1000, patch);
        let energies: Vec<f64> = (0..=steps)
            .map(|s| problem.view().metric_energy(&expanded(&problem, &start, s, patch).0))
            .collect();
        for s in 1..=steps {
            prop_assert!(energies[s] <= energies[s - 1], "move {} raises {} to {}", s, energies[s - 1], energies[s]);
        }
        prop_assert!(steps >= k);
        for s in steps - k..steps {
            prop_assert_eq!(energies[s + 1], energies[s]);
        }
        if steps > k {
            prop_assert!(energies[steps - k] < energies[steps - k - 1], "the last decrease is not move {}", steps - k);
        }
    }

    /// Fails where `max_steps` is exceeded or a run cut short is not the prefix of a longer one.
    #[test]
    fn max_steps_cuts_the_moves_short(
        (problem, start) in problems_and_starts(true),
        cap in 0usize..6,
        patch in any::<bool>(),
    ) {
        let (full, steps) = expanded(&problem, &start, 1000, patch);
        let (cut, taken) = expanded(&problem, &start, cap, patch);
        prop_assert_eq!(taken, cap.min(steps));
        if cap >= steps {
            prop_assert_eq!(cut, full);
        }
        if cap == 0 {
            prop_assert_eq!(expanded(&problem, &start, 0, patch).0, start);
        }
    }

    /// Fails where expanding from a minimum leaves it.
    #[test]
    fn a_minimum_is_kept((problem, _) in problems_and_starts(true), patch in any::<bool>()) {
        let all = problem.enumerate();
        let (best, energy) = all.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().clone();
        let (labels, _) = expanded(&problem, &best, 1000, patch);
        prop_assert_eq!(problem.view().metric_energy(&labels), energy);
    }

    /// Fails where the Potts form is not the metric form less the summed coupling.
    #[test]
    fn the_potts_energy_is_the_metric_less_the_coupling((problem, start) in problems_and_starts(false)) {
        let total: f64 = problem.coupling.iter().sum();
        let (metric, potts) = (problem.view().metric_energy(&start), problem.view().potts_energy(&start));
        prop_assert!((metric - total - potts).abs() <= RTOL * metric.abs().max(1.0));
    }

    /// Integer energies, so each move's comparison is exact: fails where a node's or a pair's index decides a cut
    /// that the problem does not.
    #[test]
    fn renaming_nodes_and_reordering_pairs_renames_the_result(
        (problem, start) in problems_and_starts(true),
        seed in any::<u64>(),
        patch in any::<bool>(),
    ) {
        let n = problem.n_nodes();
        let mut rng = common::SplitMix::new(seed);
        let mut order: Vec<usize> = (0..n).collect();
        let mut pairs: Vec<usize> = (0..problem.first.len()).collect();
        for v in [&mut order, &mut pairs] {
            for i in (1..v.len()).rev() {
                v.swap(i, rng.below(i + 1));
            }
        }
        let renamed = problem.renamed(&order, &pairs);
        let mut renamed_start = vec![0; n];
        for i in 0..n {
            renamed_start[order[i]] = start[i];
        }
        let (labels, steps) = expanded(&problem, &start, 1000, patch);
        let (renamed_labels, renamed_steps) = expanded(&renamed, &renamed_start, 1000, patch);
        prop_assert_eq!(steps, renamed_steps);
        for i in 0..n {
            prop_assert_eq!(labels[i], renamed_labels[order[i]]);
        }
    }
}

#[test]
fn one_label_takes_one_move() {
    let problem = planted_lattice(4, 5, 1, 1.0, 1.1, 3);
    let (labels, steps) = expanded(&problem, &[0; 20], 1000, false);
    assert_eq!((labels, steps), (vec![0; 20], 1));
}

#[test]
fn without_pairs_each_node_takes_its_unary_argmin() {
    let mut problem = planted_lattice(6, 7, 5, 0.5, 1.1, 11);
    problem.first.clear();
    problem.second.clear();
    problem.coupling.clear();
    let (labels, _) = expanded(&problem, &[0; 42], 1000, true);
    assert_eq!(labels, argmin_labels(&problem));
}

/// The planted lattice at the PR's fixture size: the result is a local minimum under every alpha move, each checked
/// by one more expansion from it. Fails where a large problem's move is not optimal.
#[test]
fn a_planted_lattice_s_result_is_a_fixed_point_of_every_move() {
    let problem = planted_lattice(30, 40, 4, 1.0, 1.1, 3);
    for patch in [true, false] {
        let (labels, steps) = expanded(&problem, &argmin_labels(&problem), 1000, patch);
        assert!(steps < 1000);
        let energy = problem.view().metric_energy(&labels);
        let (again, more) = expanded(&problem, &labels, 1000, patch);
        assert_eq!(more, problem.n_states);
        assert_eq!(problem.view().metric_energy(&again), energy);
    }
}

/// Two nodes at labels 0 and 1, a pair of coupling 1, label 2 costing each 0.4 more: the alpha-2 move to (2, 2) lowers
/// the metric energy from 1 to 0.8, the minimum by enumeration.
fn two_nodes() -> Owned {
    Owned {
        n_states: 3,
        unary: vec![0.0, 9.0, 0.4, 9.0, 0.0, 0.4],
        first: vec![0],
        second: vec![1],
        coupling: vec![1.0],
    }
}

/// T- #195's moves: fails where the pair's disagreement off alpha is left out of the move.
#[test]
fn a_pair_labelled_apart_off_alpha_counts_in_the_move() {
    let problem = two_nodes();
    let best = problem
        .enumerate()
        .into_iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    assert_eq!(best, (vec![2, 2], 0.8));
    assert_eq!(expanded(&problem, &[0, 1], 1000, true).0, vec![2, 2]);
}

/// The defect T- #195 states, pinned: OpenGM's auxiliary node (`addPairwise(a, x, 0, J, 0, 0)`, `addUnary(x, 0, J)`)
/// costs 0 at every configuration, so OpenGM's moves leave (0, 1). Fails where `opengm_patch` off stops being
/// OpenGM's (its oracle, `opengm_oracle.rs`, then fails too).
#[test]
fn opengm_s_move_drops_a_pair_labelled_apart_off_alpha() {
    assert_eq!(expanded(&two_nodes(), &[0, 1], 1000, false).0, vec![0, 1]);
}
