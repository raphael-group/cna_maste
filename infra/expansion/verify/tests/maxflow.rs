//! `maxflow::Graph` against enumeration of every s-t cut: the cut's capacity is the minimum and its sink side the
//! smallest minimum cut's, Boost Graph's Boykov-Kolmogorov's, on random graphs of real and of integer capacities, and
//! at edge cases. The flow's own invariants (conservation, residuals) are `maxflow.rs`'s unit tests.

mod common;

use cnamaste_expansion::maxflow::Graph;
use common::Cut;
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

/// Relative tolerance on a cut's capacity: sums of at most a few dozen terms.
const RTOL: f64 = 1e-12;

fn solve(cut: &Cut) -> (f64, Vec<bool>) {
    let mut graph = Graph::default();
    graph.build(&cut.source, &cut.sink, &cut.tails, &cut.heads, &cut.capacities);
    let mut side = Vec::new();
    graph.solve(&mut side);
    (graph.flow, side)
}

/// The cut's capacity and the flow are the enumerated minimum to `RTOL`; where `ties` is given, the sink side is the
/// smallest minimum one, cuts within `ties` of the minimum counted as minimum (0 where sums are exact).
fn assert_minimum(cut: &Cut, ties: Option<f64>) {
    let (flow, side) = solve(cut);
    let (best, smallest) = cut.brute_force(ties.unwrap_or(RTOL));
    let scale = best.abs().max(1.0);
    assert!(
        (cut.value(&side) - best).abs() <= RTOL * scale,
        "cut {} against minimum {best}",
        cut.value(&side)
    );
    assert!(
        (flow - best).abs() <= RTOL * scale,
        "flow {flow} against minimum {best}"
    );
    if ties.is_some() {
        assert_eq!(side, smallest, "not the smallest minimum sink side");
    }
}

/// A graph of 2 to 10 nodes, up to 30 edges without self-loops, capacities exponential or integers 0 to 3.
fn graphs(integer: bool) -> impl Strategy<Value = Cut> {
    (2usize..=10).prop_flat_map(move |n| {
        let capacity = move || -> BoxedStrategy<f64> {
            if integer {
                (0u32..4).prop_map(f64::from).boxed()
            } else {
                (1e-3f64..1.0).prop_map(|u| -u.ln()).boxed()
            }
        };
        // NB the head an offset 1 to n - 1 past the tail: never a self-loop.
        let edge = (0..n as u32, 1..n as u32, capacity()).prop_map(move |(u, d, c)| (u, (u + d) % n as u32, c));
        (
            proptest::collection::vec(capacity(), n),
            proptest::collection::vec(capacity(), n),
            proptest::collection::vec(edge, 0..=30),
        )
            .prop_map(|(source, sink, edges)| Cut {
                source,
                sink,
                tails: edges.iter().map(|e| e.0).collect(),
                heads: edges.iter().map(|e| e.1).collect(),
                capacities: edges.iter().map(|e| e.2).collect(),
            })
    })
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 512,
        failure_persistence: Some(Box::new(FileFailurePersistence::WithSource("regressions"))),
        ..ProptestConfig::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// Real capacities, ties counted to 1e-12: fails where the flow stops short of the maximum or the cut is misread.
    #[test]
    fn the_cut_is_the_enumerated_minimum_on_real_capacities(cut in graphs(false)) {
        assert_minimum(&cut, Some(RTOL));
    }

    /// Integer capacities tie many cuts, summed exactly: fails where a node either side could take is put on the
    /// sink's.
    #[test]
    fn the_cut_is_the_smallest_minimum_on_tied_capacities(cut in graphs(true)) {
        assert_minimum(&cut, Some(0.0));
    }
}

#[test]
fn a_graph_without_edges_cuts_each_node_at_its_smaller_terminal() {
    // NB node 2 is tied, node 3 has no capacity: both on the source's side.
    let cut = Cut {
        source: vec![2.0, 1.0, 1.5, 0.0],
        sink: vec![1.0, 2.0, 1.5, 0.0],
        tails: vec![],
        heads: vec![],
        capacities: vec![],
    };
    let (flow, side) = solve(&cut);
    assert_eq!(side, vec![false, true, false, false]);
    assert_eq!(flow, 1.0 + 1.0 + 1.5);
    assert_minimum(&cut, Some(0.0));
}

#[test]
fn parallel_edges_cut_as_one_of_their_summed_capacity() {
    let parallel = Cut {
        source: vec![3.0, 0.0, 0.0],
        sink: vec![0.0, 0.0, 3.0],
        tails: vec![0, 0, 1, 1],
        heads: vec![1, 1, 2, 2],
        capacities: vec![0.5, 0.75, 1.0, 1.5],
    };
    let merged = Cut {
        tails: vec![0, 1],
        heads: vec![1, 2],
        capacities: vec![1.25, 2.5],
        ..parallel.clone()
    };
    assert_eq!(solve(&parallel), solve(&merged));
    assert_minimum(&parallel, Some(0.0));
}

#[test]
fn zero_capacities_and_isolated_nodes_stay_on_the_source_side() {
    let cut = Cut {
        source: vec![0.0, 1.0, 0.0, 0.0],
        sink: vec![0.0, 0.0, 1.0, 0.0],
        tails: vec![0, 1, 2],
        heads: vec![1, 2, 0],
        capacities: vec![0.0, 0.0, 0.0],
    };
    let (flow, side) = solve(&cut);
    assert_eq!(side, vec![false, false, true, false]);
    assert_eq!(flow, 0.0);
    assert_minimum(&cut, Some(0.0));
}

/// Capacities 1e-12 to 1e12: the capacity and the flow are the minimum to 1e-12 relative. The side is not compared:
/// a 1e-12 capacity against a 1e12 cut is within any tie window the enumeration could use.
#[test]
fn capacities_over_twenty_four_decades_cut_at_the_minimum() {
    let mut rng = common::SplitMix::new(7);
    for _ in 0..200 {
        let n = 2 + rng.below(8);
        let mut capacity = || 10f64.powf(-12.0 + 24.0 * rng.uniform());
        let source = (0..n).map(|_| capacity()).collect();
        let sink = (0..n).map(|_| capacity()).collect();
        let (mut tails, mut heads, mut capacities) = (vec![], vec![], vec![]);
        for _ in 0..3 * n {
            let (u, v) = (rng.below(n), rng.below(n));
            if u != v {
                tails.push(u as u32);
                heads.push(v as u32);
                capacities.push(10f64.powf(-12.0 + 24.0 * rng.uniform()));
            }
        }
        assert_minimum(
            &Cut {
                source,
                sink,
                tails,
                heads,
                capacities,
            },
            None,
        );
    }
}
