//! The C entry points `cnamaste.expansion` calls through ctypes: `cnamaste_expansion_solve` is `expand` and
//! `potts_energy` bitwise, a null start every node at 0; `cnamaste_expansion_min_cut` is the enumerated minimum with
//! its smallest sink side; bad counts and indices return 1 with the message, truncated to the caller's buffer.

mod common;

use cnamaste_expansion::expansion::expand_with;
use cnamaste_expansion::trws::trws;
use cnamaste_expansion::{cnamaste_expansion_min_cut, cnamaste_expansion_solve, cnamaste_trws_solve};
use common::{planted_lattice, Cut, Owned};

struct Solved {
    status: i32,
    labels: Vec<i64>,
    value: f64,
    steps: i64,
    error: String,
}

fn solve(
    problem: &Owned,
    first: &[i64],
    start: Option<&[i64]>,
    max_steps: i64,
    patch: bool,
    error_size: usize,
) -> Solved {
    let n = problem.n_nodes();
    let second: Vec<i64> = problem.second.iter().map(|&b| b as i64).collect();
    let mut labels = vec![-1i64; n];
    let (mut value, mut steps, mut seconds) = (f64::NAN, -1i64, -1.0f64);
    let mut error = vec![0xAAu8; error_size.max(1)];
    let status = unsafe {
        cnamaste_expansion_solve(
            n as i64,
            problem.n_states as i64,
            problem.unary.as_ptr(),
            first.len() as i64,
            first.as_ptr(),
            second.as_ptr(),
            problem.coupling.as_ptr(),
            start.map_or(std::ptr::null(), |s| s.as_ptr()),
            max_steps,
            i64::from(patch),
            labels.as_mut_ptr(),
            &mut value,
            &mut steps,
            &mut seconds,
            error.as_mut_ptr(),
            error_size as i64,
        )
    };
    let end = error.iter().position(|&b| b == 0).unwrap_or(0);
    Solved {
        status,
        labels,
        value,
        steps,
        error: String::from_utf8_lossy(&error[..end]).into(),
    }
}

fn first_of(problem: &Owned) -> Vec<i64> {
    problem.first.iter().map(|&a| a as i64).collect()
}

/// OpenGM's moves and T- #195's: fails where the entry point's labels, energy or moves are not `expand_with`'s and
/// `potts_energy`'s, or `opengm_patch` is not passed through.
#[test]
fn solve_is_expand_bitwise() {
    let problem = planted_lattice(12, 15, 5, 1.0, 1.1, 5);
    let start: Vec<i64> = common::argmin_labels(&problem).iter().map(|&x| x as i64).collect();
    for patch in [true, false] {
        let solved = solve(&problem, &first_of(&problem), Some(&start), 1000, patch, 256);
        let mut labels: Vec<usize> = start.iter().map(|&x| x as usize).collect();
        let steps = expand_with(&problem.view(), &mut labels, 1000, patch);
        assert_eq!(solved.status, 0, "{}", solved.error);
        assert_eq!(solved.labels, labels.iter().map(|&x| x as i64).collect::<Vec<_>>());
        assert_eq!(solved.steps, steps as i64);
        assert_eq!(solved.value.to_bits(), problem.view().potts_energy(&labels).to_bits());
    }
}

/// Fails where a null start is not every node at 0.
#[test]
fn a_null_start_is_every_node_at_zero() {
    let problem = planted_lattice(8, 9, 3, 1.0, 1.1, 2);
    let zeros = vec![0i64; problem.n_nodes()];
    let null = solve(&problem, &first_of(&problem), None, 1000, false, 256);
    let given = solve(&problem, &first_of(&problem), Some(&zeros), 1000, false, 256);
    assert_eq!(
        (null.labels, null.steps, null.value.to_bits()),
        (given.labels, given.steps, given.value.to_bits())
    );
}

/// Fails where an index outside its range, a negative one or a bad count reaches the solver.
#[test]
fn bad_indices_and_counts_return_one_with_the_message() {
    let problem = planted_lattice(3, 3, 3, 1.0, 1.1, 1);
    let mut first = first_of(&problem);
    first[2] = 9;
    let out = solve(&problem, &first, None, 1000, false, 256);
    assert_eq!((out.status, out.error.as_str()), (1, "first 9 is out of range"));
    first[2] = -1;
    assert_eq!(
        solve(&problem, &first, None, 1000, false, 256).error,
        "first -1 is out of range"
    );
    let start = vec![0, 0, 0, 0, 3, 0, 0, 0, 0];
    let out = solve(&problem, &first_of(&problem), Some(&start), 1000, false, 256);
    assert_eq!((out.status, out.error.as_str()), (1, "start label 3 is out of range"));
    let out = solve(&problem, &first_of(&problem), None, -1, false, 256);
    assert_eq!((out.status, out.error.as_str()), (1, "counts must be positive"));
    assert_eq!(out.labels, vec![-1; 9], "labels written on an error");
}

/// Fails where the message overruns the caller's buffer or is not terminated.
#[test]
fn the_message_is_truncated_to_the_buffer() {
    let problem = planted_lattice(3, 3, 3, 1.0, 1.1, 1);
    let mut first = first_of(&problem);
    first[0] = 99;
    assert_eq!(solve(&problem, &first, None, 1000, false, 6).error, "first");
    let out = solve(&problem, &first, None, 1000, false, 0);
    assert_eq!((out.status, out.error.as_str()), (1, ""));
}

fn min_cut(n: usize, tail: &[i64], head: &[i64], capacity: &[f64]) -> (i32, Vec<i64>, f64) {
    let mut side = vec![-1i64; n];
    let mut value = f64::NAN;
    let mut error = [0u8; 128];
    let status = unsafe {
        cnamaste_expansion_min_cut(
            n as i64,
            tail.len() as i64,
            tail.as_ptr(),
            head.as_ptr(),
            capacity.as_ptr(),
            side.as_mut_ptr(),
            &mut value,
            error.as_mut_ptr(),
            error.len() as i64,
        )
    };
    (status, side, value)
}

/// Node 0 the source, 1 the sink, an edge between them included: fails where the entry point's cut is not the
/// enumerated minimum's smallest sink side, or its value leaves out an edge.
#[test]
fn min_cut_is_the_enumerated_minimum() {
    let mut rng = common::SplitMix::new(23);
    for _ in 0..200 {
        let n = 3 + rng.below(8);
        let (mut tail, mut head, mut capacity) = (vec![0i64], vec![1i64], vec![rng.below(3) as f64]);
        for _ in 0..3 * n {
            let (u, v) = (rng.below(n), rng.below(n));
            if u != v && u != 1 && v != 0 {
                tail.push(u as i64);
                head.push(v as i64);
                capacity.push(rng.below(4) as f64);
            }
        }
        let (status, side, value) = min_cut(n, &tail, &head, &capacity);
        assert_eq!(status, 0);
        // NB the same graph as terminal capacities on nodes 2.., the source-sink edge always cut.
        let inner = n - 2;
        let mut cut = Cut {
            source: vec![0.0; inner],
            sink: vec![0.0; inner],
            tails: vec![],
            heads: vec![],
            capacities: vec![],
        };
        let mut direct = 0.0;
        for e in 0..tail.len() {
            let (u, v, c) = (tail[e] as usize, head[e] as usize, capacity[e]);
            match (u, v) {
                (0, 1) => direct += c,
                (0, v) => cut.source[v - 2] += c,
                (u, 1) => cut.sink[u - 2] += c,
                (u, v) => {
                    cut.tails.push((u - 2) as u32);
                    cut.heads.push((v - 2) as u32);
                    cut.capacities.push(c);
                }
            }
        }
        let (best, smallest) = cut.brute_force(0.0);
        assert_eq!(value, best + direct);
        assert_eq!(&side[..2], &[0, 1]);
        assert_eq!(side[2..].iter().map(|&s| s == 1).collect::<Vec<_>>(), smallest);
    }
}

#[test]
fn min_cut_refuses_a_graph_without_both_terminals() {
    assert_eq!(min_cut(1, &[], &[], &[]).0, 1);
    assert_eq!(min_cut(3, &[0], &[5], &[1.0]).0, 1);
}

struct TrwsSolved {
    status: i32,
    labels: Vec<i64>,
    value: f64,
    bound: f64,
    trace: Vec<f64>,
    error: String,
}

fn trws_solve(problem: &Owned, first: &[i64], second: &[i64], max_iterations: i64) -> TrwsSolved {
    let n = problem.n_nodes();
    let mut labels = vec![-1i64; n];
    let mut trace = vec![f64::NAN; max_iterations.max(0) as usize];
    let (mut value, mut bound, mut taken, mut seconds) = (f64::NAN, f64::NAN, -1i64, -1.0f64);
    let mut error = [0u8; 256];
    let status = unsafe {
        cnamaste_trws_solve(
            n as i64,
            problem.n_states as i64,
            problem.unary.as_ptr(),
            first.len() as i64,
            first.as_ptr(),
            second.as_ptr(),
            problem.coupling.as_ptr(),
            max_iterations,
            1e-9,
            labels.as_mut_ptr(),
            &mut value,
            &mut bound,
            trace.as_mut_ptr(),
            &mut taken,
            &mut seconds,
            error.as_mut_ptr(),
            error.len() as i64,
        )
    };
    let end = error.iter().position(|&b| b == 0).unwrap_or(0);
    trace.truncate(taken.max(0) as usize);
    TrwsSolved {
        status,
        labels,
        value,
        bound,
        trace,
        error: String::from_utf8_lossy(&error[..end]).into(),
    }
}

fn second_of(problem: &Owned) -> Vec<i64> {
    problem.second.iter().map(|&b| b as i64).collect()
}

/// Fails where `cnamaste_trws_solve`'s labels, bound or trace are not `trws`'s, or its value not `potts_energy`'s.
#[test]
fn trws_solve_is_trws_bitwise() {
    let problem = planted_lattice(12, 15, 5, 1.0, 1.1, 5);
    let found = trws(&problem.view(), 1000, 1e-9);
    let solved = trws_solve(&problem, &first_of(&problem), &second_of(&problem), 1000);
    assert_eq!(solved.status, 0, "{}", solved.error);
    assert_eq!(
        solved.labels,
        found.labels.iter().map(|&x| x as i64).collect::<Vec<_>>()
    );
    assert_eq!(
        solved.value.to_bits(),
        problem.view().potts_energy(&found.labels).to_bits()
    );
    assert_eq!(solved.bound.to_bits(), found.bound.to_bits());
    assert_eq!(solved.trace, found.trace);
}

/// The trace holds `max_iterations` entries at most, as `cnamaste_opengm_solve` clamps OpenGM's. Fails where the
/// entry point writes past the caller's array.
#[test]
fn the_trws_trace_is_clamped_to_max_iterations() {
    let problem = planted_lattice(12, 15, 5, 1.0, 1.1, 5);
    let solved = trws_solve(&problem, &first_of(&problem), &second_of(&problem), 2);
    let found = trws(&problem.view(), 2, 1e-9);
    assert_eq!(solved.trace, found.trace[..2].to_vec());
}

/// Fails where a bad count, an index out of range, a self-loop or no iteration reaches TRW-S.
#[test]
fn trws_refuses_bad_input_with_the_message() {
    let problem = planted_lattice(3, 3, 3, 1.0, 1.1, 1);
    let (first, second) = (first_of(&problem), second_of(&problem));
    let out = trws_solve(&problem, &first, &second, 0);
    assert_eq!(
        (out.status, out.error.as_str()),
        (1, "TRW-S needs max_iterations > 0, as OpenGM's TRWSi")
    );
    let mut bad = first.clone();
    bad[1] = 9;
    assert_eq!(trws_solve(&problem, &bad, &second, 10).error, "first 9 is out of range");
    let mut looped = second.clone();
    looped[0] = first[0];
    assert_eq!(
        trws_solve(&problem, &first, &looped, 10).error,
        "a pair joins a node to itself"
    );
    assert_eq!(trws_solve(&problem, &first, &looped, 10).labels, vec![-1; 9]);
}
