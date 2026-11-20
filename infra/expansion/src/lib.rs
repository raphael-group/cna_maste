//! cnamaste's C entry points to its alpha expansion (`expansion`) and the Boykov-Kolmogorov min cut (`maxflow`) each
//! move makes, called by `cnamaste.expansion` through ctypes. Their signatures follow `cnamaste_opengm.cxx`'s, so the
//! two libraries are called, and compared, alike.

// NB index loops over parallel arrays read as the algorithm states them.
#![allow(clippy::needless_range_loop)]

pub mod expansion;
pub mod maxflow;
pub mod trws;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use expansion::{expand_with, Problem};
use maxflow::Graph;

/// Write `message` into the caller's `error` buffer of `size` bytes, terminated, truncated where it must be.
unsafe fn write_error(error: *mut u8, size: i64, message: &str) {
    if error.is_null() || size <= 0 {
        return;
    }
    let bytes = message.as_bytes();
    let length = bytes.len().min(size as usize - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), error, length);
    *error.add(length) = 0;
}

fn run(body: impl FnOnce() -> Result<(), String>, error: *mut u8, error_size: i64) -> i32 {
    let outcome = catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|_| Err("cnamaste_expansion panicked".into()));
    match outcome {
        Ok(()) => 0,
        Err(message) => {
            unsafe { write_error(error, error_size, &message) };
            1
        }
    }
}

fn indices(values: &[i64], bound: usize, name: &str) -> Result<Vec<usize>, String> {
    values
        .iter()
        .map(|&v| {
            if v >= 0 && (v as usize) < bound {
                Ok(v as usize)
            } else {
                Err(format!("{name} {v} is out of range"))
            }
        })
        .collect()
}

/// Expand the Potts problem from `start` (null: every node at 0), at most `max_steps` moves, T- #195's moves where
/// `opengm_patch` is nonzero (else OpenGM's), and write the labelling, its energy in the Potts form, the moves
/// made and the seconds of the expansion alone. Returns 0, or 1 with `error` holding the message.
///
/// # Safety
/// Every pointer holds the length its counts state; `start` may be null.
#[no_mangle]
pub unsafe extern "C" fn cnamaste_expansion_solve(
    n_nodes: i64,
    n_states: i64,
    unary: *const f64,
    n_edges: i64,
    first: *const i64,
    second: *const i64,
    coupling: *const f64,
    start: *const i64,
    max_steps: i64,
    opengm_patch: i64,
    labels_out: *mut i64,
    value: *mut f64,
    steps: *mut i64,
    seconds: *mut f64,
    error: *mut u8,
    error_size: i64,
) -> i32 {
    run(
        || {
            if n_nodes <= 0 || n_states <= 0 || n_edges < 0 || max_steps < 0 {
                return Err("counts must be positive".into());
            }
            let (n, k, m) = (n_nodes as usize, n_states as usize, n_edges as usize);
            let unary = std::slice::from_raw_parts(unary, n * k);
            let first = indices(std::slice::from_raw_parts(first, m), n, "first")?;
            let second = indices(std::slice::from_raw_parts(second, m), n, "second")?;
            let coupling = std::slice::from_raw_parts(coupling, m);
            let mut labels = if start.is_null() {
                vec![0; n]
            } else {
                indices(std::slice::from_raw_parts(start, n), k, "start label")?
            };
            let problem = Problem {
                n_states: k,
                unary,
                first: &first,
                second: &second,
                coupling,
            };
            let clock = Instant::now();
            let taken = expand_with(&problem, &mut labels, max_steps as usize, opengm_patch != 0);
            *seconds = clock.elapsed().as_secs_f64();
            *value = problem.potts_energy(&labels);
            *steps = taken as i64;
            let out = std::slice::from_raw_parts_mut(labels_out, n);
            for (o, &x) in out.iter_mut().zip(labels.iter()) {
                *o = x as i64;
            }
            Ok(())
        },
        error,
        error_size,
    )
}

/// TRW-S on the Potts problem (`trws::trws`, T- #198), at most `max_iterations` passes after the forward one,
/// `tolerance` relative to the bound, and write the best labelling, its energy in the Potts form, the dual bound, the
/// bound after the forward pass and each pass (at most `max_iterations` entries), the entries written, and the
/// seconds of TRW-S alone, as `cnamaste_opengm_solve` writes OpenGM's. Returns 0, or 1 with `error` holding the
/// message.
///
/// # Safety
/// Every pointer holds the length its counts state; `trace_out` holds `max_iterations` doubles.
#[no_mangle]
pub unsafe extern "C" fn cnamaste_trws_solve(
    n_nodes: i64,
    n_states: i64,
    unary: *const f64,
    n_edges: i64,
    first: *const i64,
    second: *const i64,
    coupling: *const f64,
    max_iterations: i64,
    tolerance: f64,
    labels_out: *mut i64,
    value: *mut f64,
    bound: *mut f64,
    trace_out: *mut f64,
    taken: *mut i64,
    seconds: *mut f64,
    error: *mut u8,
    error_size: i64,
) -> i32 {
    run(
        || {
            if n_nodes <= 0 || n_states <= 0 || n_edges < 0 {
                return Err("counts must be positive".into());
            }
            if max_iterations <= 0 {
                return Err("TRW-S needs max_iterations > 0, as OpenGM's TRWSi".into());
            }
            let (n, k, m) = (n_nodes as usize, n_states as usize, n_edges as usize);
            let unary = std::slice::from_raw_parts(unary, n * k);
            let first = indices(std::slice::from_raw_parts(first, m), n, "first")?;
            let second = indices(std::slice::from_raw_parts(second, m), n, "second")?;
            if first.iter().zip(&second).any(|(a, b)| a == b) {
                return Err("a pair joins a node to itself".into());
            }
            let coupling = std::slice::from_raw_parts(coupling, m);
            let problem = Problem {
                n_states: k,
                unary,
                first: &first,
                second: &second,
                coupling,
            };
            let clock = Instant::now();
            let found = trws::trws(&problem, max_iterations as usize, tolerance);
            *seconds = clock.elapsed().as_secs_f64();
            *value = problem.potts_energy(&found.labels);
            *bound = found.bound;
            let written = found.trace.len().min(max_iterations as usize);
            std::slice::from_raw_parts_mut(trace_out, written).copy_from_slice(&found.trace[..written]);
            *taken = written as i64;
            let out = std::slice::from_raw_parts_mut(labels_out, n);
            for (o, &x) in out.iter_mut().zip(found.labels.iter()) {
                *o = x as i64;
            }
            Ok(())
        },
        error,
        error_size,
    )
}

/// One min s-t cut on a directed graph, node 0 the source and node 1 the sink, as `cnamaste_min_cut` poses it:
/// `sink_side` 1 where a node reaches the sink in the residual graph (the sink itself 1, the source 0), and `value`
/// the capacity crossing from the source's side to the sink's. Returns 0, or 1 with `error` holding the message.
///
/// # Safety
/// Every pointer holds the length its counts state.
#[no_mangle]
pub unsafe extern "C" fn cnamaste_expansion_min_cut(
    n_nodes: i64,
    n_edges: i64,
    tail: *const i64,
    head: *const i64,
    capacity: *const f64,
    sink_side: *mut i64,
    value: *mut f64,
    error: *mut u8,
    error_size: i64,
) -> i32 {
    run(
        || {
            if n_nodes < 2 || n_edges < 0 {
                return Err("a cut needs the source, the sink and no negative count".into());
            }
            let (n, m) = (n_nodes as usize, n_edges as usize);
            let tail = indices(std::slice::from_raw_parts(tail, m), n, "tail")?;
            let head = indices(std::slice::from_raw_parts(head, m), n, "head")?;
            let capacity = std::slice::from_raw_parts(capacity, m);
            // NB node v >= 2 is graph node v - 2; arcs from the source and to the sink are its terminal capacities.
            let inner = n - 2;
            let (mut source, mut sink) = (vec![0.0; inner], vec![0.0; inner]);
            let (mut tails, mut heads, mut capacities) = (Vec::new(), Vec::new(), Vec::new());
            for e in 0..m {
                let (u, v, c) = (tail[e], head[e], capacity[e]);
                if u == 0 && v >= 2 {
                    source[v - 2] += c;
                } else if v == 1 && u >= 2 {
                    sink[u - 2] += c;
                } else if u >= 2 && v >= 2 {
                    tails.push((u - 2) as u32);
                    heads.push((v - 2) as u32);
                    capacities.push(c);
                }
            }
            let mut graph = Graph::default();
            graph.build(&source, &sink, &tails, &heads, &capacities);
            let mut side = Vec::new();
            graph.solve(&mut side);
            let out = std::slice::from_raw_parts_mut(sink_side, n);
            out[0] = 0;
            out[1] = 1;
            for v in 2..n {
                out[v] = side[v - 2] as i64;
            }
            let mut crossing = 0.0;
            for e in 0..m {
                if out[tail[e]] == 0 && out[head[e]] == 1 {
                    crossing += capacity[e];
                }
            }
            *value = crossing;
            Ok(())
        },
        error,
        error_size,
    )
}

#[cfg(test)]
mod tests {
    use super::run;

    /// Fails where a panic unwinds into the caller (ctypes) rather than returning 1 with the message.
    #[test]
    fn a_panic_returns_one_with_the_message() {
        let mut error = [0xAAu8; 64];
        let status = run(
            || panic!("an index past the end"),
            error.as_mut_ptr(),
            error.len() as i64,
        );
        let end = error.iter().position(|&b| b == 0).unwrap();
        assert_eq!((status, &error[..end]), (1, "cnamaste_expansion panicked".as_bytes()));
    }

    #[test]
    fn success_returns_zero_and_leaves_the_buffer() {
        let mut error = [0xAAu8; 8];
        assert_eq!(run(|| Ok(()), error.as_mut_ptr(), error.len() as i64), 0);
        assert_eq!(error, [0xAAu8; 8]);
    }
}
