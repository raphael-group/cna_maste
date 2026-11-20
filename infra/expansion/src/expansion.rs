//! Alpha expansion (Boykov, Veksler and Zabih 2001) of the Potts energy `sum unary[i, x_i] - sum J [x_a == x_b]`,
//! each move one Boykov-Kolmogorov min cut (`maxflow`).
//!
//! It is OpenGM's `AlphaExpansion` over `GraphCut<MinSTCutBoost<KOLMOGOROV>>`, as `cnamaste_opengm.cxx` runs it:
//! the metric form (0 on agreement, J off it), labels visited 0, 1, ..., every move applied, and a stop after
//! `n_states` moves in a row lower the energy by nothing, or after `max_steps` moves. Each move's capacities are summed
//! in OpenGM's order, unaries then pairs, so the min cut is the problem OpenGM poses. A node already at alpha, which
//! OpenGM ties to the source by an infinite capacity, is left out of the graph: it is on the source's side either way.
//!
//! OpenGM's auxiliary node for a pair labelled apart, neither end alpha, costs 0 at every configuration, so that
//! pair drops out of the move (T- #195). `expand_with(.., opengm_patch = true)` takes the construction that keeps
//! it: J from the auxiliary node to each end, J to the source, the pair's move energy 0 at (alpha, alpha) and J
//! otherwise. `expand` is OpenGM's.

use crate::maxflow::Graph;

/// A Potts problem in the pairwise form `cnamaste.potts.potts_problem` poses.
pub struct Problem<'a> {
    pub n_states: usize,
    pub unary: &'a [f64],
    pub first: &'a [usize],
    pub second: &'a [usize],
    pub coupling: &'a [f64],
}

impl Problem<'_> {
    pub fn n_nodes(&self) -> usize {
        self.unary.len() / self.n_states
    }

    /// The energy in the metric form, as OpenGM's `evaluate` sums it: nodes, then pairs, in order.
    pub fn metric_energy(&self, labels: &[usize]) -> f64 {
        let mut energy = 0.0;
        for (i, &x) in labels.iter().enumerate() {
            energy += self.unary[i * self.n_states + x];
        }
        for e in 0..self.first.len() {
            if labels[self.first[e]] != labels[self.second[e]] {
                energy += self.coupling[e];
            }
        }
        energy
    }

    /// The energy in the Potts form, as `cnamaste_opengm_solve` reports it: nodes, then agreeing pairs, in order.
    pub fn potts_energy(&self, labels: &[usize]) -> f64 {
        let mut energy = 0.0;
        for (i, &x) in labels.iter().enumerate() {
            energy += self.unary[i * self.n_states + x];
        }
        for e in 0..self.first.len() {
            if labels[self.first[e]] == labels[self.second[e]] {
                energy -= self.coupling[e];
            }
        }
        energy
    }
}

/// The working arrays of one move, kept across moves.
#[derive(Default)]
struct Move {
    node: Vec<u32>,
    source: Vec<f64>,
    sink: Vec<f64>,
    tails: Vec<u32>,
    heads: Vec<u32>,
    capacities: Vec<f64>,
    sink_side: Vec<bool>,
    graph: Graph,
}

const OUT: u32 = u32::MAX;

impl Move {
    /// Expand `alpha` from `labels`, in place; `opengm_patch` T- #195's auxiliary node, else OpenGM's.
    fn expand(&mut self, problem: &Problem, alpha: usize, labels: &mut [usize], opengm_patch: bool) {
        let n = problem.n_nodes();
        let k = problem.n_states;
        self.node.clear();
        self.node.resize(n, OUT);
        let mut free = 0u32;
        for i in 0..n {
            if labels[i] != alpha {
                self.node[i] = free;
                free += 1;
            }
        }
        if free == 0 {
            return;
        }
        self.source.clear();
        self.source.resize(free as usize, 0.0);
        self.sink.clear();
        self.sink.resize(free as usize, 0.0);
        self.tails.clear();
        self.heads.clear();
        self.capacities.clear();
        // NB a unary of (v0, v1), alpha then the current label: the source's capacity v1 - v0, else the sink's.
        for i in 0..n {
            if labels[i] != alpha {
                let (v0, v1) = (problem.unary[i * k + alpha], problem.unary[i * k + labels[i]]);
                let g = self.node[i] as usize;
                if v0 <= v1 {
                    self.source[g] += v1 - v0;
                } else {
                    self.sink[g] += v0 - v1;
                }
            }
        }
        for e in 0..problem.first.len() {
            let (a, b, j) = (problem.first[e], problem.second[e], problem.coupling[e]);
            let (xa, xb) = (labels[a], labels[b]);
            if xa == alpha && xb == alpha {
                continue;
            } else if xa == alpha {
                add_unary(&mut self.source, &mut self.sink, self.node[b] as usize, 0.0, j);
            } else if xb == alpha {
                add_unary(&mut self.source, &mut self.sink, self.node[a] as usize, 0.0, j);
            } else if xa == xb {
                // NB OpenGM's pairwise (A, B, C, D) = (0, J, J, 0): C - A to the source at a, C - D to the sink at b,
                //    and B + C - A - D from a to b.
                let (ga, gb) = (self.node[a] as usize, self.node[b] as usize);
                if j > 0.0 {
                    self.source[ga] += j;
                } else if j < 0.0 {
                    self.sink[ga] -= j;
                }
                if j > 0.0 {
                    self.sink[gb] += j;
                } else if j < 0.0 {
                    self.source[gb] -= j;
                }
                self.tails.push(ga as u32);
                self.heads.push(gb as u32);
                self.capacities.push(j + j - 0.0 - 0.0);
            } else {
                // NB an auxiliary node for a pair that disagrees off alpha, J to the source. OpenGM's arcs run from
                //    each end to it, which no cut needs to sever (T- #195); the corrected arcs run from it to each end.
                let x = self.source.len();
                self.source.push(0.0);
                self.sink.push(0.0);
                for end in [self.node[a], self.node[b]] {
                    let (tail, head) = if opengm_patch { (x as u32, end) } else { (end, x as u32) };
                    self.tails.push(tail);
                    self.heads.push(head);
                    self.capacities.push(j + 0.0 - 0.0 - 0.0);
                }
                add_unary(&mut self.source, &mut self.sink, x, 0.0, j);
            }
        }
        self.graph
            .build(&self.source, &self.sink, &self.tails, &self.heads, &self.capacities);
        self.graph.solve(&mut self.sink_side);
        for i in 0..n {
            if labels[i] != alpha && !self.sink_side[self.node[i] as usize] {
                labels[i] = alpha;
            }
        }
    }
}

fn add_unary(source: &mut [f64], sink: &mut [f64], g: usize, v0: f64, v1: f64) {
    if v0 <= v1 {
        source[g] += v1 - v0;
    } else {
        sink[g] += v0 - v1;
    }
}

/// Expand from `labels`, in place, until `n_states` moves in a row lower the metric energy by nothing or `max_steps`
/// moves are made; returns the moves made. OpenGM's moves, `expand_with(.., false)`.
pub fn expand(problem: &Problem, labels: &mut [usize], max_steps: usize) -> usize {
    expand_with(problem, labels, max_steps, false)
}

/// `expand`, its moves with T- #195's auxiliary node where `opengm_patch`, else OpenGM's.
pub fn expand_with(problem: &Problem, labels: &mut [usize], max_steps: usize, opengm_patch: bool) -> usize {
    let k = problem.n_states;
    let mut moves = Move::default();
    let mut energy = problem.metric_energy(labels);
    let mut unchanged = 0;
    let mut alpha = 0;
    let mut steps = 0;
    while steps < max_steps && unchanged < k {
        steps += 1;
        moves.expand(problem, alpha, labels, opengm_patch);
        let after = problem.metric_energy(labels);
        if after < energy {
            energy = after;
            unchanged = 0;
        } else {
            unchanged += 1;
        }
        alpha = (alpha + 1) % k;
    }
    steps
}
