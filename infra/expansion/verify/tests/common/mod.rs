//! The tests' and benchmarks' shared referees and generators: brute-force min cuts and labellings, seeded random
//! graphs and planted square lattices (`tests/potts_lattice.py`'s, the signal given rather than solved for).

// NB index loops over parallel arrays read as the algorithm states them, as `lib.rs` allows.
#![allow(dead_code, clippy::needless_range_loop)]

use cnamaste_expansion::expansion::Problem;

/// SplitMix64: a seeded generator for fixtures outside `proptest`.
pub struct SplitMix(u64);

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        SplitMix(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform on 0..n.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Standard normal, Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u = 1.0 - self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// A directed graph with per-node terminal capacities, as `maxflow::Graph::build` takes it.
#[derive(Clone, Debug)]
pub struct Cut {
    pub source: Vec<f64>,
    pub sink: Vec<f64>,
    pub tails: Vec<u32>,
    pub heads: Vec<u32>,
    pub capacities: Vec<f64>,
}

impl Cut {
    pub fn n_nodes(&self) -> usize {
        self.source.len()
    }

    /// The capacity of the cut with `sink_side`: a source-side node's sink capacity, a sink-side node's source
    /// capacity, and each edge from the source's side to the sink's.
    pub fn value(&self, sink_side: &[bool]) -> f64 {
        let mut value = 0.0;
        for i in 0..self.n_nodes() {
            value += if sink_side[i] { self.source[i] } else { self.sink[i] };
        }
        for e in 0..self.tails.len() {
            if !sink_side[self.tails[e] as usize] && sink_side[self.heads[e] as usize] {
                value += self.capacities[e];
            }
        }
        value
    }

    /// The minimum cut by enumeration, and its smallest sink side: the intersection of every minimum cut's sink
    /// side, itself a minimum cut, which is the side Boost Graph's Boykov-Kolmogorov returns. Cuts within `rtol` of
    /// the minimum are minimum.
    pub fn brute_force(&self, rtol: f64) -> (f64, Vec<bool>) {
        let n = self.n_nodes();
        assert!(n <= 16, "enumeration is for small graphs");
        let values: Vec<f64> = (0..1u32 << n).map(|bits| self.value(&sides(bits, n))).collect();
        let best = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut smallest = vec![true; n];
        for (bits, &v) in values.iter().enumerate() {
            if v <= best + rtol * best.abs().max(1.0) {
                for (s, x) in smallest.iter_mut().zip(sides(bits as u32, n)) {
                    *s &= x;
                }
            }
        }
        (best, smallest)
    }
}

fn sides(bits: u32, n: usize) -> Vec<bool> {
    (0..n).map(|i| bits >> i & 1 == 1).collect()
}

/// A Potts problem owning its arrays; `view` lends it as the crate's `Problem`.
#[derive(Clone, Debug)]
pub struct Owned {
    pub n_states: usize,
    pub unary: Vec<f64>,
    pub first: Vec<usize>,
    pub second: Vec<usize>,
    pub coupling: Vec<f64>,
}

impl Owned {
    pub fn view(&self) -> Problem<'_> {
        Problem {
            n_states: self.n_states,
            unary: &self.unary,
            first: &self.first,
            second: &self.second,
            coupling: &self.coupling,
        }
    }

    pub fn n_nodes(&self) -> usize {
        self.unary.len() / self.n_states
    }

    /// Every labelling and its metric energy, labellings in odometer order (node 0 fastest).
    pub fn enumerate(&self) -> Vec<(Vec<usize>, f64)> {
        let (n, k) = (self.n_nodes(), self.n_states);
        let total = k.checked_pow(n as u32).expect("enumeration overflows");
        assert!(total <= 1 << 20, "enumeration is for small problems");
        let problem = self.view();
        let mut labels = vec![0usize; n];
        let mut out = Vec::with_capacity(total);
        for _ in 0..total {
            out.push((labels.clone(), problem.metric_energy(&labels)));
            for x in labels.iter_mut() {
                *x += 1;
                if *x < k {
                    break;
                }
                *x = 0;
            }
        }
        out
    }

    /// The problem with node `i` renamed `order[i]`, its unary rows, pairs and pair order following.
    pub fn renamed(&self, order: &[usize], pair_order: &[usize]) -> Owned {
        let (n, k) = (self.n_nodes(), self.n_states);
        let mut unary = vec![0.0; n * k];
        for i in 0..n {
            unary[order[i] * k..order[i] * k + k].copy_from_slice(&self.unary[i * k..i * k + k]);
        }
        let mut first = Vec::new();
        let mut second = Vec::new();
        let mut coupling = Vec::new();
        for &e in pair_order {
            let (a, b) = (order[self.first[e]], order[self.second[e]]);
            first.push(a.min(b));
            second.push(a.max(b));
            coupling.push(self.coupling[e]);
        }
        Owned {
            n_states: k,
            unary,
            first,
            second,
            coupling,
        }
    }
}

/// `potts_problem` of a planted square lattice: `rows x cols` spots, a Voronoi truth from `n_states` seeds, the
/// field `signal` at the true label plus unit normal noise, unary its negative, and one pair of coupling
/// `spatial_weight` per 8-neighbour (`get_square_adjacency`, both directions of weight 1 summed and halved).
pub fn planted_lattice(
    rows: usize,
    cols: usize,
    n_states: usize,
    signal: f64,
    spatial_weight: f64,
    seed: u64,
) -> Owned {
    let mut rng = SplitMix::new(seed);
    let seeds: Vec<(f64, f64)> = (0..n_states)
        .map(|_| (rng.uniform() * rows as f64, rng.uniform() * cols as f64))
        .collect();
    let n = rows * cols;
    let mut unary = vec![0.0; n * n_states];
    for i in 0..n {
        let (r, c) = ((i / cols) as f64, (i % cols) as f64);
        let truth = (0..n_states)
            .min_by(|&a, &b| {
                let da = (r - seeds[a].0).powi(2) + (c - seeds[a].1).powi(2);
                let db = (r - seeds[b].0).powi(2) + (c - seeds[b].1).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        for x in 0..n_states {
            let field = rng.normal() + if x == truth { signal } else { 0.0 };
            unary[i * n_states + x] = -field;
        }
    }
    let (mut first, mut second) = (Vec::new(), Vec::new());
    for r in 0..rows {
        for c in 0..cols {
            for (dr, dc) in [(0i64, 1i64), (1, 0), (1, 1), (1, -1)] {
                let (r2, c2) = (r as i64 + dr, c as i64 + dc);
                if r2 < rows as i64 && c2 >= 0 && c2 < cols as i64 {
                    let (a, b) = (r * cols + c, r2 as usize * cols + c2 as usize);
                    first.push(a.min(b));
                    second.push(a.max(b));
                }
            }
        }
    }
    let coupling = vec![spatial_weight; first.len()];
    Owned {
        n_states,
        unary,
        first,
        second,
        coupling,
    }
}

/// Each node's unary argmin, the first on ties.
pub fn argmin_labels(problem: &Owned) -> Vec<usize> {
    let k = problem.n_states;
    (0..problem.n_nodes())
        .map(|i| {
            let row = &problem.unary[i * k..i * k + k];
            (0..k).fold(0, |best, x| if row[x] < row[best] { x } else { best })
        })
        .collect()
}
