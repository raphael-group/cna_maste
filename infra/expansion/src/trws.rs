//! Sequential tree-reweighted message passing (TRW-S; Kolmogorov 2006) of the Potts energy
//! `sum unary[i, x_i] - sum J [x_a == x_b]`: OpenGM's `TRWSi` (Savchynskyy's, MIT) as `cnamaste_opengm.cxx` runs it
//! (T- #198), in its order of operations.
//!
//! - The pairs are split into monotone chains (`MonotoneChainsDecomposition`): from each node in turn, a walk along
//!   the first unused pair to a larger node, until none is left; a node with no pair is a chain of its own, and those
//!   come first. Each node's unary is shared equally among the chains that hold it.
//! - One forward pass of min-sum along every chain, then passes that alternate direction (`BackwardMove`): at each
//!   node, in global order, each chain holding it reads its min-marginal there, and each is moved to their average.
//!   Each pair's message is the O(K) Potts update (`_PottsUnaryTransform`, `fastComputations`).
//! - After each pass, a labelling is rounded node by node in the pass's order from the summed min-marginals and the
//!   pairs to nodes already labelled (`_EstimateIntegerLabeling`); the best kept. The dual bound is the chains' minima
//!   summed.
//! - Stops: every chain's own labelling agrees at every node; the gap to the best labelling within
//!   `tolerance |bound|`; the bound raised by less than `tolerance |bound|` in a pass; or `max_iterations` passes.

use crate::expansion::Problem;

// NB OpenGM writes several sums and products in the other order (`m + avg`, `-1.0 * x`, `scale * x`); IEEE addition
//    and multiplication commute, and `-1.0 * x` is `-x`, so the bits are the same.

/// Per node, `(pair, other end)` in pair order.
type Incident = Vec<Vec<(usize, usize)>>;

/// One chain of nodes, increasing, and its min-sum state (`SequenceStorage`, `DynamicProgramming`, `MaxSumSolver`).
struct Chain {
    vars: Vec<usize>,
    pairs: Vec<usize>,
    unary: Vec<f64>,
    marginals: Vec<f64>,
    labeling: Vec<usize>,
    buffer: Vec<f64>,
    current: usize,
    direct: bool,
    init_needed: bool,
    objective: f64,
}

/// The first index of the smallest value, as `std::max_element` under OpenGM's `Minimizer::ibop`.
fn argmin(values: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[best] > values[i] {
            best = i;
        }
    }
    best
}

/// `_PottsUnaryTransform` with `params` `(J, -J)` (a Potts pair of `-J` on agreement, 0 off it, as OpenGM reads it):
/// each value bounded above by the smallest plus `params[0]`, then `params[1]` added where it is non-zero.
fn potts_transform(buffer: &mut [f64], params: (f64, f64)) {
    let best_index = argmin(buffer);
    let best = buffer[best_index];
    let mut second = best;
    let anti = params.0 < 0.0;
    if anti {
        buffer[best_index] = f64::INFINITY;
        second = buffer[argmin(buffer)];
        buffer[best_index] = best;
    }
    let cap = best + params.0;
    for x in buffer.iter_mut() {
        *x = if *x < cap { *x } else { cap };
    }
    if anti {
        let other = second + params.0;
        buffer[best_index] = if other < best { other } else { best };
    }
    if params.1 != 0.0 {
        for x in buffer.iter_mut() {
            *x += params.1;
        }
    }
}

impl Chain {
    fn len(&self) -> usize {
        self.vars.len()
    }

    fn init_buffer(&mut self, index: usize, k: usize) {
        self.current = index;
        self.buffer.clear();
        self.buffer.extend_from_slice(&self.unary[index * k..index * k + k]);
    }

    fn core_init(&mut self, direct: bool, k: usize) {
        self.direct = direct;
        let start = if direct { 0 } else { self.len() - 1 };
        self.init_buffer(start, k);
        self.init_needed = false;
    }

    fn update_marginals(&mut self, k: usize) {
        let c = self.current;
        self.marginals[c * k..c * k + k].copy_from_slice(&self.buffer);
    }

    fn init_move(&mut self, direct: bool, k: usize) {
        self.core_init(direct, k);
        self.update_marginals(k);
    }

    /// One step along the chain: the pair's Potts message, then the next node's unary.
    fn push(&mut self, params: &[(f64, f64)], k: usize) {
        let pair = self.pairs[if self.direct { self.current } else { self.current - 1 }];
        self.current = if self.direct {
            self.current + 1
        } else {
            self.current - 1
        };
        potts_transform(&mut self.buffer, params[pair]);
        let c = self.current;
        for j in 0..k {
            self.buffer[j] += self.unary[c * k + j] * 1.0;
        }
    }

    /// `Move`: a full pass, the marginals the accumulated messages.
    fn move_forward(&mut self, params: &[(f64, f64)], k: usize) {
        if self.init_needed {
            self.init_move(!self.direct, k);
        }
        for _ in 1..self.len() {
            self.push(params, k);
            self.update_marginals(k);
        }
        self.finalize(params, k);
    }

    /// `PushBack`: one step, the marginals the sum of both directions' messages less the unary counted twice.
    fn push_back(&mut self, params: &[(f64, f64)], k: usize) {
        if self.init_needed {
            self.core_init(!self.direct, k);
        }
        self.push(params, k);
        let c = self.current;
        for j in 0..k {
            let m = &mut self.marginals[c * k + j];
            // NB OpenGM's `buffer + m`, then `m + unary * (-1 / rho)`, rho 1: the same bits in either order.
            *m += self.buffer[j];
            *m += -self.unary[c * k + j];
        }
    }

    fn increase_unary_weights(&mut self, delta: &[f64], k: usize) {
        let c = self.current;
        for j in 0..k {
            self.unary[c * k + j] += delta[j];
            self.buffer[j] += delta[j] * 1.0;
        }
    }

    /// `FinalizeMove`: the chain's minimum at the pass's last node, and its own labelling traced back.
    fn finalize(&mut self, params: &[(f64, f64)], k: usize) {
        let c = self.current;
        let row = &self.marginals[c * k..c * k + k];
        self.labeling[c] = argmin(row);
        self.objective = row[self.labeling[c]];
        self.init_needed = true;
        let direct = !self.direct;
        let mut index = c;
        let mut label = self.labeling[c];
        let mut temp = vec![0.0; k];
        for _ in 1..self.len() {
            let pair = if direct {
                self.pairs[index]
            } else {
                self.pairs[index - 1]
            };
            index = if direct { index + 1 } else { index - 1 };
            temp.copy_from_slice(&self.marginals[index * k..index * k + k]);
            if label < k {
                temp[label] -= params[pair].0;
            }
            label = argmin(&temp);
            self.labeling[index] = label;
        }
    }
}

/// TRW-S's answer: the best rounded labelling, the dual bound, and the bound after the forward pass and each pass.
pub struct TrwsResult {
    pub labels: Vec<usize>,
    pub bound: f64,
    pub trace: Vec<f64>,
}

/// The monotone chains of `problem`'s pairs, the isolated nodes first, as `MonotoneChainsDecomposition` builds them:
/// `(vars, pairs)` per chain.
pub fn chains(problem: &Problem) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n = problem.n_nodes();
    let mut edges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut degree = vec![0usize; n];
    for e in 0..problem.first.len() {
        let (a, b) = (problem.first[e], problem.second[e]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        edges[lo].push((e, hi));
        degree[a] += 1;
        degree[b] += 1;
    }
    let mut out: Vec<(Vec<usize>, Vec<usize>)> =
        (0..n).filter(|&i| degree[i] == 0).map(|i| (vec![i], vec![])).collect();
    let mut next = vec![0usize; n];
    for start in 0..n {
        while next[start] < edges[start].len() {
            let (mut vars, mut pairs) = (vec![start], Vec::new());
            let mut node = start;
            while next[node] < edges[node].len() {
                let (pair, neighbour) = edges[node][next[node]];
                next[node] += 1;
                vars.push(neighbour);
                pairs.push(pair);
                node = neighbour;
            }
            out.push((vars, pairs));
        }
    }
    out
}

/// OpenGM's `evaluate`: the unaries in node order, then each pair's `-J` on agreement or 0 off it, in pair order.
fn evaluate(problem: &Problem, labels: &[usize]) -> f64 {
    let k = problem.n_states;
    let mut value = 0.0;
    for (i, &x) in labels.iter().enumerate() {
        value += problem.unary[i * k + x];
    }
    for e in 0..problem.first.len() {
        value += if labels[problem.first[e]] == labels[problem.second[e]] {
            -problem.coupling[e]
        } else {
            0.0
        };
    }
    value
}

/// TRW-S on `problem`, at most `max_iterations` passes after the forward one, `tolerance` relative to the bound.
pub fn trws(problem: &Problem, max_iterations: usize, tolerance: f64) -> TrwsResult {
    let (n, k) = (problem.n_nodes(), problem.n_states);
    // NB a Potts pair's (f(1, 0) - f(0, 0), f(0, 0)), f(0, 0) = -J; OpenGM leaves the first 0 at one label.
    let params: Vec<(f64, f64)> = problem
        .coupling
        .iter()
        .map(|&j| (if k > 1 { 0.0 - -j } else { 0.0 }, -j))
        .collect();
    let decomposition = chains(problem);
    let mut var_list: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (m, (vars, _)) in decomposition.iter().enumerate() {
        for (local, &v) in vars.iter().enumerate() {
            var_list[v].push((m, local));
        }
    }
    let mut solvers: Vec<Chain> = decomposition
        .into_iter()
        .map(|(vars, pairs)| {
            let mut unary = vec![0.0; vars.len() * k];
            for (local, &v) in vars.iter().enumerate() {
                let share = 1.0 / var_list[v].len() as f64;
                for j in 0..k {
                    unary[local * k + j] = problem.unary[v * k + j] * share;
                }
            }
            let len = vars.len();
            Chain {
                vars,
                pairs,
                unary,
                marginals: vec![0.0; len * k],
                labeling: vec![usize::MAX; len],
                buffer: Vec::with_capacity(k),
                current: usize::MAX,
                direct: true,
                init_needed: true,
                objective: 0.0,
            }
        })
        .collect();
    let (mut forward, mut backward): (Incident, Incident) = (vec![Vec::new(); n], vec![Vec::new(); n]);
    for e in 0..problem.first.len() {
        let (a, b) = (problem.first[e], problem.second[e]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        forward[hi].push((e, lo));
        backward[lo].push((e, hi));
    }
    let objective = |solvers: &[Chain]| solvers.iter().fold(0.0, |acc, s| acc + s.objective);

    for s in solvers.iter_mut() {
        s.init_move(true, k);
    }
    for s in solvers.iter_mut() {
        s.move_forward(&params, k);
    }
    let mut direct = false;
    let mut dual = objective(&solvers);
    let mut old = dual;
    let mut trace = vec![dual];
    let (mut labels, mut best_labels) = (vec![0usize; n], vec![0usize; n]);
    let mut best = f64::INFINITY;
    let (mut average, mut sum) = (vec![0.0; k], vec![0.0; k]);
    let mut deltas: Vec<Vec<f64>> = vec![vec![0.0; k]; solvers.len()];

    for _ in 0..max_iterations {
        for i in 0..n {
            let var = if direct { i } else { n - 1 - i };
            average.iter_mut().for_each(|x| *x = 0.0);
            for &(m, local) in &var_list[var] {
                let s = &mut solvers[m];
                let start = if direct { 0 } else { s.len() - 1 };
                if local != start {
                    s.push_back(&params, k);
                }
                let c = s.current;
                deltas[m].copy_from_slice(&s.marginals[c * k..c * k + k]);
                for j in 0..k {
                    average[j] += deltas[m][j];
                }
            }
            let scale = -1.0 / var_list[var].len() as f64;
            average.iter_mut().for_each(|x| *x *= scale);
            for &(m, local) in &var_list[var] {
                let s = &mut solvers[m];
                let start = if direct { 0 } else { s.len() - 1 };
                for j in 0..k {
                    deltas[m][j] = -(deltas[m][j] + average[j]);
                }
                s.increase_unary_weights(&deltas[m], k);
                if local != start {
                    s.update_marginals(k);
                } else {
                    s.init_move(!s.direct, k);
                }
            }
        }
        for s in solvers.iter_mut() {
            s.finalize(&params, k);
        }
        direct = !direct;
        for i in 0..n {
            let var = if direct { i } else { n - 1 - i };
            sum.iter_mut().for_each(|x| *x = 0.0);
            for &(m, local) in &var_list[var] {
                let row = &solvers[m].marginals[local * k..local * k + k];
                for j in 0..k {
                    sum[j] += row[j];
                }
            }
            for &(pair, other) in if direct { &forward[var] } else { &backward[var] } {
                let fixed = labels[other];
                if fixed < k {
                    sum[fixed] -= params[pair].0;
                }
            }
            labels[var] = argmin(&sum);
        }
        let value = evaluate(problem, &labels);
        if value < best {
            best = value;
            best_labels.copy_from_slice(&labels);
        }
        dual = objective(&solvers);
        trace.push(dual);

        let agree = (0..n).all(|v| {
            let first = &var_list[v][0];
            let label = solvers[first.0].labeling[first.1];
            var_list[v]
                .iter()
                .all(|&(m, local)| solvers[m].labeling[local] == label)
        });
        if agree {
            break;
        }
        let precision = (dual.abs() * f64::EPSILON).max(tolerance);
        if (best - dual).abs() <= dual.abs() * precision {
            break;
        }
        if tolerance >= 0.0 && dual < old + dual.abs() * 1.0 * tolerance {
            break;
        }
        old = dual;
    }
    TrwsResult {
        labels: best_labels,
        bound: dual,
        trace,
    }
}

#[cfg(test)]
mod tests {
    //! The parts against direct construction: the first-minimum argmin, the O(K) Potts message against the O(K^2)
    //! min-sum, the chains against the pairs, and each node's unary shared out whole.

    use super::{argmin, chains, potts_transform, trws};
    use crate::expansion::Problem;

    /// SplitMix64.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn uniform(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[test]
    fn argmin_takes_the_first_of_tied_minima() {
        assert_eq!(argmin(&[3.0, 1.0, 2.0, 1.0]), 1);
        assert_eq!(argmin(&[0.5]), 0);
        assert_eq!(argmin(&[-1.0, -1.0]), 0);
    }

    /// The min-sum message through a pair of `-J` on agreement and 0 off it, by every pair of labels.
    fn min_sum(values: &[f64], j: f64) -> Vec<f64> {
        (0..values.len())
            .map(|x| {
                (0..values.len())
                    .map(|y| values[y] + if x == y { -j } else { 0.0 })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect()
    }

    /// 1,000 random messages, 1 to 9 labels, couplings of either sign: the O(K) transform is the O(K^2) min-sum to
    /// 1e-12 relative, and bitwise on integer values. Fails where the cap, the anti-Potts second minimum or the shift
    /// by `-J` is wrong.
    #[test]
    fn the_potts_message_is_the_min_sum_message() {
        let mut rng = Rng(5);
        for case in 0..1000 {
            let k = 1 + rng.below(9);
            let integer = case % 2 == 0;
            let draw = |rng: &mut Rng| {
                let x = 20.0 * rng.uniform() - 10.0;
                if integer {
                    x.round()
                } else {
                    x
                }
            };
            let values: Vec<f64> = (0..k).map(|_| draw(&mut rng)).collect();
            let j = draw(&mut rng);
            let mut found = values.clone();
            potts_transform(&mut found, (if k > 1 { j } else { 0.0 }, -j));
            let expected = if k > 1 {
                min_sum(&values, j)
            } else {
                vec![values[0] - j]
            };
            for (f, e) in found.iter().zip(&expected) {
                if integer {
                    assert_eq!(f, e, "{values:?}, J {j}");
                } else {
                    assert!(
                        (f - e).abs() <= 1e-12 * e.abs().max(1.0),
                        "{values:?}, J {j}: {f} vs {e}"
                    );
                }
            }
        }
    }

    fn random_problem(rng: &mut Rng, n: usize, k: usize, m: usize) -> (Vec<f64>, Vec<usize>, Vec<usize>, Vec<f64>) {
        let unary = (0..n * k).map(|_| rng.uniform()).collect();
        let (mut first, mut second, mut coupling) = (vec![], vec![], vec![]);
        for _ in 0..m {
            let a = rng.below(n);
            let b = (a + 1 + rng.below(n - 1)) % n;
            first.push(a.min(b));
            second.push(a.max(b));
            coupling.push(rng.uniform());
        }
        (unary, first, second, coupling)
    }

    /// 200 random graphs, isolated nodes among them: each pair in exactly one chain, joining that chain's consecutive
    /// nodes, nodes increasing along each chain, the isolated nodes first and alone. Fails where a pair is lost,
    /// doubled or misplaced in a chain.
    #[test]
    fn the_chains_cover_every_pair_once_increasing() {
        let mut rng = Rng(7);
        for _ in 0..200 {
            let n = 2 + rng.below(30);
            let m = rng.below(3 * n);
            let (unary, first, second, coupling) = random_problem(&mut rng, n, 2, m);
            let problem = Problem {
                n_states: 2,
                unary: &unary,
                first: &first,
                second: &second,
                coupling: &coupling,
            };
            let found = chains(&problem);
            let mut seen = vec![0usize; m];
            let mut isolated = vec![true; n];
            for e in 0..m {
                isolated[first[e]] = false;
                isolated[second[e]] = false;
            }
            let leading = found
                .iter()
                .take_while(|(vars, pairs)| vars.len() == 1 && pairs.is_empty())
                .count();
            assert_eq!(leading, isolated.iter().filter(|&&x| x).count());
            for (vars, pairs) in &found {
                assert_eq!(vars.len(), pairs.len() + 1);
                for (i, &e) in pairs.iter().enumerate() {
                    seen[e] += 1;
                    assert_eq!((first[e], second[e]), (vars[i], vars[i + 1]));
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "{seen:?}");
        }
    }

    /// Two nodes at labels 0 and 1, coupling 1, label 2 costing each 0.4 more: TRW-S reaches (2, 2), Potts energy
    /// -0.2, its bound equal, the minimum by enumeration (T- #195's case, where OpenGM's expansion stops at (0, 1)).
    #[test]
    fn two_nodes_reach_the_enumerated_minimum_and_certify_it() {
        let unary = [0.0, 9.0, 0.4, 9.0, 0.0, 0.4];
        let problem = Problem {
            n_states: 3,
            unary: &unary,
            first: &[0],
            second: &[1],
            coupling: &[1.0],
        };
        let found = trws(&problem, 1000, 1e-9);
        assert_eq!(found.labels, vec![2, 2]);
        assert_eq!(problem.potts_energy(&found.labels), found.bound);
    }
}
