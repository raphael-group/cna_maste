//! Boykov-Kolmogorov max flow (Boykov and Kolmogorov 2004), with the timestamp and distance heuristics of
//! Kolmogorov's maxflow 3: two search trees grown from the terminals, augmented where they meet, and orphans adopted
//! rather than the trees regrown.
//!
//! The graph is CSR, rebuilt per problem by `Graph::build`; each node's terminal capacities are folded into one
//! `terminal` residual, positive to the source and negative to the sink. The cut returned is the one Boost Graph's
//! Boykov-Kolmogorov gives OpenGM: a node is on the sink's side where it reaches the sink in the residual graph, so a
//! node either side could take is on the source's. That set is the same for every maximum flow.

use std::collections::VecDeque;

const NONE: u32 = u32::MAX;
const TERMINAL: u32 = u32::MAX - 1;
const ORPHAN: u32 = u32::MAX - 2;
const INFINITE_DISTANCE: u32 = u32::MAX;

/// A directed graph with source and sink capacities per node, and the search state of one max flow on it.
#[derive(Default)]
pub struct Graph {
    first: Vec<u32>,
    head: Vec<u32>,
    sister: Vec<u32>,
    residual: Vec<f64>,
    terminal: Vec<f64>,
    parent: Vec<u32>,
    is_sink: Vec<bool>,
    timestamp: Vec<u32>,
    distance: Vec<u32>,
    in_active: Vec<bool>,
    active: VecDeque<u32>,
    orphans: VecDeque<u32>,
    degree: Vec<u32>,
    time: u32,
    pub flow: f64,
}

impl Graph {
    /// Rebuild the graph on `n_nodes`: `source[i]`, `sink[i]` each node's terminal capacities, and each edge
    /// `tails[e] -> heads[e]` of capacity `capacities[e]`, its reverse arc of capacity 0. Allocations are kept.
    pub fn build(&mut self, source: &[f64], sink: &[f64], tails: &[u32], heads: &[u32], capacities: &[f64]) {
        let n = source.len();
        let m = tails.len();
        self.degree.clear();
        self.degree.resize(n + 1, 0);
        for e in 0..m {
            self.degree[tails[e] as usize] += 1;
            self.degree[heads[e] as usize] += 1;
        }
        self.first.clear();
        self.first.resize(n + 1, 0);
        for i in 0..n {
            self.first[i + 1] = self.first[i] + self.degree[i];
        }
        self.head.clear();
        self.head.resize(2 * m, 0);
        self.sister.clear();
        self.sister.resize(2 * m, 0);
        self.residual.clear();
        self.residual.resize(2 * m, 0.0);
        // NB `degree` is reused as each node's next free arc.
        self.degree[..n].copy_from_slice(&self.first[..n]);
        for e in 0..m {
            let (u, v) = (tails[e] as usize, heads[e] as usize);
            let (a, b) = (self.degree[u] as usize, self.degree[v] as usize);
            self.degree[u] += 1;
            self.degree[v] += 1;
            self.head[a] = v as u32;
            self.head[b] = u as u32;
            self.sister[a] = b as u32;
            self.sister[b] = a as u32;
            self.residual[a] = capacities[e];
        }
        self.terminal.clear();
        self.flow = 0.0;
        for i in 0..n {
            self.terminal.push(source[i] - sink[i]);
            self.flow += source[i].min(sink[i]);
        }
    }

    fn arcs(&self, i: usize) -> std::ops::Range<usize> {
        self.first[i] as usize..self.first[i + 1] as usize
    }

    fn activate(&mut self, i: usize) {
        if !self.in_active[i] {
            self.in_active[i] = true;
            self.active.push_back(i as u32);
        }
    }

    /// Run the max flow, then write `sink_side[i]`: whether node `i` reaches the sink in the residual graph.
    pub fn solve(&mut self, sink_side: &mut Vec<bool>) {
        let n = self.terminal.len();
        self.parent.clear();
        self.parent.resize(n, NONE);
        self.is_sink.clear();
        self.is_sink.resize(n, false);
        self.timestamp.clear();
        self.timestamp.resize(n, 0);
        self.distance.clear();
        self.distance.resize(n, 0);
        self.in_active.clear();
        self.in_active.resize(n, false);
        self.active.clear();
        self.orphans.clear();
        self.time = 0;
        for i in 0..n {
            if self.terminal[i] != 0.0 {
                self.is_sink[i] = self.terminal[i] < 0.0;
                self.parent[i] = TERMINAL;
                self.distance[i] = 1;
                self.activate(i);
            }
        }
        let mut current: Option<usize> = None;
        loop {
            // NB the node that found the last path is searched again before the queue, as maxflow 3 does.
            let i = match current.take() {
                Some(i) if self.parent[i] != NONE => {
                    self.in_active[i] = false;
                    i
                }
                Some(i) => {
                    self.in_active[i] = false;
                    match self.next_active() {
                        Some(i) => i,
                        None => break,
                    }
                }
                None => match self.next_active() {
                    Some(i) => i,
                    None => break,
                },
            };
            let middle = self.grow(i);
            self.time += 1;
            if middle != NONE {
                self.in_active[i] = true;
                current = Some(i);
                self.augment(middle as usize);
                while let Some(o) = self.orphans.pop_front() {
                    self.adopt(o as usize);
                }
            }
        }
        self.mark_sink_side(sink_side);
    }

    fn next_active(&mut self) -> Option<usize> {
        while let Some(i) = self.active.pop_front() {
            let i = i as usize;
            self.in_active[i] = false;
            if self.parent[i] != NONE {
                return Some(i);
            }
        }
        None
    }

    /// Grow `i`'s tree by its arcs; returns the arc from the source's tree to the sink's where they meet, or NONE.
    fn grow(&mut self, i: usize) -> u32 {
        let sink = self.is_sink[i];
        for a in self.arcs(i) {
            let open = if sink {
                self.residual[self.sister[a] as usize]
            } else {
                self.residual[a]
            };
            if open <= 0.0 {
                continue;
            }
            let j = self.head[a] as usize;
            if self.parent[j] == NONE {
                self.is_sink[j] = sink;
                self.parent[j] = self.sister[a];
                self.timestamp[j] = self.timestamp[i];
                self.distance[j] = self.distance[i] + 1;
                self.activate(j);
            } else if self.is_sink[j] != sink {
                return if sink { self.sister[a] } else { a as u32 };
            } else if self.timestamp[j] <= self.timestamp[i] && self.distance[j] > self.distance[i] {
                self.parent[j] = self.sister[a];
                self.timestamp[j] = self.timestamp[i];
                self.distance[j] = self.distance[i] + 1;
            }
        }
        NONE
    }

    fn augment(&mut self, middle: usize) {
        let tail = self.head[self.sister[middle] as usize] as usize;
        let mut bottleneck = self.residual[middle];
        let mut i = tail;
        loop {
            let a = self.parent[i];
            if a == TERMINAL {
                break;
            }
            bottleneck = bottleneck.min(self.residual[self.sister[a as usize] as usize]);
            i = self.head[a as usize] as usize;
        }
        bottleneck = bottleneck.min(self.terminal[i]);
        i = self.head[middle] as usize;
        loop {
            let a = self.parent[i];
            if a == TERMINAL {
                break;
            }
            bottleneck = bottleneck.min(self.residual[a as usize]);
            i = self.head[a as usize] as usize;
        }
        bottleneck = bottleneck.min(-self.terminal[i]);

        self.residual[self.sister[middle] as usize] += bottleneck;
        self.residual[middle] -= bottleneck;
        i = tail;
        loop {
            let a = self.parent[i];
            if a == TERMINAL {
                break;
            }
            let (a, s) = (a as usize, self.sister[a as usize] as usize);
            self.residual[a] += bottleneck;
            self.residual[s] -= bottleneck;
            if self.residual[s] == 0.0 {
                self.set_orphan_front(i);
            }
            i = self.head[a] as usize;
        }
        self.terminal[i] -= bottleneck;
        if self.terminal[i] == 0.0 {
            self.set_orphan_front(i);
        }
        i = self.head[middle] as usize;
        loop {
            let a = self.parent[i];
            if a == TERMINAL {
                break;
            }
            let (a, s) = (a as usize, self.sister[a as usize] as usize);
            self.residual[s] += bottleneck;
            self.residual[a] -= bottleneck;
            if self.residual[a] == 0.0 {
                self.set_orphan_front(i);
            }
            i = self.head[a] as usize;
        }
        self.terminal[i] += bottleneck;
        if self.terminal[i] == 0.0 {
            self.set_orphan_front(i);
        }
        self.flow += bottleneck;
    }

    fn set_orphan_front(&mut self, i: usize) {
        self.parent[i] = ORPHAN;
        self.orphans.push_front(i as u32);
    }

    fn set_orphan_rear(&mut self, i: usize) {
        self.parent[i] = ORPHAN;
        self.orphans.push_back(i as u32);
    }

    /// The distance from `j` to its tree's terminal through valid parents, marking the path, or INFINITE_DISTANCE
    /// where the path meets an orphan.
    fn origin_distance(&mut self, start: usize) -> u32 {
        let mut j = start;
        let mut d: u32 = 0;
        loop {
            if self.timestamp[j] == self.time {
                d += self.distance[j];
                break;
            }
            let a = self.parent[j];
            d += 1;
            if a == TERMINAL {
                self.timestamp[j] = self.time;
                self.distance[j] = 1;
                break;
            }
            if a == ORPHAN {
                return INFINITE_DISTANCE;
            }
            j = self.head[a as usize] as usize;
        }
        let mut marked = d;
        j = start;
        while self.timestamp[j] != self.time {
            self.timestamp[j] = self.time;
            self.distance[j] = marked;
            marked -= 1;
            j = self.head[self.parent[j] as usize] as usize;
        }
        d
    }

    /// Find orphan `i` a parent in its own tree, or free it and orphan its children.
    fn adopt(&mut self, i: usize) {
        let sink = self.is_sink[i];
        let mut best = NONE;
        let mut best_distance = INFINITE_DISTANCE;
        for a0 in self.arcs(i) {
            let open = if sink {
                self.residual[a0]
            } else {
                self.residual[self.sister[a0] as usize]
            };
            if open <= 0.0 {
                continue;
            }
            let j = self.head[a0] as usize;
            if self.is_sink[j] != sink || self.parent[j] == NONE {
                continue;
            }
            let d = self.origin_distance(j);
            if d < best_distance {
                best = a0 as u32;
                best_distance = d;
            }
        }
        if best != NONE {
            self.parent[i] = best;
            self.timestamp[i] = self.time;
            self.distance[i] = best_distance + 1;
            return;
        }
        self.parent[i] = NONE;
        for a0 in self.arcs(i) {
            let j = self.head[a0] as usize;
            if self.is_sink[j] != sink || self.parent[j] == NONE {
                continue;
            }
            let open = if sink {
                self.residual[a0]
            } else {
                self.residual[self.sister[a0] as usize]
            };
            if open > 0.0 {
                self.activate(j);
            }
            let a = self.parent[j];
            if a != TERMINAL && a != ORPHAN && self.head[a as usize] as usize == i {
                self.set_orphan_rear(j);
            }
        }
    }

    /// `sink_side[i]`: node `i` reaches the sink in the residual graph, by search back from the sink.
    fn mark_sink_side(&mut self, sink_side: &mut Vec<bool>) {
        let n = self.terminal.len();
        sink_side.clear();
        sink_side.resize(n, false);
        let mut queue: Vec<u32> = Vec::new();
        for i in 0..n {
            if self.terminal[i] < 0.0 {
                sink_side[i] = true;
                queue.push(i as u32);
            }
        }
        while let Some(j) = queue.pop() {
            for a in self.arcs(j as usize) {
                let k = self.head[a] as usize;
                if !sink_side[k] && self.residual[self.sister[a] as usize] > 0.0 {
                    sink_side[k] = true;
                    queue.push(k as u32);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The flow's own invariants, read from the search state: conservation at every node, residuals within their
    //! capacities, the flow equal to the capacity of the cut returned, and a rebuilt graph carrying nothing over. On
    //! 256 seeded random graphs each; the crate has no dev-dependency, so its generator is here
    //! (`verify/` holds the property tests against enumeration).

    use super::Graph;

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

    struct Input {
        source: Vec<f64>,
        sink: Vec<f64>,
        tails: Vec<u32>,
        heads: Vec<u32>,
        capacities: Vec<f64>,
    }

    impl Input {
        /// 2 to 40 nodes, up to 3 edges a node without self-loops, capacities on [0, 2).
        fn random(rng: &mut Rng) -> Input {
            let n = 2 + rng.below(39);
            let source = (0..n).map(|_| 2.0 * rng.uniform()).collect();
            let sink = (0..n).map(|_| 2.0 * rng.uniform()).collect();
            let (mut tails, mut heads, mut capacities) = (vec![], vec![], vec![]);
            for _ in 0..rng.below(3 * n + 1) {
                let u = rng.below(n);
                tails.push(u as u32);
                heads.push(((u + 1 + rng.below(n - 1)) % n) as u32);
                capacities.push(2.0 * rng.uniform());
            }
            Input {
                source,
                sink,
                tails,
                heads,
                capacities,
            }
        }

        fn build(&self, graph: &mut Graph) {
            graph.build(&self.source, &self.sink, &self.tails, &self.heads, &self.capacities);
        }

        fn cut_value(&self, sink_side: &[bool]) -> f64 {
            let mut value = 0.0;
            for i in 0..self.source.len() {
                value += if sink_side[i] { self.source[i] } else { self.sink[i] };
            }
            for e in 0..self.tails.len() {
                if !sink_side[self.tails[e] as usize] && sink_side[self.heads[e] as usize] {
                    value += self.capacities[e];
                }
            }
            value
        }
    }

    /// Fails where an augmentation moves flow it does not account for at a node, overdraws an arc or a terminal, or
    /// the flow total is not the capacity of the cut returned (max flow, min cut).
    #[test]
    fn the_flow_is_conserved_feasible_and_the_cut_s_capacity() {
        let mut rng = Rng(11);
        for _ in 0..256 {
            let input = Input::random(&mut rng);
            let (mut graph, mut start) = (Graph::default(), Graph::default());
            input.build(&mut graph);
            input.build(&mut start);
            let mut side = Vec::new();
            graph.solve(&mut side);
            let scale = 1.0
                + input
                    .capacities
                    .iter()
                    .chain(&input.source)
                    .chain(&input.sink)
                    .sum::<f64>();
            for (a, &r) in graph.residual.iter().enumerate() {
                assert!(r >= -1e-12 * scale, "arc {a} residual {r}");
                assert!(r <= start.residual[a] + start.residual[graph.sister[a] as usize] + 1e-12 * scale);
            }
            for i in 0..input.source.len() {
                let (t0, t) = (start.terminal[i], graph.terminal[i]);
                assert!(
                    t * t0 >= 0.0 && t.abs() <= t0.abs() + 1e-12 * scale,
                    "terminal {t0} -> {t}"
                );
                let out: f64 = graph.arcs(i).map(|a| start.residual[a] - graph.residual[a]).sum();
                assert!(
                    (t0 - t - out).abs() <= 1e-12 * scale,
                    "node {i} in {} out {out}",
                    t0 - t
                );
            }
            assert!((graph.flow - input.cut_value(&side)).abs() <= 1e-12 * scale);
        }
    }

    /// Fails where `build` or `solve` leaves state from the graph before.
    #[test]
    fn a_rebuilt_graph_solves_as_a_new_one() {
        let mut rng = Rng(13);
        for _ in 0..256 {
            let (first, second) = (Input::random(&mut rng), Input::random(&mut rng));
            let (mut reused, mut side) = (Graph::default(), Vec::new());
            first.build(&mut reused);
            reused.solve(&mut side);
            second.build(&mut reused);
            reused.solve(&mut side);
            let (mut fresh, mut fresh_side) = (Graph::default(), Vec::new());
            second.build(&mut fresh);
            fresh.solve(&mut fresh_side);
            assert_eq!(side, fresh_side);
            assert_eq!(reused.flow.to_bits(), fresh.flow.to_bits());
            assert_eq!(reused.residual, fresh.residual);
        }
    }
}
