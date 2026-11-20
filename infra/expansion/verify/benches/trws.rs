//! Wall time of TRW-S (T- #198) and of its first pass on planted square lattices at the gate size (3,000 x 4) and the
//! stress size (87,025 x 16), `benches/expansion.rs`' problems. Each problem is built outside the timed loop;
//! criterion warms each before timing. The ratio to OpenGM's `TRWSi` is `test_benchmarks.py`'s, where OpenGM runs.
//!
//!     cargo bench --manifest-path infra/expansion/verify/Cargo.toml --bench trws

#[path = "../tests/common/mod.rs"]
mod common;

use cnamaste_expansion::trws::trws;
use common::planted_lattice;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

/// `config.yaml`'s `hmrf.spatial_weight`.
const SPATIAL_WEIGHT: f64 = 0.75;

/// (name, rows, cols, states, signal).
const SIZES: [(&str, usize, usize, usize, f64); 2] =
    [("gate-3000x4", 50, 60, 4, 1.0), ("stress-87025x16", 295, 295, 16, 2.0)];

fn bench(c: &mut Criterion) {
    for (name, rows, cols, k, signal) in SIZES {
        let problem = planted_lattice(rows, cols, k, signal, SPATIAL_WEIGHT, 0);
        let mut group = c.benchmark_group(name);
        if rows * cols > 10_000 {
            group.sample_size(10);
        }
        group.bench_function("trws", |b| {
            b.iter(|| black_box(trws(&problem.view(), 1000, 1e-9).bound))
        });
        group.bench_function("first_pass", |b| {
            b.iter(|| black_box(trws(&problem.view(), 1, 1e-9).bound))
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
