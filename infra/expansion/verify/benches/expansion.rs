//! Wall time of the expansion and of its first move (label 0: the graph built and cut) on planted square lattices at
//! the gate size (3,000 x 4) and the stress size (87,025 x 16, a Visium HD 16 um sample's ~87,000 bins),
//! `test_benchmarks.py`'s signals and the shipped `hmrf.spatial_weight`. Each problem is built outside the timed
//! loop; criterion warms each before timing. The ratio to OpenGM is `test_benchmarks.py`'s, where OpenGM runs.
//!
//!     cargo bench --manifest-path infra/expansion/verify/Cargo.toml --bench expansion

#[path = "../tests/common/mod.rs"]
mod common;

use cnamaste_expansion::expansion::expand;
use common::{argmin_labels, planted_lattice};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;

/// `config.yaml`'s `hmrf.spatial_weight`.
const SPATIAL_WEIGHT: f64 = 0.75;

/// (name, rows, cols, states, signal).
const SIZES: [(&str, usize, usize, usize, f64); 2] =
    [("gate-3000x4", 50, 60, 4, 1.0), ("stress-87025x16", 295, 295, 16, 2.0)];

fn bench(c: &mut Criterion) {
    for (name, rows, cols, k, signal) in SIZES {
        let problem = planted_lattice(rows, cols, k, signal, SPATIAL_WEIGHT, 0);
        let start = argmin_labels(&problem);
        let mut group = c.benchmark_group(name);
        if rows * cols > 10_000 {
            group.sample_size(10);
        }
        group.bench_function("expand", |b| {
            b.iter_batched_ref(
                || start.clone(),
                |labels| black_box(expand(&problem.view(), labels, 1000)),
                BatchSize::LargeInput,
            )
        });
        group.bench_function("one_move", |b| {
            b.iter_batched_ref(
                || start.clone(),
                |labels| black_box(expand(&problem.view(), labels, 1)),
                BatchSize::LargeInput,
            )
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
