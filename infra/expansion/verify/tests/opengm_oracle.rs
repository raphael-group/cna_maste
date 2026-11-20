//! `expand` against OpenGM's expansion (Boykov-Kolmogorov) on the cases `write_rust_expansion_oracle.py` writes:
//! labels, Potts energy and moves bitwise; on a field of few values (`energy_only`), the same energy to 1e-9 relative,
//! `tests/test_expansion.py`'s stated difference. The files are written on demand, not committed, so these tests are
//! ignored by default:
//!
//!     CNAMASTE_OPENGM_HOME=... uv run python infra/expansion/write_rust_expansion_oracle.py [--stress]
//!     cargo test --release --manifest-path infra/expansion/verify/Cargo.toml --test opengm_oracle -- --include-ignored
//!
//! `CNAMASTE_EXPANSION_ORACLE` names another directory than `infra/expansion/oracle/`.

use std::fs;
use std::path::PathBuf;

use cnamaste_expansion::expansion::{expand, Problem};

struct Case {
    n_states: usize,
    max_steps: usize,
    energy_only: bool,
    unary: Vec<f64>,
    first: Vec<usize>,
    second: Vec<usize>,
    coupling: Vec<f64>,
    start: Vec<usize>,
    labels: Vec<usize>,
    energy: f64,
    steps: usize,
}

struct Reader {
    bytes: Vec<u8>,
    at: usize,
}

impl Reader {
    fn take(&mut self, n: usize) -> &[u8] {
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        out
    }

    fn i64s(&mut self, n: usize) -> Vec<i64> {
        self.take(8 * n)
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn f64s(&mut self, n: usize) -> Vec<f64> {
        self.take(8 * n)
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn indices(&mut self, n: usize) -> Vec<usize> {
        self.i64s(n).into_iter().map(|v| usize::try_from(v).unwrap()).collect()
    }
}

fn read(path: &PathBuf) -> Case {
    let mut r = Reader {
        bytes: fs::read(path).unwrap(),
        at: 0,
    };
    assert_eq!(r.take(4), b"CNXO", "{}: not an oracle file", path.display());
    let h = r.i64s(6);
    assert_eq!(h[0], 1, "{}: oracle version {}", path.display(), h[0]);
    let (n, k, m) = (h[1] as usize, h[2] as usize, h[3] as usize);
    let case = Case {
        n_states: k,
        max_steps: h[4] as usize,
        energy_only: h[5] != 0,
        unary: r.f64s(n * k),
        first: r.indices(m),
        second: r.indices(m),
        coupling: r.f64s(m),
        start: r.indices(n),
        labels: r.indices(n),
        energy: r.f64s(1)[0],
        steps: r.indices(1)[0],
    };
    assert_eq!(r.at, r.bytes.len(), "{}: trailing bytes", path.display());
    case
}

fn oracle_files() -> Vec<PathBuf> {
    let dir = std::env::var_os("CNAMASTE_EXPANSION_ORACLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../oracle"));
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|_| panic!("no oracle at {}: run write_rust_expansion_oracle.py", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no oracle files in {}: run write_rust_expansion_oracle.py",
        dir.display()
    );
    files
}

/// Fails where a capacity, the label order or the stop departs from OpenGM's.
#[test]
#[ignore = "needs write_rust_expansion_oracle.py's files"]
fn expand_is_opengm_s_expansion() {
    for path in oracle_files() {
        let case = read(&path);
        let problem = Problem {
            n_states: case.n_states,
            unary: &case.unary,
            first: &case.first,
            second: &case.second,
            coupling: &case.coupling,
        };
        let mut labels = case.start.clone();
        let steps = expand(&problem, &mut labels, case.max_steps);
        let energy = problem.potts_energy(&labels);
        let name = path.file_stem().unwrap().to_string_lossy();
        if case.energy_only {
            assert!(
                (energy - case.energy).abs() <= 1e-9 * case.energy.abs(),
                "{name}: {energy} against {}",
                case.energy
            );
        } else {
            assert_eq!(labels, case.labels, "{name}: labels");
            assert_eq!(
                energy.to_bits(),
                case.energy.to_bits(),
                "{name}: {energy} against {}",
                case.energy
            );
            assert_eq!(steps, case.steps, "{name}: moves");
        }
        println!(
            "{name}: {} nodes x {} states, {steps} moves, energy {energy:.6e}",
            labels.len(),
            case.n_states
        );
    }
}
