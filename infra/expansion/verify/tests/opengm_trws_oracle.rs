//! `trws` against OpenGM's TRW-S (`TRWSi`) on the cases `write_rust_trws_oracle.py` writes (T- #198): labels, Potts
//! energy, bound and the bound trace, bitwise. The files are written on demand, not committed, so this test is
//! ignored by default:
//!
//!     CNAMASTE_OPENGM_HOME=... uv run python infra/expansion/write_rust_trws_oracle.py [--stress]
//!     cargo test --release --manifest-path infra/expansion/verify/Cargo.toml --test opengm_trws_oracle -- --include-ignored
//!
//! `CNAMASTE_TRWS_ORACLE` names another directory than `infra/expansion/oracle/trws/`.

use std::fs;
use std::path::PathBuf;

use cnamaste_expansion::expansion::Problem;
use cnamaste_expansion::trws::trws;

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

fn oracle_files() -> Vec<PathBuf> {
    let dir = std::env::var_os("CNAMASTE_TRWS_ORACLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../oracle/trws"));
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|_| panic!("no oracle at {}: run write_rust_trws_oracle.py", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no oracle files in {}: run write_rust_trws_oracle.py",
        dir.display()
    );
    files
}

/// Fails where the chains, a message, the averaging, the rounding or a stop departs from OpenGM's `TRWSi`.
#[test]
#[ignore = "needs write_rust_trws_oracle.py's files"]
fn trws_is_opengm_s_trws() {
    for path in oracle_files() {
        let mut r = Reader {
            bytes: fs::read(&path).unwrap(),
            at: 0,
        };
        assert_eq!(r.take(4), b"CNXT", "{}: not a TRW-S oracle file", path.display());
        let h = r.i64s(6);
        assert_eq!(h[0], 1, "{}: oracle version {}", path.display(), h[0]);
        let (n, k, m, max_iterations, n_trace) = (
            h[1] as usize,
            h[2] as usize,
            h[3] as usize,
            h[4] as usize,
            h[5] as usize,
        );
        let tolerance = r.f64s(1)[0];
        let unary = r.f64s(n * k);
        let (first, second) = (r.indices(m), r.indices(m));
        let coupling = r.f64s(m);
        let labels = r.indices(n);
        let (energy, bound) = {
            let v = r.f64s(2);
            (v[0], v[1])
        };
        let trace = r.f64s(n_trace);
        assert_eq!(r.at, r.bytes.len(), "{}: trailing bytes", path.display());
        let problem = Problem {
            n_states: k,
            unary: &unary,
            first: &first,
            second: &second,
            coupling: &coupling,
        };
        let found = trws(&problem, max_iterations, tolerance);
        let name = path.file_stem().unwrap().to_string_lossy();
        assert_eq!(found.labels, labels, "{name}: labels");
        assert_eq!(
            problem.potts_energy(&found.labels).to_bits(),
            energy.to_bits(),
            "{name}: energy"
        );
        assert_eq!(
            found.bound.to_bits(),
            bound.to_bits(),
            "{name}: {} against {bound}",
            found.bound
        );
        assert_eq!(
            found.trace[..found.trace.len().min(max_iterations)],
            trace[..],
            "{name}: trace"
        );
        println!(
            "{name}: {n} nodes x {k} states, {} passes, gap {:.1e}",
            trace.len(),
            energy - bound
        );
    }
}
