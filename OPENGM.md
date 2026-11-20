# OpenGM: building it, and what reads it

**TL;DR:** a clone of OpenGM is made and `infra/opengm/build_opengm.sh` compiles
the necessary shared library. The default potts model solver `hmrf.label_solver` is
`expansion` (Rust, `infra/expansion`), so a default run does not need opengm; `opengm_expansion`,
`opengm_icm` and `opengm_trws` do. `trws` is `opengm_trws` ported to Rust (T- #198), bitwise.

`tests/test_opengm.py` tests for the necessary `$CNAMASTE_OPENGM_HOME`.

## Why there is a build at all

OpenGM is header-only C++ templates, with no **package index**.
The required pin for the commit is 
```
commit=decdacf4caad223b0ab5478d38a855f8767a394f
```

`infra/opengm/cnamaste_opengm.cxx` is the repository's interface: it exposes
OpenGM's inference over a flat C ABI; the min s-t cut the alpha expansion and
the swap need is Boost Graph's Boykov-Kolmogorov or the shipped `infra/opengm/dinic.cxx`.

## Preparation

** `$CNAMASTE_OPENGM_HOME` must be defined, exist and contain the require checkout,
```sh
export CNAMASTE_OPENGM_HOME=/some/dir          # your choice; keep it exported
mkdir "$CNAMASTE_OPENGM_HOME"
git clone https://github.com/opengm/opengm "$CNAMASTE_OPENGM_HOME/src"
git -C "$CNAMASTE_OPENGM_HOME/src" checkout decdacf4caad223b0ab5478d38a855f8767a394f
```

About 3s and 35 MB.

### Boost

The default cut, Boykov-Kolmogorov, is Boost Graph's, so the build needs
Boost's headers (any recent release; built and tested here against 1.83).
Nothing is compiled or linked from Boost: headers only.  Suggested strategy, at your own risk,

| platform | install | where the build looks |
| --- | --- | --- |
| Debian / Ubuntu | `sudo apt install libboost-dev` | `/usr/include`, on the compiler's path |
| macOS (Homebrew) | `brew install boost` | `export BOOST_INCLUDEDIR="$(brew --prefix boost)/include"` |
| conda | `conda install -c conda-forge boost-cpp` | `export BOOST_INCLUDEDIR="$CONDA_PREFIX/include"` |
| HPC module | `module load boost` | `export BOOST_INCLUDEDIR="$BOOST_ROOT/include"`, or the module's own variable |

`$BOOST_INCLUDEDIR` is needed only where the headers are off the compiler's default path.

## Build

```sh
infra/opengm/build_opengm.sh
```

It compiles the driver with `${CXX:-g++} -std=c++11 -O2 -shared -fPIC -w` against `$CNAMASTE_OPENGM_HOME/src/include`,
and writes four files beside the checkout:

| file | what |
| --- | --- |
| `libcnamaste_opengm.so` | the library `cnamaste.external` loads (986 KB here) |
| `COMMIT` | the OpenGM commit built |
| `LICENCES` | what was built, and what was deliberately not |
| `STAMP` | `commit + sha256 of the driver and dinic.cxx + compiler + flags + Boost version` |

The following errors are possible:

| condition | message |
| --- | --- |
| `$CNAMASTE_OPENGM_HOME` unset | `set $CNAMASTE_OPENGM_HOME to the directory holding OpenGM's checkout` |
| the directory does not exist | `… does not exist; create it and clone OpenGM into src/` |
| no headers under `src/include/opengm` | `no OpenGM headers at …; clone the pinned commit there` |
| `src/` at another commit | `… is at <sha>, not the pinned <sha>; check it out there` |
| no Boost headers | `no Boost headers: install them … or set $BOOST_INCLUDEDIR` |

## Using it

A run selects the solver by name:

```yaml
hmrf:
  # icm, opengm_icm, opengm_expansion, opengm_trws, expansion (the default, Rust) or trws (Rust)
  label_solver : opengm_expansion
  # boykov_kolmogorov (where unset) or dinic: the min cut of each expansion or swap move
  opengm_mincut_solver : boykov_kolmogorov
```

`icm` is cnamaste's own sweep and needs no library; every `opengm_*` name
goes through `opengm.external_assignment` on the same Potts energy. Directly:

```python
from cnamaste import external
from cnamaste.potts import potts_problem

problem = potts_problem(single_llf, adjacency, spatial_weight)

with external.session() as session:          # one worker, many solves
    run = session.solve(problem, "expansion")                 # mincut="boykov_kolmogorov"
    run = session.solve(problem, "expansion", mincut="dinic")

external.solve(problem, "trws")              # one-shot; pays the start-up
```

`OPENGM_ALGORITHMS` names them: `icm`, `lbp`, `astar`, `trws`, `dd`, `expansion`,
`swap`. An `OpenGMResult` carries `labels`, `energy`, `bound` (`-inf` where the
algorithm gives none), the per-iteration `trace` and `iterations`.

**Its energy is the problem's**, so it compares with `potts.potts_energy`
and `potts.pairwise_potts_energy` with no conversion. On a 60-node, 3-label,
272-pair problem `expansion` returns `-83.60486` in 6 iterations, and
`potts_energy` of those labels is `-83.60486`.

## How it runs

Not in this process. `python -m cnamaste.external_worker <library>` is a
subprocess that `ctypes`-loads the `.so` once and serves one request per
line, so the native code stays out of the run's process and a session pays
the interpreter's start-up once rather than per call.

Arrays cross as **memory-mapped files** in the session's temporary directory.
The parent creates every file, the outputs included, and unlinks each when
its call returns or fails; the worker maps and never unlinks, so a worker
that dies leaves nothing behind.