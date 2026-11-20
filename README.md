# cna-maste

Allele-specific copy-number aberrations and clone inference from spatial transcriptomics.

```sh
uv sync --locked --all-extras
mkdir -p results/easy_2d4ce9a9
uv run run_cnamaste zenodo_sim_config.yaml \
    -o paths.sample_sheet=sim/easy_2d4ce9a9/sample_sheet.tsv \
    -o paths.output_dir=results/easy_2d4ce9a9 \
    -o references.geneticmap_file=ref/genetic_map_GRCh38_merged.tab.gz \
    -o references.hgtable_file=ref/hgTables_hg38_gencode.txt \
    -o references.filtergenelist_file=ref/ig_gene_list.txt \
    -o references.filterregion_file=ref/HLA_regions.bed
```

runs CalicoST easy (`2d4ce9a9`), a simulated sample committed under `sim/`, from the repository root,
after the OpenGM build below.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root, `uv sync --locked --all-extras`.

The default label solver, `hmrf.label_solver: expansion`, is cnamaste's alpha expansion, OpenGM's moves with a
Boykov-Kolmogorov cut in Rust (`infra/expansion`, no crate dependencies); `cargo` builds it once at
`$CNAMASTE_EXPANSION_HOME`:

```sh
mkdir -p "$CNAMASTE_EXPANSION_HOME" && infra/expansion/build_expansion.sh
```

`hmrf.label_solver: trws` is cnamaste's TRW-S, OpenGM's `TRWSi` (`opengm_trws`) ported to Rust in the same library
(T- #198): the same labels, energy, bound and bound trace, bitwise.

OpenGM's moves leave out a pair labelled apart off alpha (T- #195); `hmrf.opengm_patch: true` (the default) corrects
them in `expansion`, false keeps OpenGM's, bitwise. OpenGM's solvers (`opengm_expansion`, `opengm_icm`, `opengm_trws`)
need its library, built once at `$CNAMASTE_OPENGM_HOME`; see [`OPENGM.md`](OPENGM.md). The Rust tests and benchmarks
are `infra/expansion/verify/`:

```sh
cargo test --release --manifest-path infra/expansion/Cargo.toml
cargo test --release --manifest-path infra/expansion/verify/Cargo.toml
```

`ref/` holds four GRCh38 references a run requires.

## Preprocessing
Refer to [Universal-Genotyping-Pipeline](https://github.com/raphael-group/Universal-Genotyping-Pipeline/blob/main/docs/spatial_genotyping.md) to prepare the input data for `cnamaste`.

## Run

```sh
uv run run_cnamaste CONFIG [-o key=value ...]
```

Every key and its default is in [`python/cnamaste/config/config_defaults.yaml`](python/cnamaste/config/config_defaults.yaml).
`CONFIG` is a YAML file of overrides of it (`zenodo_sim_config.yaml` is the one shipped for the simulated samples,
`config.yaml` for Visium HD), and each `-o` overrides one key, `section.key=value`. A section or key the defaults do
not carry is refused. `paths`, `references` and `hmrf.lattice_type` have no default; `paths.output_dir` must exist
for a run.

### The sample sheet

`paths.sample_sheet` names a tab-separated sheet, one row per slice:

| Column | Meaning |
| --- | --- |
| `bam` | The slice's BAM; unread by `run_cnamaste`, `NULL` where absent |
| `sample_id` | The slice's name, carried into the outputs' `sample_id` |
| `spaceranger_dir` | The directory holding `filtered_feature_bc_matrix.h5ad` and `spatial/` |
| `snp_dir` | The directory holding `barcodes.txt`, `unique_snp_ids.npy` and `cell_snp_{A,B}allele.npz` |

Relative paths resolve from the directory the run starts in, e.g. each `sim/<name>_<hash>/sample_sheet.tsv`
is written for the repository root.

### Outputs

`paths.output_dir/run_<id>/`, where `<id>` is the hash of the configuration (`config_<id>.yaml`):

| File | Holds |
| --- | --- |
| `config_<id>.yaml` | The configuration the run read, with every override `-o` applied |
| `cnamaste_<id>.log` | The run's log, including each model fit's size, iterations, runtime and log-likelihood |
| `clone_labels.tsv.gz` | Per spot: `barcode`, `sample_id`, `x`, `y`, `clone_label`, `normal_candidate` |
| `clone_labels_int.tsv.gz` | The same after the integer clone merge, with `field_argmax` |
| `baf_clone_labels.tsv.gz` | The clones of the BAF-only stage |
| `cnv_<ploidy>_genelevel.tsv` | Per gene and clone: `A` and `B` |
| `cnv_<ploidy>_seglevel.tsv` | Per segment (`CHR`, `START`, `END`) and clone: the HMM state `k`, `logmu`, `p`, copy numbers `A` and `B` |
| `cnv_<ploidy>_perstate.tsv` | Per HMM state and clone: `logmu`, `p` |
| `segmentation.tsv.gz` | Per reference gene expressed in the data (chr1–22): contig, start, end, one column per segmentation level, and the phase |
| `rdrbaf_final_nstates7_smp.npz` | The final RDR-BAF fit: state parameters, `log_gamma`, `pred_cnv`, assignments, `total_llf` |
| `plots/` | PDFs: the clones in space and along the genome per stage, the copy-number profile, `combined.pdf` |

Beside it, `paths.output_dir/tmp/opengm-*/` keeps each OpenGM session's blocks; nothing reads them after the run.

### Decoding a run again

```bash
uv run run_cnamaste_decode CONFIG [-o ab_decode.KEY=VALUE ...] [-o paths.output_dir=DIR]
```

### Plotting a run again

```bash
uv run run_cnamaste_plot CONFIG [-o run.plot_field_argmax=BOOL] [-o paths.output_dir=DIR]
```

Redraws the finalize figures of the run CONFIG names into its `plots/`, once its tables reproduce unchanged.

### Zenodo simulation suite assessment

```sh 
uv run run_sim_qa --root ROOT [--config CONFIG [-o KEY=VALUE ...]] [--baseline_solver calicost|SOLVER-INIT-DECODER] [--solver SOLVER-INIT-DECODER] [--one_shot|--all_shot] [--calicost_ploidy PLOIDY] [--overwrite]
```

## Test

```sh
CNAMASTE_OPENGM_HOME=... CNAMASTE_EXPANSION_HOME=... uv run --locked pytest   # standard: not end2end, benchmark, calicost or calicost_departure
uv run --locked pytest -m "calicost or calicost_departure" [--calicost-live | --calicost-refresh]
uv run --locked pytest -m end2end tests/test_run_sim_qa_cycle.py
```

`slow` (over 5 s) and `very_slow` (over 20 s) mark measured rows in any tier; `-m "not slow"` deselects them. The
CalicoST referee is recorded in `tests/calicost/.cache/` (`tests/calicost/calicost_cache.py`): `--calicost-live`
calls CalicoST and fails where the cache differs, `--calicost-refresh` rewrites it.

## Format

```sh
uv run ruff check --select I --fix python tests && uv run ruff format python tests
```

- `pyproject.toml`'s `[tool.ruff]`: lines to 120, imports sorted with `as` imports combined. `# fmt: skip` keeps a
  line's own layout. ruff is in the `dev` group, pinned.
