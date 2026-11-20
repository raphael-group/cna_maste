# Benchmark

`benchmark.yaml` records, per sample, the files each method's labels and clone copy-number profiles are read from.
`scripts/run_annotation_page.py` draws them on one page with `cnamaste.plot.layout.plot_annotation_page`:

- (a) spatial panels, two wide: the H&E slide, each spot's cell type (epithelial or not, low-confidence spots left out),
  then the clones of Numbat, titled by the side of the bins it ran on, CalicoST and cnamaste, each keyed with its
  spots per label;
- (b) with `--cnamaste_binned`, each cnamaste clone's pseudobulk RDR and BAF per bin on
  [cnplot](https://github.com/raphael-group/cnplot)'s genome axis, by cnplot's `plot_scatter_1d`, each bin in its
  decoded `(A, B)`'s colour on cnplot's profile (`get_cn_cmap`), so (c)'s key keys it, and the clone's expected RDR
  and BAF over them: the tracks of cnamaste's own `combined.pdf` (`plot_clones_genomic`), drawn by cnplot;
- (c) each method's clone profile on the same axis, centromeres dotted: Numbat's CNV state per segment, CalicoST's
  and cnamaste's integer copies. Without `--cnamaste_binned` the profiles are (b).

The expected lines are cnamaste's (`pseudobulk.get_clone_expectation`): with a clone's decoded normal fraction f_c,
RDR `(f_c + (1 - f_c)(A + B) / 2) / Z_c` and BAF `(f_c + (1 - f_c) B) / (2 f_c + (1 - f_c)(A + B))`, with its phase
flip `1 - BAF`, `Z_c = exp(new_log_mu_shift_c)` the fit's (`rdrbaf_final_nstates<n>_smp.npz`); without f_c, the
fitted state's `exp(logmu) / Z_c` and `p`. `run_cnamaste` writes f_c to no table: pass it as
`--cnamaste_normal_fraction <clone>=<f>`, read from the run's `combined.png` key, `Clone I (<spots>%, <f_c>%)`.

## Inputs

| Method | Labels | Profile |
| --- | --- | --- |
| Numbat | `clone_post_<i>.tsv`, the last iteration, 32 um bins given to the 16 um bins inside each | `bulk_clones_final.tsv.gz` |
| CalicoST | `clone_labels.tsv` of the best rectangle by `total_llf` | `cnv_seglevel.tsv`, unconstrained ploidy |
| cnamaste | `clone_labels_int.tsv.gz` | `cnv_seglevel.tsv`; (b) `binned_data.npz`, `clone_labels.tsv.gz` and `rdrbaf_final_nstates<n>_smp.npz` beside it |
| cell type | `spot_annotation.tsv` | |

The command takes Numbat's and CalicoST's run directories and finds these files in them. The H&E image and the
bins' places come from a Space Ranger directory of the sample, any bin side. The genome axis is
`data/hg38.regions.bed` and `data/hg38.chrom.sizes`, copied from Universal-Genotyping-Pipeline's `resources/data/`.
Numbat's files are read by `scripts/read_numbat.py`.

## Run

From the repository root, after `uv sync --locked --all-extras`:

```sh
B=/Users/runpengluo/Research/proj-copy_number_variation/Copytyping-dev/analysis_data/visium_hd_3prime
uv run python python/cnamaste/sandbox/luo/benchmark/scripts/run_annotation_page.py \
    --spaceranger_dir $B/output_10092026/H1-N3KXRGQ/bb-032um \
    --celltype $B/celltyping/H1-N3KXRGQ/celltyping-16um-all_spots-10x_graphclust_panglaodb_ulm/spot_annotation.tsv \
    --cnamaste_labels $B/cnamaste/N3KXRGQ-clone4_rectangle0_w0.8_ugp/clone_labels_int.tsv.gz \
    --cnamaste_seglevel $B/cnamaste/N3KXRGQ-clone4_rectangle0_w0.8_ugp/cnv_seglevel.tsv \
    --cnamaste_binned $B/cnamaste/N3KXRGQ-clone4_rectangle0_w0.8_ugp/binned_data.npz \
    --cnamaste_normal_fraction 0=1.0 --cnamaste_normal_fraction 1=0.38 \
    --calicost_dir $B/calicost/H1-N3KXRGQ_016um \
    --numbat_dir $B/numbat/H1-N3KXRGQ-numbat-max_entropy_08
```

`-o` names the page, `.pdf` or `.png`, under `--outdir`; by default `python/cnamaste/sandbox/luo/benchmark/output/combined.pdf`. The run writes
`binned_data.npz` where `run.write_binned_data` is set. Options, each at its default above:

| Option | Default | Meaning |
| --- | --- | --- |
| `--outdir` | `python/cnamaste/sandbox/luo/benchmark/output` | The directory the page is written under, gitignored. |
| `-o` | `combined.pdf` | The page's name under `--outdir`. |
| `--side_um` | 16 | The side of the bins drawn; Numbat's coarser bins are given to each such bin inside them. |
| `--calicost_ploidy` | `unconstrained` | CalicoST's table: `cnv<_ploidy>_seglevel.tsv`. |
| `--region_bed` | `python/cnamaste/sandbox/luo/benchmark/data/hg38.regions.bed` | The regions the genome axis draws, a chromosome's arms. |
| `--chrom_sizes` | `python/cnamaste/sandbox/luo/benchmark/data/hg38.chrom.sizes` | The chromosome lengths of the genome axis. |

## UMI distributions

`scripts/plot_umi_distribution.py` draws a sample's per-spot total UMI (its `bb` AnnData) and SNP-covering UMI (A + B
of its `multi_snp` allele matrices, bins by spots), one PDF each, titled `<sample> (<side> um)` over
`#in-tissue spots=<n>`.
UGP's 16 um bins are in `bb/`, its 32 um bins in `bb-032um/`:

```sh
H=$B/output_10092026/H1-N3KXRGQ
uv run python python/cnamaste/sandbox/luo/benchmark/scripts/plot_umi_distribution.py --sample H1-N3KXRGQ --side_um 16 \
    --h5ad $H/bb/VISIUM3prime.h5ad \
    --aallele $H/bb/multi_snp/VISIUM3prime/bb.Aallele.npz \
    --ballele $H/bb/multi_snp/VISIUM3prime/bb.Ballele.npz \
    --outdir python/cnamaste/sandbox/luo/benchmark/output/umi_distribution/H1-N3KXRGQ_016um
uv run python python/cnamaste/sandbox/luo/benchmark/scripts/plot_umi_distribution.py --sample H1-N3KXRGQ --side_um 32 \
    --h5ad $H/bb-032um/VISIUM3prime.h5ad \
    --aallele $H/bb-032um/multi_snp/VISIUM3prime/bb.Aallele.npz \
    --ballele $H/bb-032um/multi_snp/VISIUM3prime/bb.Ballele.npz \
    --outdir python/cnamaste/sandbox/luo/benchmark/output/umi_distribution/H1-N3KXRGQ_032um
```

Each writes `spot_total_umi_histplot.pdf` and `spot_snp_umi_histplot.pdf` under `--outdir`, `python/cnamaste/sandbox/luo/benchmark/output` by
default.

## Supplement

`scripts/plot_supplement.py` writes the supplement, one figure per letter page, `python/cnamaste/sandbox/luo/benchmark/output/supplement.pdf` by
default. Page 1: one row per `--bb_dir <side_um>=<dir>`, in the order given and lettered (a), (b), its total UMI left
and SNP-covering UMI right.

```sh
uv run python python/cnamaste/sandbox/luo/benchmark/scripts/plot_supplement.py --sample H1-N3KXRGQ \
    --bb_dir 16=$H/bb --bb_dir 32=$H/bb-032um
```
