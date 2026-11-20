# The simulated fixtures

One directory per fixture, `<name>_<hash>/`: the sample `run_cnamaste` reads through a sample sheet, with its
planted truth; and `capture/<name>_<hash>_capture.hdf5`, one run recorded stage by stage with its tables and
`/metrics`, which the suite replays (`tests/test_capture.py`). Each directory's `README.md` names its simulation, source and size. `cnamaste.qa.sims` resolves every path;
`CNAMASTE_SIMS` names a directory of whole samples instead, each under its simulation's name.

| Fixture | Directory | Simulation |
| --- | --- | --- |
| CalicoST easy (`2d4ce9a9`) | `easy_2d4ce9a9` | `numcnas1.2_cnasize5e7_ploidy2_random0` |
| CalicoST hard (`8797710b`) | `hard_8797710b` | `numcnas6.3_cnasize1e7_ploidy2_random0` |
| CalicoST harder (`a9a68913`) | `harder_a9a68913` | `numcnas3.3_cnasize1e7_ploidy2_random6` |
| CalicoST random4 (`81ebaea6`) | `random4_81ebaea6` | `numcnas3.3_cnasize5e7_ploidy2_random4` |
| dev_tree_easy r0 (`e4bbe03f`) | `dev_tree_easy_e4bbe03f` | `dev_tree_easy` |
| dev_tree_easy_square r0 (`bdff091f`) | `dev_tree_easy_square_bdff091f` | `dev_tree_easy_square` |
| dev_tree_1s_dense r0 (`33e3471e`) | `dev_tree_1s_dense_33e3471e` | `dev_tree_1s_dense` |

Only easy and hard are recaptured. The others keep the captures they were taken with; dev_tree_easy_square has none.

**Naming.** A CalicoST simulation is named `numcnas{shared}.{unique}_cnasize{size}_ploidy{ploidy}_random{r}`:
`shared` CNAs on every tumour clone, `unique` CNAs on each clone alone, each `size` base pairs long, at baseline
`ploidy`, realization `r` (`port`, `python/port/sim/fixtures.py`). Easy, `numcnas1.2_cnasize5e7`, is 1 shared and 2
unique 50 Mb events; hard, `numcnas6.3_cnasize1e7`, 6 shared and 3 unique 10 Mb events. A `port` draw is named by its
manifest, `sim/manifests/<name>.toml` in `port`.

**Layout.** A CalicoST sample holds the six inputs `load_input_data` reads (`barcodes.txt.gz`,
`unique_snp_ids.npy.gz`, `cell_snp_{A,B}allele.npz`, `filtered_feature_bc_matrix.h5ad`,
`spatial/tissue_positions_list.csv.gz`) and the planted truth (`truth_clone_labels.tsv.gz`, `truth_acn_profile.tsv.gz`).
A `port` draw keeps its own layout, which `capture.input_files` reads: one directory per slice (its
`filtered_feature_bc_matrix.h5ad` and its positions), a shared `snp/` (the other four), the
planted truth with `truth_phase.npy.gz` and `truth_tree.tsv.gz`, the draw's `manifest.json` (the per-draw overrides of
the one template rule, `capture.template`) and a `sample_sheet.tsv` naming the slices relative to the sample.

**Form.** Every sample is written by `python -m cnamaste.sandbox.store_sample SOURCE TARGET`, which reproduces easy's
files byte for byte from easy's own content: text and `.npy` gzipped at level 9 with no name and `mtime` 0, the
`.h5ad` by `AnnData.write_h5ad(compression="gzip")` (anndata 0.13.3, h5py 3.16.0), the `.npz` as written. Harder's
`.h5ad`, uncompressed at 10,262,700 bytes, is re-stored at 3,453,903, decoding to the same matrix, observations and
variables; its other files decode to the bytes it held. The `dev_tree` draws' `cell_snp_{A,B}allele.npz` are 5.9 MB
each, over `[tool.cnamaste.capture] max_file_bytes` (5,000,000, the staged files' cap), and committed as drawn.

**Hashes.** A fixture hash is `port.sim.fixtures.realization_hash` of the sample as committed with its `.h5ad` as first
written: it hashes each file's decoded bytes but an `.h5ad`'s stored ones. dev_tree_easy and dense reproduce their
manifests' `r0_hash`; dev_tree_easy_square's draw (`23acef50`) is stored with its square grid as Visium HD positions
(`store_sample.hd_positions`), `bdff091f`; harder `a9a68913`; random4's, `81ebaea6`, is named the same way. Hashed as stored, after the `.h5ad` re-store:
`7ff89533` (dev_tree_easy), `49d08dbe` (square), `8d120166` (dense), `cf75f9ee` (harder), `f8427855` (random4).

**Regeneration.** A draw: `python -m port.sim.draw sim/manifests/<name>.toml` in `port` (its own environment), check
`realization_hash` of `r0` against the manifest's `r0_hash`, then `store_sample <draw>/r0 <name>_<hash>`. A CalicoST sample:
`git archive 1074bbd sim/<name>`, then `store_sample`.

The suite holds every file to its capture: each input's sha256 is the one the capture recorded when it read it, and
each truth file decompresses to the bytes the capture stores. A file here that drifts from its capture fails; the
captures are the referee, not these copies. A file is therefore left exactly as its capture read it -- re-compressing
one, even to identical content, changes the bytes and fails.

**Running one.** Each fixture's `sample_sheet.tsv` names its inputs relative to the repository root, so a run starts
there: `zenodo_sim_config.yaml` with `tests/test_zenodo_sim_config.yaml` laid over it (easy), its `paths.output_dir`
created first. The GRCh38 references are not in a fixture: `ref/`, else `$CNAMASTE_GRCH38`.
`uv run pytest -m end2end tests/test_zenodo_sim.py` runs easy this way and scores its clones.
