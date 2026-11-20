import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import yaml
from numba import njit

from cnamaste import expansion, external
from cnamaste.config import (
    YAMLConfig,
    label_solver_type,
    set_global_config,
    start_time,
)
from cnamaste.fit_checks import reset_fit_check_counts
from cnamaste.inference import (
    run_baf_clone_inference,
    run_baf_rdr_clone_inference,
    run_bin_segmentation,
    run_block_segmentation,
    run_gene_snp_table,
    run_initial_phasing,
    run_input_reading,
    run_integer_copy_decoding,
    run_normal_candidates,
    run_rdr_bin_segmentation,
)
from cnamaste.logger import get_logger
from cnamaste.results import (
    configure_output_dir,
    existing_fit,
    get_run_dir,
    get_use_existing_hours,
    run_config_name,
    run_id,
    run_log_name,
)


@njit
def set_numba_seed(value):
    np.random.seed(value)


logger = get_logger("cnamaste", start_time=start_time)


def run_cnamaste(config_path, over_rides=None, use_existing=False):
    logger.runtime_phase = "prep."
    logger.info("----  Welcome to cna-maste  ----")

    config = YAMLConfig.from_file(config_path)
    config.over_ride(over_rides)
    config.issue_warnings()
    config.check_paths()

    # NB stop if opengm requested and not defined.
    external.require(label_solver_type(config))
    expansion.require(label_solver_type(config))

    if config.ab_decode.decoder == "hmm_ab" and config.ab_decode.solver == "expansion":
        external.require("opengm_expansion")

    run_dir = Path(get_run_dir(config))
    run_dir.mkdir(exist_ok=True)
    with open(run_dir / run_config_name(run_id(config)), "w") as out:
        yaml.safe_dump(config.to_dict(), out, sort_keys=False)

    if use_existing and (fit := existing_fit(config)) is not None:
        logger.warning(
            f"--use_existing: {fit} written within {get_use_existing_hours()} h; the run is kept, not run again."
        )
        return

    handler = logging.FileHandler(run_dir / run_log_name(run_id(config)), mode="w")
    handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(handler)

    logger.info(f"Read configuration:\n{config}")
    logger.info(f"Wrote the resolved configuration to {run_dir / run_config_name(run_id(config))}")

    set_global_config(config)
    reset_fit_check_counts()

    logger.info(f"Set (numpy) random seed={config.hmrf.random_seed}")

    # TODO' fix reproducibility - set random seed globally.
    np.random.seed(int(config.hmrf.random_seed))
    random.seed(int(config.hmrf.random_seed))
    set_numba_seed(int(config.hmrf.random_seed))

    output_dir, plots_dir = configure_output_dir(config)

    # NB 00_inputs: the spots, genes, SNPs and positions; spot_umis.pdf.
    inputs = run_input_reading(config, plots_dir)

    # NB 01_genes: the gene and SNP table.
    df_gene_snp = run_gene_snp_table(inputs, config)

    # NB 02_blocks: the SNPs grouped into haplotype blocks, with their counts.
    block_segmentation = run_block_segmentation(inputs, df_gene_snp, config)

    # NB 03_phasing: the initial clones and each block's phase.
    initial_phasing = run_initial_phasing(inputs, block_segmentation, config, plots_dir)

    # NB 04_bins: the phased blocks merged into bins, their counts and the spot adjacency.
    df_gene_snp, bin_counts, adjacency_mat = run_bin_segmentation(
        inputs, block_segmentation, initial_phasing, config, plots_dir
    )

    # NB 05_baf: the BAF-only clones and copy states.
    baf_fit = run_baf_clone_inference(inputs, bin_counts, adjacency_mat, initial_phasing, config, plots_dir)

    # NB 06_normal: the normal spots, and the bins and genes they filter.
    df_gene_snp, bin_counts, normal_candidate = run_normal_candidates(
        inputs, baf_fit, df_gene_snp, block_segmentation.tracked_segmentations, config, output_dir
    )

    # NB 07_rebin: the bins merged to the normal spots' floors, their counts and normal baseline.
    df_gene_snp, rdr_bin_counts, df_bin_info = run_rdr_bin_segmentation(
        inputs, df_gene_snp, bin_counts, normal_candidate, block_segmentation.tracked_segmentations, config
    )

    # NB 08_rdr: the BAF and read-depth clones and copy states.
    baf_rdr_fit = run_baf_rdr_clone_inference(
        inputs,
        rdr_bin_counts,
        adjacency_mat,
        baf_fit.merged_res,
        normal_candidate,
        initial_phasing.known_clone_assignment,
        config,
        output_dir,
        plots_dir,
    )

    # NB 09_outputs: the integer copies, their tables and the final plots.
    run_integer_copy_decoding(
        inputs,
        rdr_bin_counts,
        baf_rdr_fit,
        df_gene_snp,
        df_bin_info,
        normal_candidate,
        block_segmentation,
        initial_phasing,
        config,
        output_dir,
        plots_dir,
    )

    logger.info(f"Done in {(time.time() - start_time) / 60.0:.2f} minutes.")

    logger.removeHandler(handler)
    handler.close()


# NB run_cnamaste zenodo_sim_config (-o paths.sample_sheet='dummy_sample_sheet.tsv')
def main():
    parser = argparse.ArgumentParser(description="cna-maste pipeline")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--over_rides",
        "-o",
        action="append",
        default=[],
        help="Override configuration keys with dot notation, e.g. -o paths.sample_sheet=/path/to/sheet.csv",
    )
    parser.add_argument(
        "--use_existing",
        action="store_true",
        help=f"keep the run this configuration names where its final fit was written within {get_use_existing_hours()} h",
    )

    args = parser.parse_args()

    run_cnamaste(args.config_path, over_rides=args.over_rides, use_existing=args.use_existing)


if __name__ == "__main__":
    main()
