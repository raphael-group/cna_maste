"""The run's stages, one function per capture group, in the order `scripts.run_cnamaste` calls them:
`run_input_reading` (`00_inputs`), `run_gene_snp_table` (`01_genes`), `run_block_segmentation` (`02_blocks`),
`run_initial_phasing` (`03_phasing`), `run_bin_segmentation` (`04_bins`), `run_baf_clone_inference` (`05_baf`),
`run_normal_candidates` (`06_normal`), `run_rdr_bin_segmentation` (`07_rebin`), `run_baf_rdr_clone_inference`
(`08_rdr`) and `run_integer_copy_decoding` (`09_outputs`)."""

import copy
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from cnamaste.ab_decode.decode import decode_integer_copies
from cnamaste.ab_decode.model import get_merged_ab_decoded_clones
from cnamaste.annotation import load_clone_labels
from cnamaste.calicost.adapters import calicost_diploid_decoder
from cnamaste.config import (
    assumed_ploidy,
    coordination_num,
    lattice_type,
    start_time,
)
from cnamaste.emission import CnaEmissionConfounders
from cnamaste.he import get_he_frame, he_initial_partition
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmrf import (
    field_argmax,
    get_labelling_unary,
    get_merged_fit,
    get_reindexed_fit,
    run_hmrf,
)
from cnamaste.hmrf_fit import HmrfFit
from cnamaste.hmrf_utils import get_clone_assignment, get_clone_indices
from cnamaste.initialize.genomic import get_initial_parameters
from cnamaste.initialize.spatial import SpotPositions, get_initial_labelling
from cnamaste.io import get_assignment_frame, get_sample_list, get_sample_sheet, get_spatial_positions, load_input_data
from cnamaste.logger import get_logger
from cnamaste.normal_spot import (
    determine_normal_baseline,
    determine_normal_candidates,
    filter_normal_diffexp,
    normal_baf_bin_filter,
)
from cnamaste.omics import (
    assign_initial_blocks,
    binned_gene_snp,
    create_bin_ranges,
    form_gene_snp_table,
    summarize_counts_for_bins,
    summarize_counts_for_blocks,
)
from cnamaste.phasing import cumulative_phase_switches, initial_phase_given_partition
from cnamaste.plot.layout import fitted_states_real, he_colours, plot_combined_page, plot_he_page, plot_spot_umi_page
from cnamaste.plot.plot_clones_spatial import plot_clones_spatial
from cnamaste.plot.plot_copy_number_profile import plot_copy_number_profile
from cnamaste.plot.plot_genomic import plot_clones_genomic
from cnamaste.quality_cuts import zero_genes
from cnamaste.recomb import get_sitewise_transmat
from cnamaste.results import (
    get_integer_copy_tables,
    get_output_file_meta,
    get_output_path_prefix,
)
from cnamaste.segments import (
    TrackedSegmentation,
    get_dropped,
    phasing_segments,
)
from cnamaste.spatial import construct_multislice_lattice_adjacency
from cnamaste.utils import (
    pause,
)

logger = get_logger("cnamaste", start_time=start_time)


class Inputs(NamedTuple):
    """The spots, genes and SNPs `load_input_data` reads, and the spots' samples and positions."""

    coords: np.ndarray
    barcodes: Any
    adata: Any
    exp_counts: Any
    cell_snp_Aallele: Any
    cell_snp_Ballele: Any
    unique_snp_ids: Any
    across_slice_adjacency_mat: Any
    physical: Any
    sample_list: Any
    sample_ids: Any
    positions: SpotPositions


class BlockSegmentation(NamedTuple):
    """The gene and SNP table with `block_id`, its segmentations, SNPs, block counts and phase transitions."""

    df_gene_snp: pd.DataFrame
    tracked_segmentations: TrackedSegmentation
    snps: Any
    counts: Any
    log_sitewise_transmat: Any


class InitialPhasing(NamedTuple):
    """The initial clones (spot indices per clone), the annotated clones or None, the H&E labels and slide frame or
    None, each block's phase, the phase segments' lengths and each gene's phase."""

    initial_clone_index: list[np.ndarray]
    known_clone_assignment: Any
    he_assignment: np.ndarray | None
    he_frame: Any
    phase_indicator: np.ndarray
    refined_lengths: np.ndarray
    gene_phase: np.ndarray


class BafFit(NamedTuple):
    """The BAF-only stage's counts (read depth zeroed), the read depth it zeroed, its merged fit and each merged
    clone's model B fraction per bin."""

    counts: Any
    umis: Any
    merged_res: HmrfFit
    merged_baf_profiles: np.ndarray


def run_input_reading(config: Any, plots_dir: Path) -> Inputs:
    """Read the spots, genes and SNPs, write `spot_umis.pdf`: capture group `00_inputs`.

    Args:
        config: the run's.
        plots_dir: the plots' directory.

    Returns:
        The `Inputs`.
    """
    # NB load_joint_data: adata (transcripts), cell_snp_Aallele / cell_snp_Ballele (h0 / h1 counts per SNP).
    (
        coords,
        barcodes,
        adata,
        exp_counts,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        across_slice_adjacency_mat,
        physical,
    ) = load_input_data(
        config,
        filter_gene_file=config.references.filtergenelist_file,
        filter_range_file=config.references.filterregion_file,
        min_snp_umis=config.quality.spot_min_snp_umis,
        min_percent_expressed_spots=config.quality.min_percent_expressed_spots,
    )

    # NB spot_umis.pdf: the retained spots' UMIs and SNP-UMIs.
    snp_umis = np.asarray(cell_snp_Aallele.sum(axis=1)).ravel() + np.asarray(cell_snp_Ballele.sum(axis=1)).ravel()
    spot_umis = np.asarray(adata.layers["count"].sum(axis=1)).ravel()
    output_file = get_output_file_meta("spot_umis.pdf")
    output_file.writer(plots_dir, output_file.name, plot_spot_umi_page(spot_umis, snp_umis))

    pause()

    # TODO' move to load_input_data
    sample_list, sample_ids = get_sample_list(adata)
    positions = SpotPositions(coords, physical, sample_ids)
    return Inputs(
        coords,
        barcodes,
        adata,
        exp_counts,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        across_slice_adjacency_mat,
        physical,
        sample_list,
        sample_ids,
        positions,
    )


def run_gene_snp_table(inputs: Inputs, config: Any) -> pd.DataFrame:
    """The gene and SNP table: capture group `01_genes`.

    Args:
        inputs: the `Inputs`.
        config: the run's.

    Returns:
        The table, one row per gene or SNP.
    """
    adata, unique_snp_ids = inputs.adata, inputs.unique_snp_ids

    df_gene_snp = form_gene_snp_table(unique_snp_ids, config.references.hgtable_file, adata)

    return df_gene_snp


def run_block_segmentation(inputs: Inputs, df_gene_snp: pd.DataFrame, config: Any) -> BlockSegmentation:
    """The haplotype blocks and their counts: capture group `02_blocks`.

    Args:
        inputs: the `Inputs`.
        df_gene_snp: `run_gene_snp_table`' table.
        config: the run's.

    Returns:
        The `BlockSegmentation`.
    """
    adata, unique_snp_ids = inputs.adata, inputs.unique_snp_ids
    cell_snp_Aallele, cell_snp_Ballele = inputs.cell_snp_Aallele, inputs.cell_snp_Ballele
    pause()

    # NB parse_visium::create_haplotype_block_ranges
    df_gene_snp = assign_initial_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        initial_min_umi=config.quality.phasing_min_snp_umis,
    )

    # NB track segmentations on a gene basis: one column per segmentation, `-1` where a level drops a gene.
    tracked_segmentations = TrackedSegmentation(df_gene_snp)
    tracked_segmentations.update_from_table(df_gene_snp, "block_id", "initial_blocks")
    snps = tracked_segmentations.get_snps(df_gene_snp)

    pause()

    # TODO' utilize <BLOCK COUNTS>
    # NB lengths = num. of blocks per contig;
    #    snp-based H0 and H0+H1 counts block;
    #    total umis per block.
    # NB counts.lengths[k] blocks on the k-th contig; an HMM never transitions across a contig boundary.
    counts = summarize_counts_for_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
    )

    pause()

    # NB bitwise CalicoST's at each of the four calls.
    # NB phase error rate on the basis of the recombination.
    log_sitewise_transmat = get_sitewise_transmat(
        "block_id",
        df_gene_snp,
        config.references.geneticmap_file,
        config.phasing.nu,
        config.phasing.logphase_shift,
    )

    pause()

    return BlockSegmentation(df_gene_snp, tracked_segmentations, snps, counts, log_sitewise_transmat)


def run_initial_phasing(
    inputs: Inputs, block_segmentation: BlockSegmentation, config: Any, plots_dir: Path
) -> InitialPhasing:
    """The initial clones and the blocks' phase, with their plots: capture group `03_phasing`.

    Args:
        inputs: the `Inputs`.
        block_segmentation: the `BlockSegmentation`; its table gains a `phase` column and its segmentation a level.
        config: the run's; `hmrf.n_colors` set.
        plots_dir: the plots' directory.

    Returns:
        The `InitialPhasing`.
    """
    adata, coords, physical, positions = inputs.adata, inputs.coords, inputs.physical, inputs.positions
    sample_list, sample_ids = inputs.sample_list, inputs.sample_ids
    df_gene_snp, tracked_segmentations, counts = (
        block_segmentation.df_gene_snp,
        block_segmentation.tracked_segmentations,
        block_segmentation.counts,
    )
    log_sitewise_transmat = block_segmentation.log_sitewise_transmat
    # baf-derived phasing (assuming initial / h&e derived clones
    logger.runtime_phase = "phasing"

    # DEPRECATE
    known_clone_assignment = known_nb_baseline = None

    if config.annotation.clone_label is not None:
        known_clone_assignment, known_nb_baseline = load_clone_labels(counts, config)
        initial_clone_for_phasing = known_clone_assignment.copy()

    else:
        labelling = get_initial_labelling(
            counts, positions, config, stage="phasing", rng=np.random.default_rng(config.hmrf.random_seed)
        )
        initial_clone_for_phasing = get_clone_indices(labelling, range(np.max(labelling) + 1))

    # TODO'
    initial_clone_index_baf = initial_clone_for_phasing

    # NB number of colors in the initial clone segmenation.
    config.hmrf.n_colors = sum(1 for index in initial_clone_for_phasing if len(index) > 0)

    logger.info(f"Initial phasing partition realized {config.hmrf.n_colors} labels.")

    # NB pseudobulk formed of all spots.
    initial_clone_pseudobulk = [[ii for ii in range(len(coords))]]

    pseudobulk_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        counts=counts,
        clone_index=initial_clone_pseudobulk,
        sample_list=sample_list,
    )

    output_file = get_output_file_meta("pseudobulk_clones_genomic.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        pseudobulk_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    # NB per-spot h&e label derived from gray-scale percentiles, for he.n_components classes; None without a slide.
    # TODO' num_classes?
    he_assignment = adata.obsm["he_label"].flatten() if "he_label" in adata.obsm else None

    # NB the initial partition cut by each spot's h&e class, under he.apply_to_initial_partition.
    initial_clone_for_phasing = initial_clone_index_baf = he_initial_partition(
        coords, initial_clone_for_phasing, he_assignment, config
    )

    # NB he_clones.png; the slide's frame, in um, is drawn again on the combined pages.
    he_frame = None
    if he_assignment is not None:
        slices = get_sample_sheet(config.paths.sample_sheet)["spaceranger_dir"]
        he_frame = get_he_frame(slices, get_spatial_positions(slices.iloc[0]), coords, physical, sample_ids)
        output_file = get_output_file_meta("he_clones.png")
        output_file.writer(plots_dir, output_file.name, plot_he_page(physical, he_assignment, he_frame=he_frame))

    # TODO' assignment -> phasing_assignment.
    # NB a pd.Series of "clone <k>" for every spot, k its index into initial_clone_for_phasing.
    assignment = pd.Series([f"clone {x}" for x in get_clone_assignment(len(coords), initial_clone_for_phasing)])

    phasing_clones_fig = plot_clones_spatial(
        physical,
        assignment,  # TODO' phasing_assignment
        sample_list=sample_list,
        sample_ids=sample_ids,
    )

    # TODO' all plots should be layout formatted.
    output_file = get_output_file_meta("prephasing_clones_spatial.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        phasing_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    prephasing_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        counts=counts,
        clone_index=initial_clone_for_phasing,
        sample_list=sample_list,
        known_nb_baseline=known_nb_baseline,
    )

    output_file = get_output_file_meta("prephasing_clones_genomic.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        prephasing_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    if config.phasing.run:
        n_states_phasing = config.hmm.n_states
        initial_parameters = get_initial_parameters(
            counts, get_clone_assignment(len(coords), initial_clone_for_phasing), config, stage="phasing"
        )

        # NB the baseline is zero until the normal spots are labelled.
        # NB refined_lengths[j] = num. of blocks in the j-th phase segment:
        #    each contig's blocks split where the minor baf steps by >= baf_change_threshold
        #    after >= min_new_segment_size blocks.  Never crosses a contig,
        #    so len(refined_lengths) >= len(lengths) and sum(refined_lengths) = sum(lengths).
        res_phasing, phase_indicator, refined_lengths = initial_phase_given_partition(
            counts,
            initial_clone_for_phasing,
            initial_parameters,
            n_states_phasing,
            log_sitewise_transmat,
            "sp",  # MAGIC params (start prob. & baf states, no transition).
            config.hmm.t_phaseing,
            tol=1.0e-3,  # MAGIC tol on HMM parameter end.
        )

        # TODO PATCH
        res_phasing["new_assignment"] = get_clone_assignment(len(coords), initial_clone_for_phasing)

        logger.info(
            f"Solution for initial phase given pop. phasing (eagle) & observed baf in {(time.time() - start_time):.2f} seconds."
        )
    else:
        # NB no phasing: every block keeps its reference (eagle) phase, and the phase segments are the contigs.
        phase_indicator = np.zeros(counts.n_segments)
        refined_lengths = counts.lengths

        logger.warning("Skipping phasing run.")

    tracked_segmentations.update(
        phasing_segments(tracked_segmentations["initial_blocks"], refined_lengths),
        "phasing_segments",
    )

    # NB per-gene phase: the phase of the gene's block, DROPPED for a gene in no block.
    gene_phase = tracked_segmentations["initial_blocks"].broadcast(
        np.asarray(phase_indicator).astype(np.int64), fill=get_dropped()
    )

    # NB phase is None for genes and otherwise 0/1 for snps given baf-inferred phase.
    df_gene_snp["phase"] = np.where(
        df_gene_snp.snp_id.isnull(),
        None,
        df_gene_snp.block_id.map({i: x for i, x in enumerate(phase_indicator)}),
    )

    postphasing_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        counts=counts,
        clone_index=initial_clone_for_phasing,
        res_combine=res_phasing,
        sample_list=sample_list,
    )

    output_file = get_output_file_meta("postphasing_clones_genomic.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        postphasing_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    return InitialPhasing(
        initial_clone_index_baf,
        known_clone_assignment,
        he_assignment,
        he_frame,
        phase_indicator,
        refined_lengths,
        gene_phase,
    )


def run_bin_segmentation(
    inputs: Inputs, block_segmentation: BlockSegmentation, initial_phasing: InitialPhasing, config: Any, plots_dir: Path
) -> tuple[pd.DataFrame, Any, Any]:
    """The phased bins, their counts and the spot adjacency: capture group `04_bins`.

    Args:
        inputs: the `Inputs`.
        block_segmentation: the `BlockSegmentation`; its segmentation gains a level.
        initial_phasing: the `InitialPhasing`.
        config: the run's.
        plots_dir: the plots' directory.

    Returns:
        The table with `bin_id`, the bins' counts and the sparse spot adjacency.
    """
    adata, coords, physical = inputs.adata, inputs.coords, inputs.physical
    cell_snp_Aallele, cell_snp_Ballele = inputs.cell_snp_Aallele, inputs.cell_snp_Ballele
    unique_snp_ids, across_slice_adjacency_mat = inputs.unique_snp_ids, inputs.across_slice_adjacency_mat
    sample_list, sample_ids = inputs.sample_list, inputs.sample_ids
    df_gene_snp, tracked_segmentations, counts = (
        block_segmentation.df_gene_snp,
        block_segmentation.tracked_segmentations,
        block_segmentation.counts,
    )
    he_assignment, phase_indicator, refined_lengths = (
        initial_phasing.he_assignment,
        initial_phasing.phase_indicator,
        initial_phasing.refined_lengths,
    )
    initial_clone_pseudobulk = [[ii for ii in range(len(coords))]]
    logger.runtime_phase = "phased genomic segmentation"

    # NB generates new genomic intervals ("bin_id") by genomic aggregation
    #    accounting for baf-derived phasing and user defined thresholds.
    df_gene_snp = create_bin_ranges(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        counts,
        refined_lengths,
        config.quality.secondary_min_umi,
        config.quality.secondary_min_snp_umi,
        config.quality.secondary_min_normal_umi,
        max_binlength=config.quality.max_binlength,
    )

    tracked_segmentations.update_from_table(df_gene_snp, "bin_id", "phased_bins")

    pause()

    logger.info("Recalculating counts for new (phased) baf inferred genome segmentation.")

    # TODO' summarize_counts_for_blocks can be adapted to summarize_counts_for_bins,
    #       given new df_gene_snp with "bin_id" and "phase" columns.
    # NB   counters per baf-phasing derived genomic intervals.
    counts = summarize_counts_for_bins(
        df_gene_snp,
        adata,
        counts,
        phase_indicator,
    )

    pause()

    pseudobulk_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        counts=counts,
        clone_index=initial_clone_pseudobulk,
        sample_list=sample_list,
        phase_switches=cumulative_phase_switches(df_gene_snp, phase_indicator),
    )

    output_file = get_output_file_meta("postphasing_pseudobulk_clones_genomic.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        pseudobulk_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    # NB per-H&E-class pseudobulk on the phased bins.
    if he_assignment is not None:
        he_classes = np.unique(he_assignment).astype(int)
        he_clones_genomic = plot_clones_genomic(
            df_cnv=None,
            counts=counts,
            clone_index=get_clone_indices(he_assignment, he_classes),
            sample_list=sample_list,
            colors=[he_colours(he_assignment)[k] for k in he_classes],
            labels=[f"H&E {k}" for k in he_classes],
        )

        output_file = get_output_file_meta("he_clones_genomic.pdf")
        output_file.writer(
            plots_dir,
            output_file.name,
            he_clones_genomic,
            transparent=True,
            bbox_inches="tight",
        )

    pause()

    # baf-derived inference of clone assignment and copy number profiles
    logger.runtime_phase = "baf-only clone & copy state inference"

    # NB hmrf.lattice_type:
    #        triangular, Visium's array, each spot's 6 nearest by the array's lattice vectors,
    #        square, Visium HD's bins, each bin's 8 nearest on its pixels (x pxl_col, y pxl_row), or its 4 edges at
    #        hmrf.coordination_num 4
    #    asymmetrics at the slice perimeter.
    #    unset, each spot's 8 nearest in its coordinates as given; construct_multislice_lattice_adjacency warns.
    adjacency_mat = construct_multislice_lattice_adjacency(
        sample_ids,
        sample_list,
        physical,
        across_slice_adjacency_mat,
        lattice_type(config),
        coordination_num(config),
    )

    pause()
    return df_gene_snp, counts, adjacency_mat


def run_baf_clone_inference(
    inputs: Inputs, counts: Any, adjacency_mat: Any, initial_phasing: InitialPhasing, config: Any, plots_dir: Path
) -> BafFit:
    """The BAF-only clones and copy states, merged under the floors, with their plots: capture group `05_baf`.

    Args:
        inputs: the `Inputs`.
        counts: `run_bin_segmentation`' counts.
        adjacency_mat: the spot adjacency.
        initial_phasing: the `InitialPhasing`.
        config: the run's.
        plots_dir: the plots' directory.

    Returns:
        The `BafFit`.
    """
    coords, physical = inputs.coords, inputs.physical
    sample_list, sample_ids = inputs.sample_list, inputs.sample_ids
    initial_clone_index_baf = initial_phasing.initial_clone_index

    # NB no normal spots labelled: the baseline is zero until they are.
    assert counts.normal_baseline_sized.is_zero()

    # NB a reference, not a copy: `with_umis` restores it after the BAF-only run.
    umis_ref = counts.umis

    logger.info("Assuming initial clone configuration for baf-inferred clones & copy states.")

    counts.with_labelling("clone", get_clone_assignment(len(coords), initial_clone_index_baf)).get_pseudobulk("clone")

    # TODO initial_baf_assignment
    assignment = pd.Series([f"clone {x}" for x in get_clone_assignment(len(coords), initial_clone_index_baf)])

    initial_clones_fig = plot_clones_spatial(
        physical,
        assignment,  # TODO initial_baf_assignment
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
    )

    # NB initial clone assignment for baf-only inference.
    output_file = get_output_file_meta("initial_baf_spatial.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        initial_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    pause()

    logger.info("Solving hmm & hmrf for copy states and clone assignment assuming baf only.")

    # NB new counts with the umis zeroed (dense or CSR); the snp umis are shared.
    counts = counts.with_umis(counts.umis.zeros(counts.umis.dtype))

    # TODO' baf_res
    labelling = get_clone_assignment(len(coords), initial_clone_index_baf)
    initial_parameters = get_initial_parameters(counts, labelling, config, stage="baf")
    res = run_hmrf(counts, labelling, adjacency_mat, config, parameters=initial_parameters, stage="baf")

    # NB not live: HmrfFit.copy is a shallow copy, which carries _locked, and get_merged_fit copies this
    #    result and writes new_assignment and total_llf to the copy. Locking here raises there.
    # res.lock()

    logger.info(f"Given single_X.shape={(counts.n_segments, 2, counts.n_spots)}, solved for res=\n{res}")
    logger.info(f"Inferred {len(np.unique(res['new_assignment']))} clones given baf data.")

    # TODO'
    assignment = pd.Series([f"clone {x}" for x in res["new_assignment"]])
    bafonly_clones_fig = plot_clones_spatial(
        physical,
        assignment,
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
    )

    # NB inferred clones from baf-only run.
    output_file = get_output_file_meta("bafonly_clones_spatial.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        bafonly_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    bafonly_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        counts=counts,
        clone_index=get_clone_indices(res["new_assignment"], np.unique(res["new_assignment"])),
        res_combine=res,
        sample_list=sample_list,
        palette_name="chisel_single",  # NB integer state lookup, no (A,B).
    )

    # NB inferred per-clone copy number profiles from baf-only run.
    output_file = get_output_file_meta("bafonly_clones_genomic.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        bafonly_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    pause()

    # NB a shallow copy.
    merged_res = res.copy()

    logger.info(f"Inferred {len(np.unique(merged_res['new_assignment']))} clones given baf data.")

    # NB merge according to min. number of spots per clone criterion; the counts' segments are dynamic.
    n_obs = counts.n_segments
    min_umicount_thresholds = n_obs * config.hmrf.min_avgumi_per_clone  # MAGIC 31_420 SNP UMIs

    merged_res = get_merged_fit(
        merged_res, counts.total_snp_umis, config.hmrf.min_spots_per_clone, min_umicount_thresholds
    )

    # NB nothing below writes to the merged result; it is read, and the rdr stage starts from its assignment.
    # merged_res.lock()

    logger.info(f"Inferred {len(np.unique(merged_res['new_assignment']))} clones given baf data after min spots merge.")

    # TODO named assignment
    assignment = pd.Series([f"clone {x}" for x in merged_res["new_assignment"]])
    merged_bafonly_clones_fig = plot_clones_spatial(
        physical,
        assignment,
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
    )

    # NB inferred clones from baf-only run, after the min-spots merge.
    output_file = get_output_file_meta("merged_bafonly_clones_spatial.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        merged_bafonly_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    merged_bafonly_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        counts=counts,
        clone_index=get_clone_indices(merged_res["new_assignment"], np.unique(merged_res["new_assignment"])),
        res_combine=merged_res,
        sample_list=sample_list,
    )

    # NB inferred copy number profiles from baf-only run, after the min-spots merge
    output_file = get_output_file_meta("merged_bafonly_clones_genomic.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        merged_bafonly_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    pause()

    # NB the counts' segments follow the current segmentation.
    n_obs = counts.n_segments

    # NB clone assignment based on BAF only, after merging similar clones.
    #    number of assigned / selected clones may be less than max. possible ("M").
    merged_baf_assignment = copy.copy(merged_res["new_assignment"])
    n_baf_clones = len(np.unique(merged_baf_assignment))

    # NB predicted copy state (MAP), (n_clones, n_obs).
    pred = merged_res["pred_cnv"]

    logger.info(f"Found {100.0 * np.mean(pred[:, :] < config.hmm.n_states)}% of baf-only copy states to have phase 0.")

    # NB used for normal spot determination only.
    # NB contains __model baf profiles__, accounted for baf-derived phase switching.
    merged_baf_profiles = np.array(
        [
            np.where(
                pred[c, :] < config.hmm.n_states,
                merged_res.parameters.p_binom[pred[c, :] % config.hmm.n_states],
                1.0 - merged_res.parameters.p_binom[pred[c, :] % config.hmm.n_states],
            )
            for c in range(n_baf_clones)
        ]
    )

    pause()

    return BafFit(counts, umis_ref, merged_res, merged_baf_profiles)


def run_normal_candidates(
    inputs: Inputs,
    baf_fit: BafFit,
    df_gene_snp: pd.DataFrame,
    tracked_segmentations: Any,
    config: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, Any, np.ndarray]:
    """The normal spots, `baf_clone_labels.tsv.gz`, and the bins and genes they filter: capture group `06_normal`.

    Args:
        inputs: the `Inputs`; `adata`'s counts zeroed at the filtered genes.
        baf_fit: the `BafFit`.
        df_gene_snp: `run_bin_segmentation`' table.
        tracked_segmentations: the segmentations; gains a level.
        config: the run's.
        output_dir: the tables' directory.

    Returns:
        The filtered table, its counts and each spot's normal candidacy.
    """
    adata, barcodes, coords, exp_counts = inputs.adata, inputs.barcodes, inputs.coords, inputs.exp_counts
    sample_list, sample_ids = inputs.sample_list, inputs.sample_ids
    counts, umis_ref, merged_res, merged_baf_profiles = baf_fit
    # clone assignment and copy number profile refinement with gene transcripts / umis
    logger.runtime_phase = "normal candidate determination"

    # NB normal candidates (per-spot boolean) with baf only.
    normal_candidate = determine_normal_candidates(
        config,
        merged_res,
        merged_baf_profiles,
        umis_ref,
    )

    # NB written once the normal spots are known, so each label table carries them.
    df_clone_label = get_assignment_frame(
        barcodes,
        coords,
        merged_res["new_assignment"],
        normal_candidate,
    )

    output_file = get_output_file_meta("baf_clone_labels.tsv.gz")
    output_file.writer(
        output_dir,
        output_file.name,
        df_clone_label,
        header=True,
        index=True,
        index_label="barcode",
        prefix="baf inferred clone labels",
    )

    pause()

    # NB restores the umis zeroed for the baf-only run, in the counts' backing (dense or CSR) and dtype.
    counts = counts.with_umis(umis_ref)

    # NB drop segments with allele-specific expression in the normal candidates.
    # TODO' normal mis-classification leads to dropped segments due to identifying CNAs as allele-specific expression?
    normal_idx = np.where(normal_candidate)[0]

    df_gene_snp, counts = normal_baf_bin_filter(
        df_gene_snp,
        counts,
        normal_idx,
    )
    tracked_segmentations.update_from_table(df_gene_snp, "bin_id", "normal_filtered_bins")

    # NB table of per-bin intervals with set(genes) and set(sites).
    df_bin_info = binned_gene_snp(df_gene_snp)

    umis_ref = counts.umis

    # NB zeros out high-umi differentially expressed genes,
    #    which may bias rdr estimates.   same as CalicoST's filter_de_genes_tri.
    if config.quality.filter_normal_diffexp:
        umis_ref, filtered_out_set = filter_normal_diffexp(
            exp_counts,
            df_bin_info,
            normal_candidate,
            sample_list=sample_list,
            sample_ids=sample_ids,
        )
    else:
        filtered_out_set = set()
        logger.warning("Assuming no filter for normal differential expression.")

    counts = counts.with_umis(umis_ref)
    adata.layers["count"] = zero_genes(adata.layers["count"], adata.var.index.isin(filtered_out_set))

    pause()

    return df_gene_snp, counts, normal_candidate


def run_rdr_bin_segmentation(
    inputs: Inputs,
    df_gene_snp: pd.DataFrame,
    counts: Any,
    normal_candidate: np.ndarray,
    tracked_segmentations: Any,
    config: Any,
) -> tuple[pd.DataFrame, Any, pd.DataFrame]:
    """The bins merged to the normal spots' floors, their counts and normal baseline: capture group `07_rebin`.

    Args:
        inputs: the `Inputs`.
        df_gene_snp: `run_normal_candidates`'s table.
        counts: `run_normal_candidates`'s counts.
        normal_candidate: each spot's normal candidacy.
        tracked_segmentations: the segmentations; gains a level.
        config: the run's.

    Returns:
        The table, the counts with the normal spots labelled, and the per-bin genes and SNPs.
    """
    adata, unique_snp_ids = inputs.adata, inputs.unique_snp_ids
    cell_snp_Aallele, cell_snp_Ballele = inputs.cell_snp_Aallele, inputs.cell_snp_Ballele
    # NB merges consecutive bins on a contig until each meets secondary_min_{umi,snp_umi,normal_umi}, the last
    #    summed over the normal candidates only, or exceeds max_binlength.  The old bin_id is kept as block_id.
    #    lengths here is bins per contig, not the phase segments, so a merge crosses a phase segment boundary
    #    but never a contig; the bins are already phased, so the baf numerators remain comparable.
    df_gene_snp = create_bin_ranges(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        counts,
        counts.lengths,
        config.quality.secondary_min_umi,
        config.quality.secondary_min_snp_umi,
        config.quality.secondary_min_normal_umi,
        max_binlength=config.quality.max_binlength,
        normal_candidates=normal_candidate,
        key="bin_id",
    )
    tracked_segmentations.update_from_table(df_gene_snp, "bin_id", "normal_rebinned_bins")

    df_bin_info = binned_gene_snp(df_gene_snp)

    phase_indicator = np.ones(counts.n_segments)

    # NB new segmentation and associated counts given normal candidate-based
    #    __filtering__ of baf-derived segments.
    counts = summarize_counts_for_bins(
        df_gene_snp,
        adata,
        counts,
        phase_indicator,
    )

    # NB zeros bins under quality.min_normal_count_perbin and labels the normal spots, which size the baseline.
    counts = determine_normal_baseline(counts, normal_candidate, config)

    pause()

    return df_gene_snp, counts, df_bin_info


def run_baf_rdr_clone_inference(
    inputs: Inputs,
    counts: Any,
    adjacency_mat: Any,
    merged_res: HmrfFit,
    normal_candidate: np.ndarray,
    known_clone_assignment: Any,
    config: Any,
    output_dir: Path,
    plots_dir: Path,
) -> HmrfFit:
    """The read-depth and BAF clones and copy states from the BAF clones, merged and renumbered normal first, with
    `clone_labels.tsv.gz`, the npz and their plots: capture group `08_rdr`.

    Args:
        inputs: the `Inputs`.
        counts: `run_rdr_bin_segmentation`'s counts.
        adjacency_mat: the spot adjacency.
        merged_res: the BAF-only fit.
        normal_candidate: each spot's normal candidacy.
        known_clone_assignment: the annotated clones, or None.
        config: the run's.
        output_dir: the tables' directory.
        plots_dir: the plots' directory.

    Returns:
        The final fit, locked.
    """
    barcodes, coords, physical, positions = inputs.barcodes, inputs.coords, inputs.physical, inputs.positions
    sample_list, sample_ids = inputs.sample_list, inputs.sample_ids
    merged_baf_assignment = copy.copy(merged_res["new_assignment"])
    n_baf_clones = len(np.unique(merged_baf_assignment))
    logger.runtime_phase = "baf/rdr clone & copy state inference"

    logger.info(
        f"Refinining {n_baf_clones} baf-identified clones with umi data assuming n_clones_rdr={config.hmrf.n_clones_rdr}"
    )

    # TODO' >>>>>>>>
    if config.annotation.clone_label is not None:
        global_initial_clone_index = known_clone_assignment
    else:
        labelling = get_initial_labelling(
            counts.with_labelling("clone", merged_baf_assignment),
            positions,
            config,
            stage="rdr",
            rng=np.random.default_rng(config.hmrf.random_seed),
        )

        # NB per-clone list of spot index arrays for the initial rdr clones, each baf clone split spatially;
        #    the same form as initial_clone_index_baf.
        # TODO defined function for this?
        global_initial_clone_index = get_clone_indices(labelling, range(np.max(labelling) + 1))

    labelling = get_clone_assignment(len(coords), global_initial_clone_index)
    initial_parameters = get_initial_parameters(counts, labelling, config, stage="rdr")
    res_combine = run_hmrf(counts, labelling, adjacency_mat, config, parameters=initial_parameters, stage="rdr")

    logger.info(f"Solved for res_combine=\n{res_combine}")

    # TODO' named_assignment.
    assignment = pd.Series([f"clone {x}" for x in res_combine["new_assignment"]])
    rdr_baf_clones_fig = plot_clones_spatial(
        physical,
        assignment,
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
    )

    # NB inferred clones from baf-only run.
    output_file = get_output_file_meta("rdr_baf_clones_spatial.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        rdr_baf_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    rdr_baf_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        counts=counts,
        clone_index=get_clone_indices(res_combine["new_assignment"], np.unique(res_combine["new_assignment"])),
        res_combine=res_combine,
        sample_list=sample_list,
        palette_name="chisel_single",  # NB integer state lookup, no (A,B).
    )

    # NB inferred per-clone copy number profiles from baf-only run.
    output_file = get_output_file_meta("rdr_baf_clones_genomic.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        rdr_baf_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    pause()

    # NB a shallow copy.
    merged_res_combine = res_combine.copy()

    logger.info(f"Inferred {len(np.unique(merged_res_combine['new_assignment']))} clones given rdr-baf data.")

    # NB merge according to min. number of spots per clone criterion; the counts' segments are dynamic.
    n_obs = counts.n_segments
    min_umicount_thresholds = n_obs * config.hmrf.min_avgumi_per_clone  # MAGIC 31_420 SNP UMIs

    merged_res_combine = get_merged_fit(
        merged_res_combine, counts.total_snp_umis, config.hmrf.min_spots_per_clone, min_umicount_thresholds
    )

    logger.info(
        f"Inferred {len(np.unique(merged_res_combine['new_assignment']))} clones given rdr-baf data after min spots merge."
    )
    res_combine = merged_res_combine

    # TODO'
    assignment = pd.Series([f"clone {x}" for x in merged_res_combine["new_assignment"]])
    merged_rdr_baf_clones_fig = plot_clones_spatial(
        physical,
        assignment,
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
    )

    # NB inferred clones from baf-only run, after the min-spots merge
    output_file = get_output_file_meta("merged_rdr_baf_clones_spatial.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        merged_rdr_baf_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    merged_rdr_baf_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        counts=counts,
        clone_index=get_clone_indices(
            merged_res_combine["new_assignment"],
            np.unique(merged_res_combine["new_assignment"]),
        ),
        res_combine=merged_res_combine,
        sample_list=sample_list,
    )

    # NB inferred copy number profiles from baf-only run, after the min-spots merge
    output_file = get_output_file_meta("merged_rdr_baf_clones_genomic.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        merged_rdr_baf_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    # TODO' prev_assignment renaming.
    n_final_clones = len(np.unique(res_combine["prev_assignment"]))

    logger.info(f"Inferred {n_final_clones} clones given rdr & baf data.")
    fit = res_combine.parameters
    logger.info(f"Found rdr-split clone rdrs:\n{np.exp(fit.log_mu)}.")
    logger.info(f"Found rdr-split clone bafs:\n{fit.p_binom}.")

    logger.info(f"Assuming max. alpha dispersion={np.max(fit.alphas):.4f} between clones given current:\n{fit.alphas}")
    logger.info(f"Assuming min. tau dispersion={np.min(fit.taus):.4f} between clones given current:\n{fit.taus}")

    # NB re-order clones such that the index of the normal-defined clone is 0.
    res_combine = get_reindexed_fit(res_combine)
    res_combine.lock()

    final_clone_ids, final_clone_counts = np.unique(res_combine["new_assignment"], return_counts=True)

    # NB get_reindexed_fit labels the normal clone 0 and the rest 1 .. N-1.
    assert np.array_equal(final_clone_ids, np.arange(len(final_clone_ids))), (
        f"final clone ids are not 0 .. N-1: {final_clone_ids}"
    )

    logger.info(
        f"Inferred final clones=\n{final_clone_ids}\nwith fractions=\n{final_clone_counts / np.sum(final_clone_counts)}."
    )

    # NB final clone labels; independent of the integer copy number solution and its ploidy.
    df_clone_label = get_assignment_frame(
        barcodes,
        coords,
        res_combine["new_assignment"],
        normal_candidate,
    )

    output_file = get_output_file_meta("clone_labels.tsv.gz")
    output_file.writer(
        output_dir,
        output_file.name,
        df_clone_label,
        header=True,
        index=True,
        index_label="barcode",
        prefix="inferred clone labels",
    )

    pause()

    npz_fields = res_combine.get_npz_fields()
    for key in ["new_log_mu", "new_alphas", "new_p_binom", "new_taus", "total_llf", "pred_cnv"]:
        logger.info(f"Solved for {key}:\n{npz_fields[key]}")

    output_file = get_output_file_meta("rdrbaf_final_nstates{n_states}_smp.npz")
    output_file.writer(output_dir, output_file.name.format(n_states=config.hmm.n_states), **npz_fields)

    pause()

    return res_combine


def run_integer_copy_decoding(
    inputs: Inputs,
    counts: Any,
    res_combine: HmrfFit,
    df_gene_snp: pd.DataFrame,
    df_bin_info: pd.DataFrame,
    normal_candidate: np.ndarray,
    block_segmentation: BlockSegmentation,
    initial_phasing: InitialPhasing,
    config: Any,
    output_dir: Path,
    plots_dir: Path,
) -> None:
    """The integer copies, their tables, the segmentations and the final plots: capture group `09_outputs`.

    Args:
        inputs: the `Inputs`.
        counts: `run_rdr_bin_segmentation`'s counts.
        res_combine: the final fit.
        df_gene_snp: `run_rdr_bin_segmentation`'s table.
        df_bin_info: the per-bin genes and SNPs.
        normal_candidate: each spot's normal candidacy.
        block_segmentation: the `BlockSegmentation`, its segmentations and SNPs.
        initial_phasing: the `InitialPhasing`, its gene phase and slide frame.
        config: the run's.
        output_dir: the tables' directory.
        plots_dir: the plots' directory.
    """
    barcodes, coords, physical = inputs.barcodes, inputs.coords, inputs.physical
    sample_list, sample_ids = inputs.sample_list, inputs.sample_ids
    tracked_segmentations, snps = block_segmentation.tracked_segmentations, block_segmentation.snps
    gene_phase, he_frame = initial_phasing.gene_phase, initial_phasing.he_frame
    final_clone_ids = np.unique(res_combine["new_assignment"])
    # integer copy number determination gived inferrered per-state (rdr,baf) and assumed ploidy.
    logger.runtime_phase = "integer copy number determination"

    # TODO' updated res_combine keys for integer copies, and clone assignment according to unique inferred states.

    # NB the final clones' pseudobulk, which the decoder scores.
    pseudobulk = counts.with_labelling("clone", res_combine["new_assignment"]).get_pseudobulk("clone")
    base_nb_mean = pseudobulk.normal_baseline_sized.T

    # NB ab_decode.assumed_ploidy: "diploid" or several (comma-separated), each tumour clone decoded at each and its best kept.
    #    output named accordingly: cnv_<ploidy>_*.tsv at one, cnv_best_ploidy_*.tsv at several.
    ploidies = assumed_ploidy(config)

    # NB the integer copy tables' prefix: cnv_<ploidy> at one ploidy, cnv_best_ploidy at several.
    cnv_table_prefix = get_output_path_prefix(ploidies)

    # NB ab_decode.decoder: the (A, B) HMM per bin and clone, or CalicoST's diploid hill climb per state and clone.
    decoder = config.ab_decode.decoder
    clone_normal_fractions, clone_ploidy = None, {}

    if decoder == "hmm_ab":
        # NB the normal clone, id 0, held at (1, 1) and theta 1; every tumour clone from each of normal_fraction_starts.
        given_normal_fraction = np.where(np.asarray(final_clone_ids) == 0, 1.0, 0.0)
        hmm_ab_ploidy_decoded = decode_integer_copies(
            pseudobulk,
            CnaHmmParameters(None, None, None, res_combine.parameters.alphas, res_combine.parameters.taus),
            # NB apply_logmu_shift False decodes every clone at Z_c = 1, under a genome-wide ploidy its own.
            CnaEmissionConfounders(
                given_normal_fraction.reshape(-1, 1),
                res_combine.confounders.log_mu_shift
                if config.ab_decode.apply_logmu_shift
                else np.zeros_like(res_combine.confounders.log_mu_shift),
            ),
            counts.lengths,
            config,
        )
        bin_copies = hmm_ab_ploidy_decoded.copies
        logger.info(
            f"{decoder} normal fractions per clone: {hmm_ab_ploidy_decoded.normal_fractions.tolist()}, status "
            f"{ {c: v.value for c, v in hmm_ab_ploidy_decoded.status.items()} }."
        )
        # NB by the label the clone map carries, so the key reads `Clone I (9%)` whichever clones survived.
        clone_normal_fractions = {
            f"clone {cid}": float(f)
            for cid, f in zip(final_clone_ids, hmm_ab_ploidy_decoded.normal_fractions, strict=True)
        }
        clone_ploidy = hmm_ab_ploidy_decoded.ploidy
    else:
        state_copies = calicost_diploid_decoder(
            res_combine.parameters.log_mu,
            base_nb_mean,
            res_combine.parameters.p_binom,
            res_combine["pred_cnv"].T,
            nondiploid_rdrdist=config.ab_decode.nondiploid_rdrdist,
        )
        bin_copies = np.stack(
            [state_copies[res_combine["pred_cnv"][s], s] for s in range(len(final_clone_ids))], axis=1
        )

    # NB A/B integer copy number per gene, genome segment and state.
    df_genelevel_cnv, df_seglevel_cnv, state_cnv = get_integer_copy_tables(
        df_bin_info,
        df_gene_snp,
        res_combine,
        bin_copies,
        final_clone_ids,
        base_nb_mean,
        n_states=config.hmm.n_states,
        ploidies=ploidies,
        clone_ploidy=clone_ploidy,
    )

    pause()

    # NB write the gene-level integer copies.
    output_file = get_output_file_meta("{prefix}_genelevel.tsv")
    output_file.writer(
        output_dir,
        output_file.name.format(prefix=cnv_table_prefix),
        df_genelevel_cnv,
        header=True,
        index=True,
    )

    # NB the bins where any clone departs from (1, 1), logged at debug.
    mask = df_seglevel_cnv.filter(regex=r" [AB]$").ne(1).any(axis=1)

    # NB display the segment-level copy number for the selected segments.
    with pd.option_context(
        "display.expand_frame_repr",
        False,
        "display.max_columns",
        None,
        "display.width",
        100_000,
        "display.max_colwidth",
        None,
    ):
        logger.debug(
            "Solved for integer copy numbers @ segments:\n%s",
            df_seglevel_cnv[mask].to_string(index=False),
        )

    # NB write integer copies for the current genome segmentation.
    output_file = get_output_file_meta("{prefix}_seglevel.tsv")
    output_file.writer(
        output_dir,
        output_file.name.format(prefix=cnv_table_prefix),
        df_seglevel_cnv,
        header=True,
        index=False,
    )

    with pd.option_context(
        "display.expand_frame_repr",
        False,
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_colwidth",
        None,
    ):
        logger.info(
            "Solved for the model parameters @ states:\n%s",
            state_cnv.to_string(index=False),
        )

    # NB write the model parameters for the inferred states.
    output_file = get_output_file_meta("{prefix}_perstate.tsv")
    output_file.writer(
        output_dir,
        output_file.name.format(prefix=cnv_table_prefix),
        state_cnv,
        header=True,
        index=False,
    )

    logger.runtime_phase = "finalize"

    # NB the clones merged where their integer (A, B) profiles agree (ab_decode.int_clone_agreement of bins)
    int_clone_agreement = config.ab_decode.int_clone_agreement
    int_groups = get_merged_ab_decoded_clones(df_seglevel_cnv, int_clone_agreement)
    int_labels = int_groups[np.asarray(res_combine["new_assignment"])]

    field = get_labelling_unary(
        counts, res_combine.parameters, res_combine.confounders, np.asarray(res_combine["pred_cnv"]), phased=False
    )

    field_labels = int_groups[field_argmax(field)]

    output_file = get_output_file_meta("clone_labels_int.tsv.gz")
    output_file.writer(
        output_dir,
        output_file.name,
        get_assignment_frame(
            barcodes,
            coords,
            int_labels,
            normal_candidate,
            field_labels,
        ),
        header=True,
        index=True,
        index_label="barcode",
    )
    field_assignment = pd.Series([f"clone {x}" for x in field_labels]) if config.run.plot_field_argmax else None

    output_file = get_output_file_meta("segmentation.tsv.gz")
    output_file.writer(
        output_dir,
        output_file.name,
        tracked_segmentations.frame().assign(phase=gene_phase),
        header=True,
        index=False,
    )

    output_file = get_output_file_meta("snp_segmentation.tsv.gz")
    output_file.writer(
        output_dir,
        output_file.name,
        tracked_segmentations.get_snp_frame(snps, gene_phase),
        header=True,
        index=False,
    )

    # NB the fit's real-valued states per clone, no integer copies.
    real_rdr_baf_fig = plot_clones_genomic(
        counts,
        res_combine=res_combine,
        sample_list=sample_list,
        chrtext_shift=-0.3,
    )

    output_file = get_output_file_meta("real_clones_genomic.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        real_rdr_baf_fig,
        transparent=True,
        bbox_inches="tight",
    )

    # NB the integer copies the run wrote, f"{cnv_table_prefix}_seglevel.tsv".
    rdr_baf_fig = plot_clones_genomic(
        counts,
        df_cnv=df_seglevel_cnv,  # segment level: chr, start, end, HMM states (k), A/B copies, & model (log_mu, p_binom) for each clone.
        res_combine=res_combine,
        sample_list=sample_list,
        chrtext_shift=-0.3,
    )

    output_file = get_output_file_meta("clones_genomic.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        rdr_baf_fig,
        transparent=True,
        bbox_inches="tight",
    )

    # NB clones fig., the fitted clones, independent of the integer copies.
    assignment = pd.Series([f"clone {x}" for x in res_combine["new_assignment"]])
    clones_fig = plot_clones_spatial(
        physical,
        assignment,
        sample_list=sample_list,
        sample_ids=sample_ids,
    )

    output_file = get_output_file_meta("clones_spatial.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    fig_copy_number_profile = plot_copy_number_profile(
        df_seglevel_cnv,  # segment level: chr, start, end, HMM states (k), A/B copies, & model (log_mu, p_binom) for each clone.
    )

    output_file = get_output_file_meta("copy_number_profile.pdf")
    output_file.writer(
        plots_dir,
        output_file.name,
        fig_copy_number_profile,
        transparent=True,
        bbox_inches="tight",
    )

    # NB one page: the slide and the clones in space, over the copy-number profile, over the clones' RDR and BAF.
    combined_fig = plot_combined_page(
        counts,
        df_seglevel_cnv,
        res_combine,
        physical,
        assignment,
        sample_list=sample_list,
        sample_ids=sample_ids,
        he_frame=he_frame,
        int_groups=int_groups,
        normal_fractions=clone_normal_fractions,
        field_assignment=field_assignment,
    )

    output_file = get_output_file_meta("combined.pdf")
    output_file.writer(plots_dir, output_file.name, combined_fig)

    # NB the same page before the integer decode: the clones the fit inferred, no copy-number profile, and the
    #    tracks carrying the data alone.
    real_fig = plot_combined_page(
        counts,
        df_seglevel_cnv,
        res_combine,
        physical,
        assignment,
        sample_list=sample_list,
        sample_ids=sample_ids,
        he_frame=he_frame,
        normal_fractions=clone_normal_fractions,
        states=fitted_states_real,
    )

    output_file = get_output_file_meta("real_combined.pdf")
    output_file.writer(plots_dir, output_file.name, real_fig)
