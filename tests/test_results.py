"""`results.get_integer_copy_tables` as `run_cnamaste` writes it: `cnv_<ploidy>_seglevel.tsv`, `_genelevel.tsv` and
`_perstate.tsv`, written by `results.write_tsv` and read back as `qa.metrics.read_estimate` reads them.

Input, seeded: 3 clones on 12 bins of 1 kb over two contigs, 4 states (state 3 unused), 20 genes each placed inside one
bin and two in a bin the run removed (`bin_id` null), and SNP rows. The referees are the inputs themselves: the decoded
`(A, B)` and states per clone and bin, the fit's per-state parameters, and each gene's position against the bins'
`[START, END)` (not its `bin_id`).

Floats are written by `DataFrame.to_csv` at full precision and read back exactly with `float_precision="round_trip"`;
the readers' default parser (`read_estimate`, `read_csv`) is within 1 ulp. The file runs in under 1 s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest

from cnamaste.emission import get_unconfounded
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmrf_fit import CloneAssignment, HMMProfile, HmrfFit
from cnamaste.results import get_integer_copy_tables, get_output_path, get_output_path_prefix, write_tsv

N_CLONES, N_OBS, N_STATES, N_GENES, BIN = 3, 12, 4, 20, 1_000


class Decoded(NamedTuple):
    df_bin_info: pd.DataFrame
    df_gene_snp: pd.DataFrame
    res: dict[str, Any]
    bin_copies: np.ndarray
    base_nb_mean: np.ndarray


def get_decoded(seed: int = 0, duplicate_gene: bool = False) -> Decoded:
    """Bins, genes and a decode as `run_cnamaste` holds them before the tables; `duplicate_gene` names one gene at
    two loci in two bins, as the reference carries LINC01505."""
    rng = np.random.default_rng(seed)
    chrom = np.repeat([1, 2], N_OBS // 2)
    start = np.tile(np.arange(N_OBS // 2), 2) * BIN
    df_bin_info = pd.DataFrame({"CHR": chrom, "START": start, "END": start + BIN})

    gene_bin = rng.integers(0, N_OBS, N_GENES)
    genes = pd.DataFrame(
        {
            "CHR": chrom[gene_bin],
            "START": start[gene_bin] + rng.integers(0, BIN // 2, N_GENES),
            "gene": [f"g{i}" for i in range(N_GENES)],
            "is_interval": True,
            "bin_id": gene_bin.astype(float),
        }
    )
    # NB two genes in a bin normal_baf_bin_filter removed: no call.
    genes.loc[:1, "bin_id"] = np.nan
    if duplicate_gene:
        genes.loc[N_GENES - 2 :, "gene"] = "LINC01505"
    snps = pd.DataFrame({"CHR": 1, "START": rng.integers(0, BIN, 5), "gene": None, "is_interval": False, "bin_id": 0.0})
    df_gene_snp = pd.concat([genes, snps], ignore_index=True)

    pred_cnv = rng.integers(0, N_STATES - 1, (N_CLONES, N_OBS))
    parameters = CnaHmmParameters(None, rng.normal(0.0, 0.3, N_STATES), rng.uniform(0.1, 0.5, N_STATES), None, None)
    res = HmrfFit(
        parameters,
        get_unconfounded(N_CLONES),
        np.zeros((N_STATES, N_STATES)),
        HMMProfile(np.zeros((N_CLONES, N_OBS, N_STATES)), pred_cnv),
        np.nan,
        N_STATES,
        CloneAssignment(),
    )
    bin_copies = rng.integers(0, 4, (N_OBS, N_CLONES, 2))
    return Decoded(df_bin_info, df_gene_snp, res, bin_copies, np.ones((N_OBS, N_CLONES)))


def write_tables(directory: Path, decoded: Decoded, ploidies: list[str], **kwargs: Any) -> None:
    """The three tables as `run_cnamaste` writes them: the same calls, headers and indices."""
    tables = get_integer_copy_tables(
        decoded.df_bin_info,
        decoded.df_gene_snp,
        decoded.res,
        decoded.bin_copies,
        np.arange(N_CLONES),
        decoded.base_nb_mean,
        n_states=N_STATES,
        ploidies=ploidies,
        clone_ploidy=kwargs.pop("clone_ploidy", {}),
        **kwargs,
    )
    prefix = get_output_path_prefix(ploidies)
    write_tsv(directory, f"{prefix}_genelevel.tsv", tables.genelevel, header=True, index=True)
    write_tsv(directory, f"{prefix}_seglevel.tsv", tables.seglevel, header=True, index=False)
    write_tsv(directory, f"{prefix}_perstate.tsv", tables.perstate, header=True, index=False)


def read_table(directory: Path, level: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(get_output_path(directory, level), sep="\t", float_precision="round_trip", **kwargs)


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("tables")
    write_tables(directory, get_decoded(), ["diploid"])
    return directory


def test_seglevel_is_each_clones_decoded_states_and_copies_per_bin(written: Path) -> None:
    """Fails where a clone's `k`, `logmu`, `p`, `A` or `B` column is not its decode at every bin, the columns leave
    `CHR START END` then per clone `k logmu p A B`, or a value changes in the round trip."""
    decoded, seglevel = get_decoded(), read_table(written, "seglevel")
    names = [f"clone{c} {n}" for c in range(N_CLONES) for n in ("k", "logmu", "p", "A", "B")]
    assert list(seglevel.columns) == ["CHR", "START", "END", *names]
    pd.testing.assert_frame_equal(seglevel[["CHR", "START", "END"]], decoded.df_bin_info)
    pred = decoded.res["pred_cnv"]
    for c in range(N_CLONES):
        assert np.array_equal(seglevel[f"clone{c} k"], pred[c])
        assert np.array_equal(seglevel[f"clone{c} logmu"], decoded.res.parameters.log_mu[pred[c]])
        assert np.array_equal(seglevel[f"clone{c} p"], decoded.res.parameters.p_binom[pred[c]])
        assert np.array_equal(seglevel[[f"clone{c} A", f"clone{c} B"]], decoded.bin_copies[:, c])


def test_the_readers_default_parser_reads_seglevel_floats_within_one_ulp(written: Path) -> None:
    """`read_estimate`'s `read_csv` default parser, not `round_trip`: fails where a float departs by more than 1 ulp,
    i.e. the writer drops precision."""
    decoded = get_decoded()
    seglevel = pd.read_csv(get_output_path(written, "seglevel"), sep="\t")
    for c in range(N_CLONES):
        expected = decoded.res.parameters.log_mu[decoded.res["pred_cnv"][c]]
        assert np.all(np.abs(seglevel[f"clone{c} logmu"] - expected) <= np.spacing(np.abs(expected)))


def test_genelevel_is_seglevel_through_the_bin_that_contains_each_gene(written: Path) -> None:
    """Every called gene once, its copies those of the seglevel row whose `[START, END)` holds it on its contig; an
    uncalled gene and a SNP absent. Fails on a gene mapped through the wrong bin, dropped, repeated or invented."""
    decoded = get_decoded()
    genelevel, seglevel = read_table(written, "genelevel", index_col=0), read_table(written, "seglevel")
    genes = decoded.df_gene_snp[decoded.df_gene_snp.is_interval & decoded.df_gene_snp.bin_id.notna()]
    assert list(genelevel.index) == list(genes.gene)
    assert list(genelevel.columns) == [f"clone{c} {a}" for c in range(N_CLONES) for a in "AB"]
    for gene in genes.itertuples():
        row = seglevel[(seglevel.CHR == gene.CHR) & (seglevel.START <= gene.START) & (gene.START < seglevel.END)]
        assert len(row) == 1
        for column in genelevel.columns:
            assert genelevel.at[gene.gene, column] == row[column].item()


@pytest.mark.xfail(strict=True, reason="T- #30: a gene at k loci is written k ** n_clones times")
def test_a_gene_the_reference_carries_twice_is_written_once_per_locus(tmp_path: Path) -> None:
    """LINC01505 at two loci, as `hgTables_hg38_gencode.txt` (`7e9af08d`) carries it on chr9: its rows are the loci's,
    2. Fails today at 2 ** 3 = 8, the per-clone `join` on a duplicated index; 16 rows at 4 clones on CalicoST easy
    (`2d4ce9a9`)."""
    write_tables(tmp_path, get_decoded(duplicate_gene=True), ["diploid"])
    genelevel = read_table(tmp_path, "genelevel", index_col=0)
    assert int((genelevel.index == "LINC01505").sum()) == 2


def test_perstate_is_the_fits_parameters(written: Path) -> None:
    """Per clone, `logmu` and `p` the fit's per state, and no other column. Fails where a column is another clone's or
    another state's, or a copy number returns to the table."""
    decoded, perstate = get_decoded(), read_table(written, "perstate")
    assert list(perstate.columns) == [f"clone{c} {n}" for c in range(N_CLONES) for n in ("logmu", "p")]
    for c in range(N_CLONES):
        assert np.array_equal(perstate[f"clone{c} logmu"], decoded.res.parameters.log_mu)
        assert np.array_equal(perstate[f"clone{c} p"], decoded.res.parameters.p_binom)


def test_perstate_names_each_clones_ploidy(tmp_path: Path) -> None:
    """At two assumed ploidies: a `ploidy` column per clone carries `clone_ploidy`, `diploid` where it has none, and
    the prefix is `cnv_best_ploidy`. Fails where a ploidy lands on another clone."""
    write_tables(tmp_path, get_decoded(), ["diploid", "triploid"], clone_ploidy={1: "triploid"})
    assert get_output_path(tmp_path, "perstate").name == "cnv_best_ploidy_perstate.tsv"
    perstate = read_table(tmp_path, "perstate")
    for c, ploidy in enumerate(("diploid", "triploid", "diploid")):
        assert set(perstate[f"clone{c} ploidy"]) == {ploidy}


def test_a_clone_without_transcripts_is_left_out_of_every_table(tmp_path: Path) -> None:
    """Clone 1 with zero pseudobulk baseline: no `clone1` column in any table, clones 0 and 2 as decoded. Fails where
    the empty clone is written, or its removal shifts another clone's columns."""
    decoded = get_decoded()
    decoded.base_nb_mean[:, 1] = 0.0
    write_tables(tmp_path, decoded, ["diploid"])
    for level in ("seglevel", "genelevel", "perstate"):
        table = read_table(tmp_path, level)
        assert not any(str(column).startswith("clone1 ") for column in table.columns)
    seglevel = read_table(tmp_path, "seglevel")
    assert np.array_equal(seglevel[["clone2 A", "clone2 B"]], decoded.bin_copies[:, 2])
