"""`results.get_integer_copy_tables` against CalicoST `c1abcae`'s integer-copy tables, `cnv_seglevel.tsv`,
`_genelevel.tsv` and `_perstate.tsv`, as `main()` assembles them.

The referee is a verbatim copy of `main()`'s lines that assemble the tables from each clone's integer copies per
state (`call_calicost_tables`): they are inline in `main()`, after the integer-copy fit. Its outputs are recorded
(`calicost_cache`). Inputs: 2 clones on 8 bins over two contigs through 3 states, one per-state fit shared by the
clones (cnamaste fits one), each clone's integer copies per state, and 6 genes, one in a bin the run removed (`bin_id`
null), seeded. The file runs in under 1 s.
"""

from __future__ import annotations

import copy
import functools
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest

from calicost_cache import Referee, departs
from cnamaste.emission import get_unconfounded
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmrf_fit import CloneAssignment, HMMProfile, HmrfFit
from cnamaste.results import IntegerCopyTables, get_integer_copy_tables

N_CLONES, N_OBS, N_STATES = 2, 8, 3


class Decoded(NamedTuple):
    df_bininfo: pd.DataFrame
    df_gene_snp: pd.DataFrame
    pred_cnv: np.ndarray
    """`(n_obs, n_clones)`, CalicoST's layout."""
    log_mu: np.ndarray
    p_binom: np.ndarray
    state_copies: np.ndarray
    """`(n_clones, n_states, 2)` each clone's `best_integer_copies`."""


@pytest.fixture(scope="module")
def decoded() -> Decoded:
    rng = np.random.default_rng(0)
    start = np.tile(np.arange(N_OBS // 2), 2) * 1_000
    df_bininfo = pd.DataFrame({"CHR": np.repeat([1, 2], N_OBS // 2), "START": start, "END": start + 1_000})
    bin_id = np.array([0.0, 2.0, 3.0, np.nan, 5.0, 7.0])
    genes = pd.DataFrame({"gene": [f"g{i}" for i in range(6)], "is_interval": True, "bin_id": bin_id})
    snps = pd.DataFrame({"gene": None, "is_interval": False, "bin_id": [1.0, 4.0]})
    return Decoded(
        df_bininfo,
        pd.concat([genes, snps], ignore_index=True),
        rng.integers(0, N_STATES, (N_OBS, N_CLONES)),
        rng.normal(0.0, 0.3, N_STATES),
        rng.uniform(0.1, 0.5, N_STATES),
        rng.integers(0, 4, (N_CLONES, N_STATES, 2)),
    )


def call_calicost_tables(d: Decoded) -> tuple[Any, ...]:
    """CalicoST's tables, each as its columns and values; the gene table also its index."""
    n_obs, df_bininfo, df_gene_snp = N_OBS, d.df_bininfo, d.df_gene_snp
    config = {"n_states": N_STATES}
    res_combine = {
        "pred_cnv": d.pred_cnv,
        "new_log_mu": np.repeat(d.log_mu[:, None], N_CLONES, 1),
        "new_p_binom": np.repeat(d.p_binom[:, None], N_CLONES, 1),
    }
    allele_specific_copy, state_cnv, df_genelevel_cnv = [], [], None
    # NB copied from calicost_main.py:323 and 344-361, each clone's `best_integer_copies` given (c1abcae).
    for s, cid in enumerate(np.arange(N_CLONES)):
        best_integer_copies = d.state_copies[s]
        allele_specific_copy.append(
            pd.DataFrame(
                best_integer_copies[res_combine["pred_cnv"][:, s], 0].reshape(1, -1),
                index=[f"clone{cid} A"],
                columns=np.arange(n_obs),
            )
        )
        allele_specific_copy.append(
            pd.DataFrame(
                best_integer_copies[res_combine["pred_cnv"][:, s], 1].reshape(1, -1),
                index=[f"clone{cid} B"],
                columns=np.arange(n_obs),
            )
        )
        state_cnv.append(
            pd.DataFrame(
                res_combine["new_log_mu"][:, s].reshape(-1, 1),
                columns=[f"clone{cid} logmu"],
                index=np.arange(config["n_states"]),
            )
        )
        state_cnv.append(
            pd.DataFrame(
                res_combine["new_p_binom"][:, s].reshape(-1, 1),
                columns=[f"clone{cid} p"],
                index=np.arange(config["n_states"]),
            )
        )
        state_cnv.append(
            pd.DataFrame(
                best_integer_copies[:, 0].reshape(-1, 1), columns=[f"clone{cid} A"], index=np.arange(config["n_states"])
            )
        )
        state_cnv.append(
            pd.DataFrame(
                best_integer_copies[:, 1].reshape(-1, 1), columns=[f"clone{cid} B"], index=np.arange(config["n_states"])
            )
        )
        bin_Acopy_mappers = {i: x for i, x in enumerate(best_integer_copies[res_combine["pred_cnv"][:, s], 0])}
        bin_Bcopy_mappers = {i: x for i, x in enumerate(best_integer_copies[res_combine["pred_cnv"][:, s], 1])}
        tmpdf = pd.DataFrame(
            {
                "gene": df_gene_snp[df_gene_snp.is_interval].gene,
                f"clone{s} A": df_gene_snp[df_gene_snp.is_interval]["bin_id"].map(bin_Acopy_mappers),
                f"clone{s} B": df_gene_snp[df_gene_snp.is_interval]["bin_id"].map(bin_Bcopy_mappers),
            }
        ).set_index("gene")
        if df_genelevel_cnv is None:
            df_genelevel_cnv = copy.copy(tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int))
        else:
            df_genelevel_cnv = df_genelevel_cnv.join(tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int))
    # NB copied from calicost_main.py:367-369 and 372 (c1abcae).
    allele_specific_copy = pd.concat(allele_specific_copy)
    df_seglevel_cnv = pd.DataFrame(
        {"CHR": df_bininfo.CHR.values, "START": df_bininfo.START.values, "END": df_bininfo.END.values}
    )
    df_seglevel_cnv = df_seglevel_cnv.join(allele_specific_copy.T)
    state_cnv = functools.reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="inner"), state_cnv
    )
    return (
        list(df_genelevel_cnv.index),
        list(df_genelevel_cnv.columns),
        df_genelevel_cnv.to_numpy(),
        list(df_seglevel_cnv.columns),
        df_seglevel_cnv.to_numpy(),
        list(state_cnv.columns),
        state_cnv.to_numpy(),
    )


def get_tables(referee: Referee, d: Decoded) -> tuple[IntegerCopyTables, tuple[Any, ...]]:
    """cnamaste's `IntegerCopyTables` at `assumed_ploidy: diploid`, and `call_calicost_tables`, from the cache."""
    res = HmrfFit(
        CnaHmmParameters(None, d.log_mu, d.p_binom, None, None),
        get_unconfounded(N_CLONES),
        np.zeros((N_STATES, N_STATES)),
        HMMProfile(np.zeros((N_CLONES, N_OBS, N_STATES)), np.ascontiguousarray(d.pred_cnv.T)),
        np.nan,
        N_STATES,
        CloneAssignment(),
    )
    bin_copies = np.stack([d.state_copies[c][d.pred_cnv[:, c]] for c in range(N_CLONES)], axis=1)
    ours = get_integer_copy_tables(
        d.df_bininfo,
        d.df_gene_snp,
        res,
        bin_copies,
        np.arange(N_CLONES),
        np.ones((N_OBS, N_CLONES)),
        n_states=N_STATES,
        ploidies=["diploid"],
        clone_ploidy={},
    )
    inputs = (d.df_bininfo, d.df_gene_snp, d.pred_cnv, d.log_mu, d.p_binom, d.state_copies)
    theirs = referee("results.tables", inputs, functools.partial(call_calicost_tables, d), kept=range(7))
    return ours, theirs


def test_the_integer_copy_tables_hold_calicost_s_values(decoded: Decoded, referee: Referee) -> None:
    """Each table's values under CalicoST's column names are CalicoST's, exactly: the gene table whole (5 called
    genes), the bin table's positions and A and B copies per clone, the state table's `logmu` and `p` per clone.
    Fails where a copy is read through the wrong state or bin, a gene through the wrong bin, or a value changes."""
    ours, (genes, gene_columns, gene_values, seg_columns, seg_values, state_columns, state_values) = get_tables(
        referee, decoded
    )
    assert list(ours.genelevel.index) == genes and len(genes) == 5
    assert list(ours.genelevel.columns) == gene_columns
    assert np.array_equal(ours.genelevel.to_numpy(), gene_values)
    assert np.array_equal(ours.seglevel[seg_columns].to_numpy(), seg_values)
    shared = [i for i, name in enumerate(state_columns) if name.endswith(("logmu", " p"))]
    assert np.array_equal(ours.perstate[[state_columns[i] for i in shared]].to_numpy(), state_values[:, shared])


@departs("cnamaste's bin table adds each clone's state, logmu and p; its state table drops the copies per state")
def test_the_integer_copy_tables_have_calicost_s_columns(decoded: Decoded, referee: Referee) -> None:
    """The bin and state tables' columns, cnamaste's against CalicoST's: cnamaste decodes copies per bin, not per
    state, and reports each bin's state alongside its copies."""
    ours, theirs = get_tables(referee, decoded)
    assert (list(ours.seglevel.columns), list(ours.perstate.columns)) == (theirs[3], theirs[5])
