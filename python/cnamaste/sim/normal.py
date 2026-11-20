"""The normal baseline and the coverage laws, fit on the spots CalicoST's truth calls `normal`, pooled over samples.

- `normal_baseline.txt.gz`: one row per gene of the gene table the assay measures (`name2`, its first row), with
  `chrom`, `cdsStart`, `cdsEnd` and `lambda`, the gene's share of normal-spot UMI; a name the assay repeats sums its
  columns, and a gene no normal spot expresses is kept at 0.
- `normal_coverage.toml`: `[coverage.<law>]` per law of `get_coverage_laws()`: a spot's UMI and SNP reads, and a
  SNP's A, B and A + B reads summed over spots, each the family of smaller KS statistic; then the entry laws, each
  the scanned value whose draw is nearest the observed nonzero entries in total variation.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import kstest, nbinom, poisson

from cnamaste.io import read_numpy, read_text_table
from cnamaste.sim.laws import Coverage, draw_umis, get_mixture_pmf, get_snp_mixture

COVERAGE_LAWS = ("spot_umi", "spot_snp_umi", "snp_a", "snp_b", "snp_total")
"""The per-spot and per-SNP laws, in the order `normal_coverage.toml` states them."""

CONCENTRATIONS = (50.0, 70.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 140.0)
"""`kappa` scanned for the gene entry law."""

SNP_DISPERSIONS = (0.0, 0.25, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 4.0)
"""`b` scanned for the SNP entry law."""


def get_coverage_laws() -> tuple[str, ...]:
    """Return the coverage laws, `COVERAGE_LAWS`."""
    return COVERAGE_LAWS


def get_concentrations() -> tuple[float, ...]:
    """Return the scanned concentrations, `CONCENTRATIONS`."""
    return CONCENTRATIONS


def get_snp_dispersions() -> tuple[float, ...]:
    """Return the scanned SNP dispersions, `SNP_DISPERSIONS`."""
    return SNP_DISPERSIONS


class NormalCounts(NamedTuple):
    """Normal spots' counts, as a CalicoST sample states them.

    Args:
        umis: (spots, assay genes) UMI.
        genes: (assay genes,) names, repeats kept.
        a_counts: (spots, SNPs) A reads.
        b_counts: (spots, SNPs) B reads.
        snp_ids: (SNPs,) ids.
    """

    umis: sp.csr_matrix
    genes: np.ndarray
    a_counts: sp.csr_matrix
    b_counts: sp.csr_matrix
    snp_ids: np.ndarray


class Law(NamedTuple):
    """A fitted law and its evidence; a statistic not computed is None.

    Args:
        family: lognormal (`mu`, `sigma` of the log), negative_binomial (`mean`, `dispersion`, `var = mean +
            dispersion mean^2`), or an entry law's family.
        parameters: by name.
        ks: the KS statistic of the fit.
        ks_alternative: the other family's.
        n: the values fit.
        expressed: the observed nonzero share of an entry law.
    """

    family: str
    parameters: dict[str, float]
    ks: float | None
    ks_alternative: float | None
    n: int
    expressed: float | None


def read_normal_counts(sample: Path) -> NormalCounts:
    """A CalicoST sample's counts on the spots its truth calls `normal`, in `barcodes.txt`'s order.

    Args:
        sample: the sample directory.

    Returns:
        The counts.

    Raises:
        ValueError: where a barcode has no truth label, or the AnnData's spots are not `barcodes.txt`'s.
    """
    listed = read_text_table(sample / "barcodes.txt", header=None)  # type: ignore[no-untyped-call]
    barcodes = listed[0].astype(str).to_list()
    truth = read_text_table(sample / "truth_clone_labels.tsv", sep="\t", index_col=0)  # type: ignore[no-untyped-call]
    labels = truth.iloc[:, 0].reindex(barcodes)
    if labels.isna().any():
        raise ValueError(f"{sample}: {int(labels.isna().sum())} barcodes carry no truth label")
    normal = np.asarray(labels.to_numpy() == "normal")
    assay = anndata.read_h5ad(sample / "filtered_feature_bc_matrix.h5ad")
    if list(assay.obs_names) != barcodes:
        raise ValueError(f"{sample}: the AnnData's spots are not barcodes.txt's")
    alleles = [sp.load_npz(sample / f"cell_snp_{x}allele.npz").tocsr()[normal] for x in "AB"]
    snp_ids = read_numpy(sample / "unique_snp_ids.npy", allow_pickle=True)  # type: ignore[no-untyped-call]
    return NormalCounts(
        umis=sp.csr_matrix(assay.X)[normal],
        genes=np.asarray(assay.var_names).astype(str),
        a_counts=alleles[0],
        b_counts=alleles[1],
        snp_ids=np.asarray(snp_ids).astype(str),
    )


def get_pooled_counts(samples: list[Path]) -> NormalCounts:
    """The samples' normal spots, stacked in order.

    Args:
        samples: CalicoST sample directories sharing genes and SNPs.

    Returns:
        The pooled counts.

    Raises:
        ValueError: where a sample's genes or SNPs are not the first's.
    """
    counts = [read_normal_counts(s) for s in samples]
    for sample, read in zip(samples[1:], counts[1:], strict=True):
        if not (np.array_equal(read.genes, counts[0].genes) and np.array_equal(read.snp_ids, counts[0].snp_ids)):
            raise ValueError(f"{sample} does not share genes and SNPs with {samples[0]}")
    return NormalCounts(
        umis=sp.vstack([c.umis for c in counts]).tocsr(),
        genes=counts[0].genes,
        a_counts=sp.vstack([c.a_counts for c in counts]).tocsr(),
        b_counts=sp.vstack([c.b_counts for c in counts]).tocsr(),
        snp_ids=counts[0].snp_ids,
    )


def fit_baseline(pooled: NormalCounts, gene_table: Path) -> pd.DataFrame:
    """`gene chrom cdsStart cdsEnd lambda`, one row per gene of `gene_table` the assay measures; `lambda` sums to 1.

    Args:
        pooled: the normal spots.
        gene_table: CalicoST's `hgTables_hg38_gencode.txt`.

    Returns:
        The baseline.
    """
    per_column = np.asarray(pooled.umis.sum(axis=0)).ravel().astype(np.float64)
    per_gene = pd.Series(per_column).groupby(pooled.genes, sort=False).sum()
    table = pd.read_csv(gene_table, sep="\t", index_col=0).drop_duplicates("name2").set_index("name2")
    measured = table.loc[table.index.isin(per_gene.index)]
    total = per_gene.loc[measured.index].to_numpy()
    return pd.DataFrame(
        {
            "gene": measured.index.to_numpy(),
            "chrom": measured["chrom"].to_numpy(),
            "cdsStart": measured["cdsStart"].to_numpy(),
            "cdsEnd": measured["cdsEnd"].to_numpy(),
            "lambda": total / total.sum(),
        }
    )


def fit_count_law(values: np.ndarray) -> Law:
    """A lognormal on the positive values' logs (`ddof = 1`) and an NB by moments (Poisson where underdispersed),
    each scored by its KS statistic, the NB's at the observed support; the smaller's, the lognormal on a tie.

    Args:
        values: the counts.

    Returns:
        The law.
    """
    logs = np.log(np.asarray(values, dtype=np.float64)[np.asarray(values) > 0])
    mu, sigma = float(logs.mean()), float(logs.std(ddof=1))
    ks_lognormal = float(kstest((logs - mu) / sigma, "norm").statistic)
    counts = np.sort(np.asarray(values, dtype=np.float64))
    mean = float(counts.mean())
    dispersion = max((float(counts.var(ddof=1)) - mean) / mean**2, 0.0)
    support = np.unique(counts)
    empirical = np.searchsorted(counts, support, side="right") / counts.size
    if dispersion > 0:
        fitted = nbinom.cdf(support, 1.0 / dispersion, 1.0 / (1.0 + dispersion * mean))
    else:
        fitted = poisson.cdf(support, mean)
    ks_nb = float(np.max(np.abs(empirical - fitted)))
    if ks_lognormal <= ks_nb:
        return Law("lognormal", {"mu": mu, "sigma": sigma}, ks_lognormal, ks_nb, values.size, None)
    return Law("negative_binomial", {"mean": mean, "dispersion": dispersion}, ks_nb, ks_lognormal, values.size, None)


def fit_coverage(pooled: NormalCounts) -> dict[str, Law]:
    """`get_coverage_laws()`, fit on the pooled normal spots.

    Args:
        pooled: the normal spots.

    Returns:
        Name -> law.
    """
    trials = pooled.a_counts + pooled.b_counts
    per_spot = {"spot_umi": pooled.umis, "spot_snp_umi": trials}
    per_snp = {"snp_a": pooled.a_counts, "snp_b": pooled.b_counts, "snp_total": trials}
    laws = {k: fit_count_law(np.asarray(m.sum(axis=1)).ravel()) for k, m in per_spot.items()}
    return laws | {k: fit_count_law(np.asarray(m.sum(axis=0)).ravel()) for k, m in per_snp.items()}


def get_gene_umis(pooled: NormalCounts, genes: np.ndarray) -> sp.csr_matrix:
    """Pooled UMI as (spots, `genes`), a name's assay columns summed.

    Args:
        pooled: the normal spots.
        genes: the baseline's genes.

    Returns:
        The int64 UMI.
    """
    index = {name: i for i, name in enumerate(genes)}
    columns = np.array([index.get(name, -1) for name in pooled.genes])
    kept = np.flatnonzero(columns >= 0)
    gather = sp.csr_matrix((np.ones(kept.size), (kept, columns[kept])), shape=(columns.size, genes.size))
    matrix = (pooled.umis @ gather).tocsr()
    matrix.data = np.rint(matrix.data).astype(np.int64)
    return matrix


def get_observed_pmf(matrix: sp.csr_matrix, top: int | None) -> np.ndarray:
    """`P(k)`, `k = 0..top`, over every entry, zeros included; values above `top` counted at `top`.

    Args:
        matrix: the counts.
        top: the largest `k`; None is the largest value.

    Returns:
        The pmf.
    """
    values = np.asarray(matrix.data, dtype=np.int64)
    values = values[values > 0]
    top = int(values.max()) if top is None else top
    pmf = np.bincount(np.minimum(values, top), minlength=top + 1).astype(np.float64)
    pmf[0] = matrix.shape[0] * matrix.shape[1] - values.size
    return np.asarray(pmf / (matrix.shape[0] * matrix.shape[1]))


def get_nonzero_distance(model: np.ndarray, observed: np.ndarray) -> float:
    """Total variation between two pmfs given `k >= 1`; `model`'s mass past `observed`'s support counts in full.

    Args:
        model: the law's pmf.
        observed: the data's pmf.

    Returns:
        The distance.
    """
    top = observed.size - 1
    p = observed[1:] / observed[1:].sum()
    q = model[1 : top + 1] / (1.0 - model[0])
    return float(0.5 * (np.abs(p - q).sum() + max(1.0 - q.sum(), 0.0)))


def fit_entry_law(scanned: dict[float, np.ndarray], observed: np.ndarray, family: str, key: str, n: int) -> Law:
    """The scanned value whose pmf is nearest `observed` given `k >= 1`, the first on a tie.

    Args:
        scanned: value -> the law's pmf at it.
        observed: the data's pmf.
        family: the law's family.
        key: the value's name.
        n: the entries.

    Returns:
        The law, `expressed` the observed nonzero share.
    """
    distance = {value: get_nonzero_distance(pmf, observed) for value, pmf in scanned.items()}
    best = min(distance, key=distance.__getitem__)
    return Law(family, {key: best}, None, None, n, float(1.0 - observed[0]))


def fit_entry_laws(pooled: NormalCounts, baseline: pd.DataFrame, laws: dict[str, Law]) -> dict[str, Law]:
    """`gene_spot_umi`, the Dirichlet-multinomial `kappa`, each scanned value drawing the spots again at their own
    totals from one seed; `snp_spot_umi`, the SNP entries' NB `b`, at `laws`' spot SNP law and weight dispersion.

    Args:
        pooled: the normal spots.
        baseline: `fit_baseline`'s.
        laws: `fit_coverage`'s.

    Returns:
        Name -> law.
    """
    umis = get_gene_umis(pooled, baseline["gene"].to_numpy())
    observed = get_observed_pmf(umis, None)
    totals = np.asarray(umis.sum(axis=1)).ravel()
    weights = baseline["lambda"].to_numpy()[:, None]
    labels = np.zeros(totals.size, dtype=np.int64)
    seed = int(np.random.default_rng(0).integers(2**63))
    genes = {
        kappa: get_observed_pmf(
            draw_umis("gamma", totals, weights, labels, kappa, np.random.default_rng(seed)), observed.size - 1
        )
        for kappa in get_concentrations()
    }
    trials = (pooled.a_counts + pooled.b_counts).tocsr()
    observed_snps = get_observed_pmf(trials, None)
    spot, total = laws["spot_snp_umi"].parameters, laws["snp_total"].parameters
    coverage = Coverage((0.0, 0.0), (spot["mu"], spot["sigma"]), total.get("dispersion", 0.0))
    snps = {
        b: get_mixture_pmf(get_snp_mixture(coverage, b, trials.shape[1]), observed_snps.size - 1)
        for b in get_snp_dispersions()
    }
    return {
        "gene_spot_umi": fit_entry_law(
            genes, observed, "dirichlet_multinomial", "concentration", umis.shape[0] * umis.shape[1]
        ),
        "snp_spot_umi": fit_entry_law(
            snps, observed_snps, "negative_binomial", "dispersion", trials.shape[0] * trials.shape[1]
        ),
    }


def get_source_name(samples: list[Path]) -> str:
    """The samples as the headers name them: `CalicoST easy (2d4ce9a9) and hard (8797710b)` from `<name>_<hash>`.

    Args:
        samples: the sample directories.

    Returns:
        The name.
    """
    named = ["{} ({})".format(*s.name.rsplit("_", 1)) for s in samples]
    return "CalicoST " + (named[0] if len(named) == 1 else ", ".join(named[:-1]) + " and " + named[-1])


def get_baseline_text(baseline: pd.DataFrame, source: str, gene_table: Path) -> str:
    """`normal_baseline.txt`: two header lines, then the table, `lambda` at `%.9e`.

    Args:
        baseline: `fit_baseline`'s.
        source: `get_source_name`'s.
        gene_table: the gene table it was fit against.

    Returns:
        The text.
    """
    header = (
        f"# lambda: each gene's share of normal-spot UMI, fitted on the normal spots of {source}.\n"
        f"# {len(baseline)} genes of {gene_table.name} the assay measures; "
        f"{int((baseline['lambda'] == 0).sum())} at lambda 0; lambda sums to 1.\n"
    )
    return header + str(baseline.to_csv(sep="\t", index=False, float_format="%.9e"))


def get_coverage_text(laws: dict[str, Law], source: str) -> str:
    """`normal_coverage.toml`: a header line, then `[coverage.<law>]` per law, parameters at `.6g`, statistics at
    `.4g`.

    Args:
        laws: `fit_coverage`'s and `fit_entry_laws`'.
        source: `get_source_name`'s.

    Returns:
        The text.
    """
    lines = [f"# Per-spot and per-entry count laws fitted on the normal spots of {source}."]
    for name, law in laws.items():
        lines += ["", f"[coverage.{name}]", f'family = "{law.family}"']
        lines += [f"{k} = {v:.6g}" for k, v in law.parameters.items()]
        scores = {"ks": law.ks, "ks_alternative": law.ks_alternative, "expressed": law.expressed}
        lines += [f"{k} = {v:.4g}" for k, v in scores.items() if v is not None]
        lines += [f"n = {law.n}"]
    return "\n".join(lines) + "\n"
