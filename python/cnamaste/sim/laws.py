"""The simulation's laws: event lengths, depth and allele share under admixture, and the count laws.

Per slice and spot `s` of clone `c`:

- gene UMI: a total `N_s` lognormal (`[coverage.spot_umi]`) times the clone's library factor `sum_g lambda_g d_g(c)`;
  shares `p_s ~ Dirichlet(kappa q_c)`, `q_c` proportional to `lambda_g d_g(c)`; and `Multinomial(N_s, p_s)`, drawn
  by normalized gammas (`gamma`) or a Polya urn (`urn`). `d_g(c) = (1 - f) (A + B) / 2 + f` is the depth factor at
  the gene's `(A, B)`, `f` the normal fraction;
- SNP reads: each (SNP, spot) entry drawn independently from a mixture of NBs, `NB(M_s v_j / J, b)` with `M_s` the
  spot law `[coverage.spot_snp_umi]` and `v_j ~ Gamma(1/a, a)`, independent of copies; every nonzero value drawn
  exactly, with no tail cut;
- the haplotype-A count `BetaBinomial(n, share, rho)`, `share` the admixed haplotype-A share.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from numba import njit
from scipy.stats import gamma, nbinom, norm, poisson

from cnamaste.sim.manifest import Length, StatedLaw

QUADRATURE = 64
"""Quantiles a continuous law (spot depth, SNP weight) is summed over."""

MEAN_BINS = 4000
"""Log-spaced bins the SNP entries' means are pooled into."""

MAX_COUNT = 10_000_000
"""A quantile walk this long means the mean was not finite; the walk stops there."""


class Coverage(NamedTuple):
    """The fitted laws the counts are drawn from.

    Args:
        spot_umi: `(mu, sigma)` of a spot's log gene UMI.
        spot_snp_umi: `(mu, sigma)` of a spot's log SNP reads.
        snp_weight_dispersion: `a`, the SNP weights' gamma dispersion (`[coverage.snp_total] dispersion`).
    """

    spot_umi: tuple[float, float]
    spot_snp_umi: tuple[float, float]
    snp_weight_dispersion: float


class Mixture(NamedTuple):
    """One SNP entry's law: NB at `centres` weighted `mass` out of `entries`; the remaining mass at 0."""

    centres: np.ndarray
    mass: np.ndarray
    entries: float
    dispersion: float


def read_coverage(path: Path) -> Coverage:
    """The laws of a `normal_coverage.toml`.

    Args:
        path: the file.

    Returns:
        The `Coverage`.
    """
    tables = tomllib.loads(path.read_text())["coverage"]
    assert tables["spot_umi"]["family"] == tables["spot_snp_umi"]["family"] == "lognormal"
    return Coverage(
        spot_umi=(float(tables["spot_umi"]["mu"]), float(tables["spot_umi"]["sigma"])),
        spot_snp_umi=(float(tables["spot_snp_umi"]["mu"]), float(tables["spot_snp_umi"]["sigma"])),
        snp_weight_dispersion=float(tables["snp_total"]["dispersion"]),
    )


def get_lognormal_median(length: Length) -> float:
    """A lognormal length law's median: stated, or `mean exp(-sigma^2 / 2)`, so a stated mean is the law's.

    Args:
        length: the law.

    Returns:
        The median, in bp.
    """
    assert length.sigma is not None
    if length.median is not None:
        return length.median
    assert length.mean is not None
    return length.mean * float(np.exp(-(length.sigma**2) / 2.0))


def draw_event_length(length: Length, rng: np.random.Generator) -> int:
    """An event's length: fixed, `max(Exponential(mean), minimum)` or `max(LogNormal(log median, sigma), minimum)`.

    Args:
        length: the law.
        rng: the tree's generator.

    Returns:
        The length, in bp.
    """
    if length.law == "fixed":
        assert length.size is not None
        return int(length.size)
    assert length.minimum is not None
    if length.law == "lognormal":
        assert length.sigma is not None
        return int(max(rng.lognormal(np.log(get_lognormal_median(length)), length.sigma), length.minimum))
    assert length.mean is not None
    return int(max(rng.exponential(length.mean), length.minimum))


def draw_clone_count(law: StatedLaw, rng: np.random.Generator) -> int:
    """A truth's tumour clone count: `uniform` on `minimum..maximum`, or `Poisson(mean)` redrawn below `minimum`.

    Args:
        law: `[cna] n_clones`.
        rng: the tree's generator.

    Returns:
        The count.
    """
    if law.law == "uniform":
        return int(rng.integers(int(law.values["minimum"]), int(law.values["maximum"]) + 1))
    count = 0
    while count < int(law.values["minimum"]):
        count = int(rng.poisson(law.values["mean"]))
    return count


def draw_clone_radius(law: StatedLaw, rng: np.random.Generator) -> float:
    """A clone's polygon radius, `max(minimum, Normal(mean, sd))`.

    Args:
        law: `[layout] radius`.
        rng: the layout's generator.

    Returns:
        The radius, a share of the array's shorter extent.
    """
    return max(law.values["minimum"], float(rng.normal(law.values["mean"], law.values["sd"])))


def draw_clone_size(law: StatedLaw, rng: np.random.Generator) -> int:
    """A clone's size in spots, at least 1: `exp(U(log minimum, log maximum))` or `median exp(sigma Z)`, rounded.

    Args:
        law: `[layout.size]`.
        rng: the layout's generator.

    Returns:
        The size.
    """
    if law.law == "loguniform":
        low, high = np.log(law.values["minimum"]), np.log(law.values["maximum"])
        return max(1, round(float(np.exp(rng.uniform(low, high)))))
    return max(1, round(float(law.values["median"] * np.exp(law.values["sigma"] * rng.standard_normal()))))


def get_depth_factors(copies: np.ndarray, normal_frac: np.ndarray) -> np.ndarray:
    """Each locus's read-depth ratio to normal per clone, `(1 - f) (A + B) / 2 + f`; normal (clone 0) is 1.

    Args:
        copies: (loci, clones, 2) `(A, B)`.
        normal_frac: (clones,) each clone's normal fraction.

    Returns:
        (loci, clones) factors.
    """
    total = copies.sum(axis=2).astype(np.float64)
    tumour = 1.0 - normal_frac
    factor = tumour * total / 2.0 + (1.0 - tumour)
    factor[:, 0] = 1.0
    return np.ascontiguousarray(factor)


def get_allele_share(copies: np.ndarray, normal_frac: np.ndarray, admixture: str) -> np.ndarray:
    """The expected haplotype-A share of reads at `(A, B)`, admixed by reads or by cells.

    Args:
        copies: (loci, clones, 2) `(A, B)`.
        normal_frac: (clones,) each clone's normal fraction.
        admixture: read or cell.

    Returns:
        (clones, loci) shares.
    """
    a, b = copies[:, :, 0].T.astype(np.float64), copies[:, :, 1].T.astype(np.float64)
    frac = normal_frac[:, None]
    total = a + b
    if admixture == "cell":
        tumour = 1.0 - frac
        denominator = tumour * total + 2.0 * frac
        return np.asarray((tumour * a + frac) / np.where(denominator > 0, denominator, 1.0))
    share = np.where(total > 0, a / np.maximum(total, 1), 0.5)
    return np.asarray((1.0 - frac) * share + frac / 2.0)


def draw_spot_totals(law: tuple[float, float], n_spots: int, rng: np.random.Generator) -> np.ndarray:
    """Each spot's total from a lognormal `(mu, sigma)`, rounded, at least 1.

    Args:
        law: `(mu, sigma)` of the log.
        n_spots: the spots.
        rng: the counts' generator.

    Returns:
        (spots,) int64 totals.
    """
    return np.maximum(np.rint(rng.lognormal(law[0], law[1], n_spots)), 1).astype(np.int64)


def get_quantile_nodes(law: tuple[float, float]) -> np.ndarray:
    """`QUADRATURE` midpoint quantiles of a lognormal `(mu, sigma)`.

    Args:
        law: `(mu, sigma)` of the log.

    Returns:
        The nodes.
    """
    z = norm.ppf((np.arange(QUADRATURE) + 0.5) / QUADRATURE)
    return np.asarray(np.exp(law[0] + law[1] * z))


def get_snp_mixture(coverage: Coverage, snp_dispersion: float, n_snps: int) -> Mixture:
    """The SNP entry law over every (spot-depth node, weight node) pair, means pooled into `MEAN_BINS` log bins.

    Args:
        coverage: the fitted laws.
        snp_dispersion: `b`, each entry's NB dispersion.
        n_snps: the SNPs a spot's reads spread over.

    Returns:
        The `Mixture`.
    """
    a = coverage.snp_weight_dispersion
    quantiles = (np.arange(QUADRATURE) + 0.5) / QUADRATURE
    weights = np.asarray(gamma.ppf(quantiles, 1.0 / a, scale=a)) if a > 0 else np.ones(QUADRATURE)
    means = np.ravel(get_quantile_nodes(coverage.spot_snp_umi)[:, None] * (weights / n_snps)[None, :])
    edges = np.geomspace(means.min(), means.max() * (1 + 1e-12), MEAN_BINS + 1)
    which = np.clip(np.searchsorted(edges, means, side="right") - 1, 0, MEAN_BINS - 1)
    mass = np.bincount(which, np.ones(means.size), MEAN_BINS)
    centre = np.bincount(which, means, MEAN_BINS)
    kept = mass > 0
    return Mixture(centre[kept] / mass[kept], mass[kept], float(means.size), snp_dispersion)


def get_zero_probability(mu: np.ndarray, dispersion: float) -> np.ndarray:
    """`P(0)` of `NB(mu, dispersion)`, `var = mu + dispersion mu^2`; Poisson at 0.

    Args:
        mu: the means.
        dispersion: the NB dispersion.

    Returns:
        The probabilities.
    """
    if dispersion > 0:
        return np.asarray((1.0 + dispersion * mu) ** (-1.0 / dispersion))
    return np.asarray(np.exp(-mu))


def get_mixture_pmf(law: Mixture, top: int) -> np.ndarray:
    """`P(k)`, `k = 0..top`, of one entry of `law`, its zero-inflated mass at 0.

    Args:
        law: the entry law.
        top: the largest `k`.

    Returns:
        The pmf.
    """
    k = np.arange(top + 1)
    if law.dispersion > 0:
        r = 1.0 / law.dispersion
        components = nbinom.pmf(k[None, :], r, (r / (r + law.centres))[:, None])
    else:
        components = poisson.pmf(k[None, :], law.centres[:, None])
    pmf = law.mass @ components / law.entries
    pmf[0] += 1.0 - law.mass.sum() / law.entries
    return np.asarray(pmf)


@njit(cache=True)
def get_nb_quantile(u: float, mu: float, dispersion: float) -> int:
    """The smallest `k >= 1` with `CDF(k) > u` of `NB(mu, dispersion)`, given `u >= P(0)`, walked in log space.

    Args:
        u: the level.
        mu: the mean.
        dispersion: the NB dispersion; Poisson at 0.

    Returns:
        k.
    """
    if dispersion > 0.0:
        log_p = -np.log1p(dispersion * mu) / dispersion
        r = 1.0 / dispersion
        log_q = np.log(dispersion * mu) - np.log1p(dispersion * mu)
    else:
        log_p = -mu
        r = 0.0
        log_q = np.log(mu)
    cdf = np.exp(log_p)
    k = 0
    while cdf <= u and k < MAX_COUNT:
        k += 1
        if dispersion > 0.0:
            log_p += np.log((k - 1 + r) / k) + log_q
        else:
            log_p += log_q - np.log(k)
        cdf += np.exp(log_p)
    return k


@njit(cache=True)
def draw_zero_truncated(
    cumulative: np.ndarray,
    centres: np.ndarray,
    zeros: np.ndarray,
    dispersion: float,
    pick: np.ndarray,
    level: np.ndarray,
) -> np.ndarray:
    """`k >= 1` from a mixture of NBs: `pick` chooses the component, `level` is inverted above its `P(0)`.

    Args:
        cumulative: the components' cumulative weights given `k >= 1`.
        centres: the components' means.
        zeros: the components' `P(0)`.
        dispersion: the NB dispersion.
        pick: one uniform per draw.
        level: one uniform per draw.

    Returns:
        The draws.
    """
    out = np.empty(pick.size, dtype=np.int64)
    last = centres.size - 1
    for i in range(pick.size):
        c = min(np.searchsorted(cumulative, pick[i], side="right"), last)
        u = zeros[c] + level[i] * (1.0 - zeros[c])
        out[i] = max(get_nb_quantile(u, centres[c], dispersion), 1)
    return out


def draw_snp_reads(law: Mixture, n_spots: int, n_snps: int, rng: np.random.Generator) -> sp.csr_matrix:
    """A (spots, SNPs) matrix of independent entries of `law`: per row a `Binomial(n_snps, P(k >= 1))` count of
    nonzero entries at uniformly chosen SNPs, each drawn given `k >= 1`.

    Args:
        law: the SNP entry law.
        n_spots: the rows.
        n_snps: the columns.
        rng: the counts' generator.

    Returns:
        The reads, A + B.
    """
    zeros = get_zero_probability(law.centres, law.dispersion)
    weight = law.mass * (1.0 - zeros)
    nonzero = float(law.mass @ (1.0 - zeros) / law.entries)
    counts = rng.binomial(n_snps, nonzero, size=n_spots)
    n_drawn = int(counts.sum())
    values = draw_zero_truncated(
        np.cumsum(weight) / weight.sum(), law.centres, zeros, law.dispersion, rng.random(n_drawn), rng.random(n_drawn)
    )
    indices = np.concatenate([np.sort(rng.choice(n_snps, n, replace=False)) for n in counts]).astype(np.int32)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    return sp.csr_matrix((values, indices, indptr), shape=(n_spots, n_snps))


@njit(cache=True)
def get_table_sizes(total: int, kappa: float, uniforms: np.ndarray) -> np.ndarray:
    """Table sizes of `total` customers seated by a Chinese restaurant at concentration `kappa`.

    Customer `i` opens a table with probability `kappa / (kappa + i)`, else joins customer `floor(u i)`'s; labelling
    each table independently from `q` gives `DM(total, kappa q)`, the Polya urn.

    Args:
        total: the customers.
        kappa: the concentration.
        uniforms: (total, 2) uniforms.

    Returns:
        The occupied tables' sizes.
    """
    sizes = np.zeros(total, dtype=np.int64)
    table = np.zeros(total, dtype=np.int64)
    opened = 0
    for i in range(total):
        if uniforms[i, 0] * (kappa + i) < kappa:
            table[i] = opened
            opened += 1
        else:
            table[i] = table[int(uniforms[i, 1] * i)]
        sizes[table[i]] += 1
    return sizes[:opened]


def draw_umis(
    sampler: str,
    totals: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    """Row `s`: `DM(totals[s], kappa q)`, `q` column `labels[s]` of `weights` over its sum; a gene of weight 0 is never
    drawn. `gamma` draws normalized gammas, `O(genes)` per spot; `urn` a Polya urn, `O(UMIs)` per spot; the two consume
    `rng` differently.

    Args:
        sampler: gamma or urn.
        totals: (spots,) UMI per spot.
        weights: (genes, clones) expected shares, unnormalized.
        labels: (spots,) each spot's clone.
        kappa: the Dirichlet concentration.
        rng: the counts' generator.

    Returns:
        (spots, genes) UMIs.
    """
    q = weights / weights.sum(axis=0, keepdims=True)
    if sampler == "gamma":
        rows = []
        for total, clone in zip(totals, labels, strict=True):
            live = np.flatnonzero(q[:, clone] > 0)
            shares = rng.gamma(kappa * q[live, clone])
            row = np.zeros(q.shape[0], dtype=np.int64)
            if shares.sum() > 0:
                row[live] = rng.multinomial(int(total), shares / shares.sum())
            rows.append(sp.csr_matrix(row))
        return sp.vstack(rows, format="csr")
    cumulative = np.cumsum(q, axis=0).T.copy()
    indices, values, indptr = [], [], [0]
    for total, clone in zip(totals, labels, strict=True):
        cdf = cumulative[clone]
        if int(total) <= 0 or not cdf[-1] > 0:
            indptr.append(indptr[-1])
            continue
        sizes = get_table_sizes(int(total), float(kappa), rng.random((int(total), 2)))
        genes = np.searchsorted(cdf, rng.random(sizes.size) * cdf[-1], side="right")
        drawn = np.bincount(genes, weights=sizes, minlength=q.shape[0])
        kept = np.flatnonzero(drawn)
        indices.append(kept)
        values.append(drawn[kept].astype(np.int64))
        indptr.append(indptr[-1] + kept.size)
    return sp.csr_matrix(
        (
            np.concatenate(values) if values else np.zeros(0, np.int64),
            np.concatenate(indices).astype(np.int32) if indices else np.zeros(0, np.int32),
            np.asarray(indptr),
        ),
        shape=(len(totals), q.shape[0]),
    )


def draw_ab_counts(
    reads: sp.csr_matrix,
    share: np.ndarray,
    rho: float,
    switched: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """The written A and B counts: haplotype A `BetaBinomial(reads, share, rho)` of each entry, exchanged with B
    where the phase is switched.

    Args:
        reads: (spots, SNPs) A + B.
        share: (clones, SNPs) the haplotype-A share.
        rho: the beta-binomial overdispersion; binomial at 0.
        switched: (SNPs,) True where A and B are exchanged.
        labels: (spots,) each spot's clone.
        rng: the counts' generator.

    Returns:
        (A, B) counts.
    """
    entries = reads.tocoo()
    p = share[labels[entries.row], entries.col]
    if rho > 0:
        inner = (p > 0) & (p < 1)
        scale = 1 / rho - 1
        p = p.copy()
        p[inner] = rng.beta(p[inner] * scale, (1 - p[inner]) * scale)
    first = rng.binomial(entries.data, p)
    second = entries.data - first
    flip = switched[entries.col]
    at = (entries.row, entries.col)
    return (
        sp.csr_matrix((np.where(flip, second, first), at), shape=entries.shape),
        sp.csr_matrix((np.where(flip, first, second), at), shape=entries.shape),
    )
