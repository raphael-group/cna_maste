"""The planted phase: switches between consecutive SNPs of a chromosome at Numbat's law, as `recomb` computes it.

One phasing for all slices, as the SNPs are phased on the pseudobulk. Between consecutive SNPs the reported phase
switches with probability `(1 - exp(-2 nu d)) / 2`, `d` the genetic map distance (`recomb`) in `[phasing] unit`, with no floor:
`recomb`'s `min_prob` and `logphase_shift` are the inference's, not the draw's. A switched SNP's A and B are exchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cnamaste.recomb import assign_centiMorgans, compute_numbat_phase_switch_prob
from cnamaste.reference import get_reference_recomb_rates

UNITS = {"morgan": 100.0, "centimorgan": 1.0}
"""`[phasing] unit` -> cM per unit: `d` in Morgans is Haldane's recombination law, in cM Numbat's switch law."""


def get_units() -> dict[str, float]:
    """Return `[phasing] unit` -> cM per unit, morgan 100 and centimorgan 1."""
    return UNITS


def get_switch_probabilities(
    chromosome: np.ndarray, position: np.ndarray, geneticmap_file: Path, nu: float, unit: str
) -> np.ndarray:
    """Each SNP's probability of a phase switch from the SNP before it; 0 at each chromosome's first.

    Args:
        chromosome: (SNPs,) `1`-`22`, in logical order.
        position: (SNPs,) bp, in logical order.
        geneticmap_file: the GRCh38 genetic map.
        nu: switches per unit.
        unit: morgan or centimorgan.

    Returns:
        (SNPs,) probabilities.
    """
    loci = list(zip(chromosome.astype(int), position, strict=True))
    genetic_map = get_reference_recomb_rates(geneticmap_file)  # type: ignore[no-untyped-call]
    position_cm = assign_centiMorgans(loci, genetic_map)  # type: ignore[no-untyped-call]
    position_cm = position_cm / get_units()[unit]
    to_next = compute_numbat_phase_switch_prob(position_cm, loci, nu, min_prob=0.0)  # type: ignore[no-untyped-call]
    return np.concatenate([[0.0], to_next[:-1]])


def draw_switched(switch_probabilities: np.ndarray, chromosome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw of the phase: switches at `switch_probabilities`, their running parity per chromosome.

    Args:
        switch_probabilities: (SNPs,) per `get_switch_probabilities`.
        chromosome: (SNPs,) each SNP's chromosome.
        rng: the phase's generator.

    Returns:
        (SNPs,) True where A and B are exchanged.
    """
    switches = rng.random(switch_probabilities.size) < switch_probabilities
    switched = np.zeros(switch_probabilities.size, dtype=bool)
    for name in np.unique(chromosome):
        at = np.flatnonzero(chromosome == name)
        switched[at] = np.cumsum(switches[at]) % 2 == 1
    return switched
