"""`cnamaste.sim` draws dev_tree easy r0 bitwise as it was first drawn (`e4bbe03f`), given that draw's switch
probabilities: Haldane's `(1 - exp(-2 nu d)) / 2` on `np.interp`'s cM, which `recomb` replaces in the package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cnamaste.qa.sims import get_fixture_hash, get_resources, grch38
from cnamaste.reference import get_reference_recomb_rates
from cnamaste.sim.draw import get_snp_ids, realize
from cnamaste.sim.manifest import get_simulation
from cnamaste.sim.perfect import get_snp_loci
from cnamaste.sim.write import write_realization

REFERENCES = grch38()
pytestmark = pytest.mark.skipif(REFERENCES is None, reason="no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")


def get_haldane_switch_probabilities(chromosome: np.ndarray, position: np.ndarray, nu: float) -> np.ndarray:
    """The first draw's switch probabilities: `0.5 (1 - exp(-2 nu d))`, `d` the cM distance by `np.interp`.

    Args:
        chromosome: (SNPs,) contigs.
        position: (SNPs,) bp.
        nu: switches per cM.

    Returns:
        (SNPs,) probabilities, 0 at each chromosome's first SNP.
    """
    assert REFERENCES is not None
    genetic_map = get_reference_recomb_rates(REFERENCES / get_resources()["geneticmap_file"])  # type: ignore[no-untyped-call]
    contig = np.asarray(genetic_map["chrom"]).astype(int)
    probabilities = np.zeros(position.size)
    for name in np.unique(chromosome):
        at = np.flatnonzero(chromosome == name)
        on = contig == name
        cm = np.interp(position[at], np.asarray(genetic_map["pos"])[on], np.asarray(genetic_map["pos_cm"])[on])
        probabilities[at[1:]] = 0.5 * (1 - np.exp(-2 * nu * np.diff(cm)))
    return probabilities


@pytest.mark.slow
def test_dev_tree_easy_r0_is_its_first_draw_byte_for_byte(tmp_path: Path) -> None:
    """Every stage but the switch law -- tree, layout, barcodes, counts, phase draw, files -- against the fixture."""
    simulation = get_simulation("dev_tree_easy")
    chromosome, position = get_snp_loci(get_snp_ids(simulation))
    switches = get_haldane_switch_probabilities(chromosome, position, simulation.nu)
    realization = next(realize(simulation, switches, None))
    assert get_fixture_hash(write_realization(realization, simulation, tmp_path)) == "e4bbe03f"
