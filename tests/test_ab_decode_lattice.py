"""`ab_decode.lattice`: the pairs and the neutral pair, each against a case written out by hand."""

from __future__ import annotations

import numpy as np

from cnamaste.ab_decode.lattice import get_ab_is_ploidy_neutral, get_ab_lattice

CAPS = {"max_total_copy": 4, "max_allele_copy": 3}


def as_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    """The pairs as a set of tuples."""
    return {(int(a), int(b)) for a, b in pairs}


def test_the_lattice_is_the_pairs_written_out() -> None:
    """At caps 4 and 3: folded `A >= B`, unfolded each reflected, B fraction alone one pair per fraction; no (0, 0),
    no allele above 3, no total above 4. Fails where a cap, the fold or the reflection is misread."""
    folded = {(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1)}
    assert as_set(get_ab_lattice(params="mp", fold=True, **CAPS)) == folded
    unfolded = get_ab_lattice(params="mp", fold=False, **CAPS)
    assert as_set(unfolded) == folded | {(b, a) for a, b in folded}
    assert len(unfolded) == len(as_set(unfolded))
    assert as_set(get_ab_lattice(params="p", fold=True, **CAPS)) == {(1, 0), (1, 1), (2, 1), (3, 1)}


def test_the_neutral_pair_is_half_the_ploidy_on_each_allele_and_none_at_odd_ploidy() -> None:
    """(1, 1) at 2, (2, 2) at 4, none at 3. Fails where an odd ploidy takes the pairs nearest balance."""
    pairs = get_ab_lattice(params="mp", fold=False, **CAPS)
    assert as_set(pairs[get_ab_is_ploidy_neutral(pairs, 2)]) == {(1, 1)}
    assert as_set(pairs[get_ab_is_ploidy_neutral(pairs, 4)]) == {(2, 2)}
    assert not get_ab_is_ploidy_neutral(pairs, 3).any()
