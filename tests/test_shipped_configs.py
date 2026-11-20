"""What the two shipped configurations promise about the `(A, B)` decoder.

`config.yaml` and `zenodo_sim_config.yaml` are both run, by hand and by `batch_zenodo_run.sh`, and a decoder
variable that drifts between them makes every figure drawn from one incomparable with the other. These rows hold
them to the same values and hold both to keys the configuration knows; the decoder reads every setting from
`ab_decode`, so the code states no default of its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import yaml

from cnamaste.ab_decode.decode import decode_integer_copies
from cnamaste.config import YAMLConfig, config_defaults
from cnamaste.emission import CnaEmissionConfounders
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.qa.sims import get_project
from cnamaste.spatio_genomic_counts import PseudobulkCounts

SHIPPED = ("config.yaml", "zenodo_sim_config.yaml")
"""The configurations a run is started from, both of which decode integer `(A, B)`."""

DECODER_KEYS = tuple(sorted(config_defaults["ab_decode"]))
"""The `ab_decode` keys the configuration knows: what a shipped file may name."""


def block(name: str) -> dict[str, Any]:
    """The file's `ab_decode` section, empty where the file sets none."""
    with (get_project() / name).open() as handle:
        return dict(yaml.safe_load(handle).get("ab_decode") or {})


def case(seed: int = 5, n_obs: int = 40, n_clones: int = 3, n_states: int = 5) -> dict[str, Any]:
    """One clone-pseudobulk case: three clones over `n_obs` bins, two of them altered, the first two bins unbaselined."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(80.0, 120.0, (n_obs, n_clones))
    base[:2] = 0.0
    total = rng.integers(30, 60, (n_obs, n_clones)).astype(float)
    truth = rng.integers(0, 3, (n_obs, n_clones))
    depth = np.where(truth == 0, 1.0, np.where(truth == 1, 0.55, 1.45))
    p = np.where(truth == 0, 0.5, np.where(truth == 1, 0.08, 0.3))
    counts = rng.poisson(base * depth).astype(float)
    allele = rng.binomial(total.astype(int), p).astype(float)
    return {
        "X": np.stack([counts, allele], axis=1),
        "base": base,
        "total": total,
        "lengths": np.array([n_obs]),
        "alphas": np.full(n_states, 0.02),
        "taus": np.full(n_states, 60.0),
        "shift": np.log(rng.uniform(0.9, 1.1, n_clones)),
    }


def decode(given: dict[str, Any], settings: dict[str, Any], *, shifted: bool):
    """`given` decoded at `settings`, the keys `run_cnamaste` passes."""
    X = given["X"]
    n_clones = X.shape[2]
    return decode_integer_copies(
        PseudobulkCounts(X[:, 0].T, given["base"].T, X[:, 1].T, given["total"].T, None, {}),
        CnaHmmParameters(None, None, None, given["alphas"], given["taus"]),
        CnaEmissionConfounders(
            np.where(np.arange(n_clones) == 0, 1.0, 0.1)[:, None],
            np.reshape(given["shift"], (-1, 1)) if shifted else np.zeros((n_clones, 1)),
        ),
        given["lengths"],
        YAMLConfig({"ab_decode": {"max_iter": 6, **settings}}),
    )


def test_the_two_shipped_configs_set_the_same_decoder_keys() -> None:
    """Both files carry the same `ab_decode` keys, so neither falls back to a default the other states."""
    first, second = (set(block(name)) for name in SHIPPED)
    assert sorted(first - second) == [], f"{SHIPPED[0]} alone sets {sorted(first - second)}"
    assert sorted(second - first) == [], f"{SHIPPED[1]} alone sets {sorted(second - first)}"


@pytest.mark.parametrize("key", sorted(set().union(*(block(name) for name in SHIPPED))))
def test_the_two_shipped_configs_agree_on_each_decoder_variable(key: str) -> None:
    """One value per key either file sets, the rest `config_defaults`': a decode under either is the same decode."""
    first, second = (block(name) for name in SHIPPED)
    assert first[key] == second[key], f"{key}: {SHIPPED[0]} {first[key]!r}, {SHIPPED[1]} {second[key]!r}"


@pytest.mark.parametrize("shifted", [False, True], ids=["unshifted", "shifted"])
def test_the_decode_is_the_same_under_either_shipped_config(shifted: bool) -> None:
    """The decoder's own test of the above: the same copies, normal fractions, score and `Z_c`, to the last bit.

    Each file's `ab_decode` is the decode's configuration, so this fails on a value
    that differs between them whether or not the key comparison above catches it.
    """
    given = case()
    first, second = (decode(given, block(name), shifted=shifted) for name in SHIPPED)
    assert np.array_equal(first.copies, second.copies)
    assert np.array_equal(first.normal_fractions, second.normal_fractions)
    assert np.array_equal(first.log_mu_shift, second.log_mu_shift)
    assert first.score == second.score
