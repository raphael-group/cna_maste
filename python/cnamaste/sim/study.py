"""A study: a `[tool.cnamaste.study.<name>]` grid of CNA counts and fixed CNA lengths over one simulation, a
simulation per pair, each of `realizations` truths, and a seeded order of its (simulation, realization) samples.

Group `g`, in the grid's order (`cnas` outer, `length_mb` inner), is `<simulation>_cnas<n>_length<L>mb`, its
`[sample] seed` the study's `seed + g`; its realization `k` is the sample `<group>_r<k>`, written as
`<group>_r<k>_<hash>/`. The order is a permutation drawn by `default_rng([seed, 2])`; a sample does not depend on it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from cnamaste.qa.sims import get_project
from cnamaste.sim.draw import Realization, stream
from cnamaste.sim.manifest import Simulation, get_checked_simulation, get_definition, get_merged_tables

STUDY_KEYS = ("simulation", "cnas", "length_mb", "realizations", "seed")
"""Every key a `[tool.cnamaste.study.<name>]` table states."""


def get_study_keys() -> tuple[str, ...]:
    """Return the keys a study table states, ("simulation", "cnas", "length_mb", "realizations", "seed")."""
    return STUDY_KEYS


def get_study_table(name: str) -> dict[str, Any]:
    """`[tool.cnamaste.study.<name>]`, as stated.

    Args:
        name: the study.

    Returns:
        The table.

    Raises:
        ValueError: where it leaves out one of `get_study_keys()`.
    """
    table: dict[str, Any] = tomllib.loads((get_project() / "pyproject.toml").read_text())["tool"]["cnamaste"]["study"][
        name
    ]
    missing = [k for k in get_study_keys() if k not in table]
    if missing:
        raise ValueError(f"[tool.cnamaste.study.{name}]: {', '.join(missing)}")
    return table


def get_study_simulations(name: str) -> tuple[Simulation, ...]:
    """The study's simulations, one per `(cnas, length_mb)` pair in grid order, each its simulation's table with
    `[cna] cnas`, a fixed `[cna.length]` and `[sample] realizations` and `seed` laid over it.

    Args:
        name: the study.

    Returns:
        The simulations.
    """
    table = get_study_table(name)
    base = get_definition(table["simulation"])
    pairs = [(n, length) for n in table["cnas"] for length in table["length_mb"]]
    simulations = []
    for group, (n, length) in enumerate(pairs):
        over = {
            "sample": {"seed": int(table["seed"]) + group, "realizations": int(table["realizations"])},
            "cna": {"cnas": int(n), "length": {"law": "fixed", "size": float(length) * 1e6}},
        }
        named = f"{table['simulation']}_cnas{n}_length{length}mb"
        simulations.append(get_checked_simulation(named, get_merged_tables(base, over)))
    return tuple(simulations)


def get_study_order(name: str) -> list[tuple[Simulation, int]]:
    """The study's samples, `(simulation, realization)`, in the order its seed permutes them.

    Args:
        name: the study.

    Returns:
        Every sample once.
    """
    table = get_study_table(name)
    samples = [(s, k) for s in get_study_simulations(name) for k in range(s.realizations)]
    # NB [seed, 2]: a stream apart from each realization's [seed + g, 1] (`sim.draw`).
    order = np.random.default_rng([int(table["seed"]), 2]).permutation(len(samples))
    return [samples[i] for i in order]


def get_sample_name(simulation: Simulation, k: int) -> str:
    """A study sample's name, `<simulation>_r<k>`; its directory appends its fixture hash.

    Args:
        simulation: the sample's simulation.
        k: its realization.

    Returns:
        The name.
    """
    return f"{simulation.name}_r{k}"


def draw_study_sample(simulation: Simulation, k: int, out: Path) -> tuple[Realization, Path]:
    """Realization `k` of `simulation`, drawn alone and written as `out/<simulation>_r<k>_<hash>/`.

    Args:
        simulation: a study simulation.
        k: the realization.
        out: the parent directory, existing.

    Returns:
        The realization and its directory.
    """
    realization, directory = next(stream(simulation._replace(name=get_sample_name(simulation, k)), {k}, out))
    assert directory is not None
    return realization, directory
