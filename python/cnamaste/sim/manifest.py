"""A simulation's definition: its `[tool.cnamaste.sim.<name>]` table, laid over the table it `extends`, checked and
read into typed tables.

The base table states every required key; a table that `extends` it states what differs, key by key. A law table
(one stating `law`) replaces the base's whole, so a law never inherits another's parameters; a list (`[[slice]]`,
`[[cna.node]]`, `[[cna.event]]`) replaces the base's.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, NamedTuple

from cnamaste.qa.sims import get_project

REQUIRED = {
    "sample": ("seed", "realizations", "vary"),
    "reference": ("baseline", "coverage", "snps"),
    "genome": ("chromosome_lengths",),
    "array": ("kind", "rows", "columns"),
    "barcodes": ("length", "suffix", "sample_id_bytes"),
    "model": (
        "admixture",
        "normal_frac",
        "dirichlet_concentration",
        "bb_overdispersion",
        "snp_dispersion",
        "counts_sampler",
    ),
    "cna": (
        "mode",
        "n_clones",
        "shared",
        "unique",
        "trunk",
        "per_leaf",
        "per_internal",
        "expected_cnas",
        "loh",
        "states",
        "length",
        "node",
        "event",
    ),
    "phasing": ("switch_errors", "nu", "unit"),
    "layout": ("shape", "overlap", "radius", "vertices", "jitter", "max_placements", "unplaced"),
    "config": ("base",),
}
"""Every table and key a simulation states; nothing the draw reads has a default in code."""

CHOICES = {
    ("sample", "vary"): ("counts", "truth"),
    ("array", "kind"): ("hex", "square"),
    ("model", "admixture"): ("read", "cell"),
    ("model", "counts_sampler"): ("gamma", "urn"),
    ("cna", "mode"): ("shared.unique", "tree", "felsenstein", "explicit"),
    ("cna", "loh"): ("reversible", "irreversible"),
    ("phasing", "unit"): ("morgan", "centimorgan"),
    ("layout", "shape"): ("polygons", "rectangles"),
    ("layout", "unplaced"): ("refuse", "stop", "drop"),
}
"""`(table, key)` -> the values it may take. `vary`: what a realization after the first redraws, its counts and
phase, or also its truth (clone count, layout, tree). `unplaced`: what a clone no placement clears does, refuse the
draw, stop with the clones placed, or drop it."""

LAWS: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {
    ("cna", "length"): {"fixed": ("size",), "exponential": ("mean", "minimum"), "lognormal": ("sigma", "minimum")},
    ("cna", "n_clones"): {"poisson": ("mean", "minimum"), "uniform": ("minimum", "maximum")},
    ("layout", "radius"): {"normal": ("mean", "sd", "minimum")},
    ("layout", "size"): {"loguniform": ("minimum", "maximum"), "lognormal": ("median", "sigma")},
}
"""`(table, key)` stated as a law table -> each law and the keys it reads. `[cna.length]` is always a law, a
lognormal also stating exactly one of `mean` or `median`; `n_clones` and `radius` may be numbers instead;
`[layout.size]`, a clone's size in spots, is optional and states `edge`."""

EDGES = ("grow", "clip")
"""`[layout.size] edge`: a clone at the array's edge grows until it claims its size on the array, or is sized on the
array continued past its edge and clipped."""

ALL = "all"
"""`[[slice]] clones = "all"`: every tumour clone the truth has; required where `n_clones` is a law."""


def get_choices() -> dict[tuple[str, str], tuple[str, ...]]:
    """Return `(table, key)` -> the values it may take, `CHOICES`."""
    return CHOICES


def get_laws() -> dict[tuple[str, str], dict[str, tuple[str, ...]]]:
    """Return `(table, key)` -> each law and the keys it reads, `LAWS`."""
    return LAWS


def get_edges() -> tuple[str, ...]:
    """Return the `[layout.size] edge` choices, grow or clip."""
    return EDGES


def get_all() -> str:
    """Return the `[[slice]] clones` value listing every tumour clone, `"all"`."""
    return ALL


class Event(NamedTuple):
    """One planted `(A, B)` over `[start, end)` of a chromosome."""

    chromosome: str
    start: int
    end: int
    a: int
    b: int


class StatedLaw(NamedTuple):
    """A law stated as a table: its name and the values it reads."""

    law: str
    values: dict[str, float]


class Length(NamedTuple):
    """`[cna.length]`: an event's length law, in bp; a key the law does not read is None.

    Args:
        law: fixed, exponential or lognormal.
        size: fixed's length.
        mean: exponential's mean, or a lognormal's stated mean.
        median: a lognormal's stated median.
        sigma: a lognormal's log spread.
        minimum: the floor of exponential and lognormal draws.
    """

    law: str
    size: float | None
    mean: float | None
    median: float | None
    sigma: float | None
    minimum: float | None


class Cna(NamedTuple):
    """`[cna]`: the tumour clones and their events.

    Args:
        mode: shared.unique (CalicoST's numcnas{shared}.{unique}), tree, felsenstein, or explicit (`node`, `event`).
        n_clones: tumour clones, the leaves; or the law of their count, drawn per truth.
        shared: shared.unique's events on every clone.
        unique: shared.unique's events on each clone alone.
        trunk: tree's events on the edge out of normal.
        per_leaf: tree's events on each edge into a clone.
        per_internal: tree's events on each edge into an ancestor.
        expected_cnas: felsenstein's expected events over the tree.
        cnas: tree's exact total of events, optional, in place of `trunk`, `per_leaf` and `per_internal`: `1 +
            Poisson` on each edge below normal, redrawn until the total is `cnas`, and no `founder`.
        loh: reversible, or irreversible (a lost haplotype stays lost).
        states: the (A, B) an event plants.
        length: the event length law.
        nodes: explicit's `[[cna.node]]`, `(name, parent)`, rooted at normal.
        events: explicit's `[[cna.event]]`, `(node, event)` in the order stated.
    """

    mode: str
    n_clones: int | StatedLaw
    shared: int
    unique: int
    trunk: int
    per_leaf: int
    per_internal: int
    expected_cnas: float
    cnas: int | None
    loh: str
    states: tuple[tuple[int, int], ...]
    length: Length
    nodes: tuple[tuple[str, str], ...]
    events: tuple[tuple[str, Event], ...]


class Model(NamedTuple):
    """`[model]`: the count laws.

    Args:
        admixture: read or cell, how a tumour spot's normal fraction mixes.
        normal_frac: a tumour spot's normal fraction.
        dirichlet_concentration: the gene shares' Dirichlet concentration, kappa.
        bb_overdispersion: the A count's beta-binomial rho; 0 is binomial.
        snp_dispersion: the SNP entries' NB dispersion, b.
        counts_sampler: gamma (normalized gammas) or urn (Polya urn), two draws of one law.
    """

    admixture: str
    normal_frac: float
    dirichlet_concentration: float
    bb_overdispersion: float
    snp_dispersion: float
    counts_sampler: str


class Layout(NamedTuple):
    """`[layout]`: each clone a jittered polygon placed in turn, or a block of a rectangular grid.

    Args:
        shape: polygons or rectangles.
        overlap: whether a clone may take another's spots.
        radius: a polygon's radius, a share of the array's shorter extent; or its law, drawn per clone.
        vertices: a polygon's vertices.
        jitter: each vertex's radius is `radius (1 + jitter U(-1, 1))`.
        max_placements: placements tried per clone.
        unplaced: refuse, stop or drop, for a clone no placement clears.
        normal_share: rectangles' normal block, a share of the frame's area.
        size: `[layout.size]`, a polygon's size law in spots, or None to size by `radius`.
        edge: `[layout.size] edge`, grow or clip.
    """

    shape: str
    overlap: bool
    radius: float | StatedLaw
    vertices: int
    jitter: float
    max_placements: int
    unplaced: str
    normal_share: float | None
    size: StatedLaw | None
    edge: str | None


class Region(NamedTuple):
    """`[[slice]] regions`: one clone's polygon, placed on its slice; `center` None draws it.

    Args:
        clone: the clone.
        center: in shares of the array's extent, from the slice's origin.
        radius: a share of the array's shorter extent.
        vertices: the polygon's vertices.
        jitter: the radii's relative spread.
    """

    clone: str
    center: tuple[float, float] | None
    radius: float
    vertices: int
    jitter: float


class Slice(NamedTuple):
    """`[[slice]]`: one slice's clones (`("all",)` for every one), its offset in the shared frame, in shares of the
    array's extent, and its stated regions."""

    clones: tuple[str, ...]
    offset: tuple[float, float]
    regions: tuple[Region, ...]


class Simulation(NamedTuple):
    """A simulation, read from its resolved table.

    Args:
        name: the table's name.
        seed: `[sample] seed`.
        realizations: `[sample] realizations`, count draws of one truth.
        vary: `[sample] vary`, counts or truth.
        baseline: `[reference]`'s per-gene normal expression, `lambda`.
        coverage: `[reference]`'s fitted count laws.
        snps: `[reference]`'s SNP ids.
        chromosome_lengths: `[genome]`'s, chr1 first.
        array: `[array]` kind, rows, columns.
        barcodes: `[barcodes]` length, suffix, sample_id_bytes.
        model: `[model]`.
        cna: `[cna]`.
        switch_errors: `[phasing]`'s, whether the written phase switches.
        nu: `[phasing]`'s switch rate per `unit`.
        unit: `[phasing]`'s map unit, morgan or centimorgan.
        layout: `[layout]`.
        slices: `[[slice]]`.
        config_base: `[config] base`, the run configuration a realization's `config.yaml` starts from.
        config: `[config.<section>]` tables laid over it.
    """

    name: str
    seed: int
    realizations: int
    vary: str
    baseline: Path
    coverage: Path
    snps: Path
    chromosome_lengths: tuple[int, ...]
    array: tuple[str, int, int]
    barcodes: tuple[int, str, int]
    model: Model
    cna: Cna
    switch_errors: bool
    nu: float
    unit: str
    layout: Layout
    slices: tuple[Slice, ...]
    config_base: Path
    config: dict[str, dict[str, Any]]


def get_sim_tables() -> dict[str, dict[str, Any]]:
    """Every `[tool.cnamaste.sim.<name>]` table, as stated.

    Returns:
        Name -> table.
    """
    tables: dict[str, dict[str, Any]] = tomllib.loads((get_project() / "pyproject.toml").read_text())["tool"][
        "cnamaste"
    ]["sim"]
    return tables


def get_merged_tables(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`over` laid on `base` key by key; `over` wins, a law table (one stating `law`) replaces the base's whole.

    Args:
        base: the extended table.
        over: the extending table.

    Returns:
        The merged table.
    """
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict) and "law" not in value:
            merged[key] = get_merged_tables(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_definition(name: str) -> dict[str, Any]:
    """`name`'s table laid over the one it `extends`, recursively.

    Args:
        name: the simulation.

    Returns:
        The resolved table, without `extends`.
    """
    table = dict(get_sim_tables()[name])
    parent = table.pop("extends", None)
    return table if parent is None else get_merged_tables(get_definition(parent), table)


def get_law_problems(definition: dict[str, Any]) -> list[str]:
    """What the law tables of `get_laws()` leave out; `[cna.length]` must be one, `[layout.size]` may be absent.

    Args:
        definition: the resolved table.

    Returns:
        One line per problem.
    """
    problems = []
    for (table, key), laws in get_laws().items():
        stated = definition.get(table, {}).get(key)
        if not isinstance(stated, dict):
            problems += [f"[{table}.{key}]: a law table"] if key == "length" else []
        elif stated.get("law") not in laws:
            problems.append(f"[{table}.{key}] law: one of {sorted(laws)}")
        else:
            problems += [f"[{table}.{key}] {k}" for k in laws[stated["law"]] if k not in stated]
    length = definition.get("cna", {}).get("length", {})
    if length.get("law") == "lognormal" and ("mean" in length) == ("median" in length):
        problems.append("[cna.length] lognormal: exactly one of mean or median")
    size = definition.get("layout", {}).get("size")
    if size is not None and size.get("edge") not in get_edges():
        problems.append(f"[layout.size] edge: one of {get_edges()}")
    return problems


def get_tree_problems(definition: dict[str, Any]) -> list[str]:
    """What an explicit tree breaks: unique nodes other than normal, each parent a node or normal and normal every
    node's root, its clones exactly `clone_0, ...`, and each event on a node within its chromosome with `A, B >= 0`.

    Args:
        definition: the resolved table.

    Returns:
        One line per problem.
    """
    cna = definition["cna"]
    names = [str(n.get("name")) for n in cna["node"]]
    parent = {str(n.get("name")): str(n.get("parent")) for n in cna["node"]}
    problems = [f"[[cna.node]] {n}: stated twice" for n in sorted({n for n in names if names.count(n) > 1})]
    problems += ["[[cna.node]] normal: the root, not a node"] if "normal" in parent else []
    for name in parent:
        seen = [name]
        while seen[-1] in parent and len(seen) <= len(parent):
            seen.append(parent[seen[-1]])
        if seen[-1] != "normal":
            problems.append(f"[[cna.node]] {name}: does not descend from normal")
    clones = sorted(n for n in names if n.startswith("clone_"))
    tumour = sorted(f"clone_{k}" for k in range(cna["n_clones"])) if isinstance(cna["n_clones"], int) else []
    problems += [f"[[cna.node]] clones {clones}: not {tumour}"] if clones != tumour else []
    lengths = definition["genome"]["chromosome_lengths"]
    for index, event in enumerate(cna["event"]):
        keys = ("node", "chr", "start", "length", "A", "B")
        if any(k not in event for k in keys):
            problems.append(f"[[cna.event]] {index}: states {keys}")
            continue
        within = 1 <= event["chr"] <= len(lengths) and 0 <= event["start"]
        within = within and 0 < event["length"] and event["start"] + event["length"] <= lengths[event["chr"] - 1]
        problems += [f"[[cna.event]] {index}: node {event['node']!r} unstated"] if event["node"] not in parent else []
        problems += [f"[[cna.event]] {index}: outside chr{event['chr']}"] if not within else []
        problems += [f"[[cna.event]] {index}: A, B >= 0"] if min(event["A"], event["B"]) < 0 else []
    return problems


def get_problems(definition: dict[str, Any]) -> list[str]:
    """What a resolved table leaves out or states outside its choices.

    Args:
        definition: the resolved table.

    Returns:
        One line per problem; empty where none.
    """
    problems = [f"[{t}] {k}" for t, keys in REQUIRED.items() for k in keys if k not in definition.get(t, {})]
    if problems:
        return problems
    problems += [
        f"[{t}] {k} {definition[t][k]!r}: one of {values}"
        for (t, k), values in get_choices().items()
        if definition[t][k] not in values
    ]
    problems += get_law_problems(definition)
    cna, layout, pieces = definition["cna"], definition["layout"], definition.get("slice", [])
    drawn = isinstance(cna["n_clones"], dict)
    leaves = int(cna["n_clones"]["minimum"] if drawn else cna["n_clones"])
    if cna["mode"] == "felsenstein" and not 2 <= leaves <= cna["expected_cnas"]:
        problems.append("[cna] felsenstein: 2 <= n_clones <= expected_cnas")
    uniform = not drawn or cna["n_clones"]["law"] == "uniform"
    if "cnas" in cna and (cna["mode"] != "tree" or not uniform or not 2 <= leaves or 2 * leaves - 1 > cna["cnas"]):
        problems.append("[cna] cnas: tree's, n_clones a number or uniform, k >= 2 (its minimum) and 2k - 1 <= cnas")
    if cna["mode"] == "explicit":
        problems += ["[cna] explicit: n_clones a number"] if drawn else get_tree_problems(definition)
        problems += ['[cna] explicit: [layout] unplaced = "refuse"'] if layout["unplaced"] != "refuse" else []
    if layout["shape"] == "rectangles" and ("normal_share" not in layout or "size" in layout):
        problems.append("[layout] rectangles: normal_share, and no [layout.size]")
    tumour = set() if drawn else {f"clone_{k}" for k in range(leaves)}
    if not pieces:
        problems.append("[[slice]]")
    for index, piece in enumerate(pieces):
        every = piece.get("clones") == get_all()
        named = set() if every else set(piece.get("clones", ()))
        regions = piece.get("regions", [])
        unknown = (named | {r["clone"] for r in regions}) - tumour
        if isinstance(layout["radius"], dict) and any("radius" not in r for r in regions):
            problems.append(f"[[slice]] {index}: a region states its radius where [layout] radius is a law")
        problems += [f"[[slice]] {index}: unknown clones {sorted(unknown)}"] if unknown and not every else []
        problems += [f'[[slice]] {index}: n_clones is a law, clones = "{get_all()}"'] if drawn and not every else []
    return problems


def get_stated_law(table: dict[str, Any] | float) -> StatedLaw | float:
    """A law table as a `StatedLaw`; a number as it is.

    Args:
        table: the stated value.

    Returns:
        The law, or the number.
    """
    if not isinstance(table, dict):
        return table
    return StatedLaw(str(table["law"]), {k: float(v) for k, v in table.items() if k not in {"law", "edge"}})


def get_length(table: dict[str, Any]) -> Length:
    """`[cna.length]` as a `Length`.

    Args:
        table: the stated `[cna.length]`.

    Returns:
        The law.
    """
    stated = {k: float(table[k]) if k in table else None for k in ("size", "mean", "median", "sigma", "minimum")}
    return Length(law=str(table["law"]), **stated)


def get_slice(piece: dict[str, Any], layout: dict[str, Any]) -> Slice:
    """A `[[slice]]`, its regions' polygon keys defaulting to `[layout]`'s.

    Args:
        piece: the stated slice.
        layout: the stated `[layout]`.

    Returns:
        The slice.
    """
    regions = tuple(
        Region(
            clone=str(r["clone"]),
            center=None if "center" not in r else (float(r["center"][0]), float(r["center"][1])),
            radius=float(r.get("radius", layout["radius"])),
            vertices=int(r.get("vertices", layout["vertices"])),
            jitter=float(r.get("jitter", layout["jitter"])),
        )
        for r in piece.get("regions", [])
    )
    clones = (get_all(),) if piece["clones"] == get_all() else tuple(piece["clones"])
    return Slice(clones, (float(piece["offset"][0]), float(piece["offset"][1])), regions)


def get_simulation(name: str) -> Simulation:
    """`name`'s resolved table, checked, as a `Simulation`; paths are the project's.

    Args:
        name: the simulation.

    Returns:
        The simulation.
    """
    return get_checked_simulation(name, get_definition(name))


def get_checked_simulation(name: str, definition: dict[str, Any]) -> Simulation:
    """A resolved table, checked, as the `Simulation` `name`; paths are the project's.

    Args:
        name: the simulation.
        definition: its resolved table (`get_definition`, or a study's group laid over one).

    Returns:
        The simulation.

    Raises:
        ValueError: where the table leaves out a required key or states a value outside its choices.
    """
    problems = get_problems(definition)
    if problems:
        raise ValueError(f"[tool.cnamaste.sim.{name}]: {'; '.join(problems)}")
    sample, reference, array, codes = (definition[k] for k in ("sample", "reference", "array", "barcodes"))
    model, cna, phasing, layout = (definition[k] for k in ("model", "cna", "phasing", "layout"))
    project = get_project()
    size = layout.get("size")
    counted = {"node", "event", "states", "length", "n_clones"}
    return Simulation(
        name=name,
        seed=int(sample["seed"]),
        realizations=int(sample["realizations"]),
        vary=str(sample["vary"]),
        baseline=project / reference["baseline"],
        coverage=project / reference["coverage"],
        snps=project / reference["snps"],
        chromosome_lengths=tuple(int(n) for n in definition["genome"]["chromosome_lengths"]),
        array=(str(array["kind"]), int(array["rows"]), int(array["columns"])),
        barcodes=(int(codes["length"]), str(codes["suffix"]), int(codes["sample_id_bytes"])),
        model=Model(**{k: model[k] for k in REQUIRED["model"]}),
        cna=Cna(
            **{k: cna[k] for k in REQUIRED["cna"] if k not in counted},
            n_clones=get_stated_law(cna["n_clones"]),  # type: ignore[arg-type]
            cnas=int(cna["cnas"]) if "cnas" in cna else None,
            states=tuple((int(a), int(b)) for a, b in cna["states"]),
            length=get_length(cna["length"]),
            nodes=tuple((str(n["name"]), str(n["parent"])) for n in cna["node"]),
            events=tuple(
                (
                    str(e["node"]),
                    Event(str(e["chr"]), int(e["start"]), int(e["start"] + e["length"]), int(e["A"]), int(e["B"])),
                )
                for e in cna["event"]
            ),
        ),
        switch_errors=bool(phasing["switch_errors"]),
        nu=float(phasing["nu"]),
        unit=str(phasing["unit"]),
        layout=Layout(
            shape=str(layout["shape"]),
            overlap=bool(layout["overlap"]),
            radius=get_stated_law(layout["radius"]),
            vertices=int(layout["vertices"]),
            jitter=float(layout["jitter"]),
            max_placements=int(layout["max_placements"]),
            unplaced=str(layout["unplaced"]),
            normal_share=float(layout["normal_share"]) if "normal_share" in layout else None,
            size=None if size is None else get_stated_law(size),  # type: ignore[arg-type]
            edge=None if size is None else str(size["edge"]),
        ),
        slices=tuple(get_slice(s, layout) for s in definition["slice"]),
        config_base=project / definition["config"]["base"],
        config={k: v for k, v in definition["config"].items() if isinstance(v, dict)},
    )


def get_tumour_clones(simulation: Simulation) -> tuple[str, ...]:
    """The tumour clones, `clone_0, ...`.

    Args:
        simulation: the simulation, its clone count resolved.

    Returns:
        Their names.

    Raises:
        TypeError: where `n_clones` is a law, drawn per truth.
    """
    if isinstance(simulation.cna.n_clones, StatedLaw):
        raise TypeError(f"{simulation.name}: n_clones is a law; draw the truth's count first")
    return tuple(f"clone_{k}" for k in range(simulation.cna.n_clones))


def get_listing_slices(simulation: Simulation, clone: str) -> list[bool]:
    """Per slice, whether it lists `clone`, by name or as `"all"`.

    Args:
        simulation: the simulation.
        clone: the clone.

    Returns:
        One flag per slice.
    """
    return [piece.clones == (get_all(),) or clone in piece.clones for piece in simulation.slices]
