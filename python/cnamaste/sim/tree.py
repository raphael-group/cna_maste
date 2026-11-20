"""The clone tree: its topology, the events on each edge, each clone's copies, and the truth tables.

`shared.unique` is CalicoST's star: a founder below normal carries `shared` events, each clone `unique`. `tree` and
`felsenstein` draw an unrooted binary topology on normal and the clones by stepwise insertion, uniform over the
`(2n - 3)!!` trees on `n` clones (Felsenstein 1978), rooted at normal. `tree` places `trunk`, `per_leaf` and
`per_internal` events per edge, or, given `cnas`, `1 + Poisson(lambda)` on each of the `m = 2k - 1` edges below normal,
redrawn until they total `cnas`, with no `founder` and the clone count `k` uniform on those with `m <= cnas`;
`felsenstein` `Poisson(m)` per edge plus one on each leaf edge, `m` spreading `expected_cnas` over the edges. A clone
carries every event on its path from the root, a later one overriding an earlier one where they overlap; `(1, 1)`
elsewhere. Under `loh = "irreversible"` an event keeps every haplotype its lineage has lost at 0 and changes the lineage
somewhere in its span; the edges are then drawn root first. `explicit` states the tree and its events, the LOH rule
checked on each.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
import pandas as pd

from cnamaste.sim.laws import draw_clone_count, draw_event_length
from cnamaste.sim.manifest import Event, Simulation, StatedLaw, get_tumour_clones

PLACEMENTS = 1_000
"""Placements tried per irreversible event, and edge counts per truth under `[cna] cnas`, before the draw is
refused."""


def get_placements() -> int:
    """Return the draws tried before a draw is refused, `PLACEMENTS`."""
    return PLACEMENTS


class CloneTree(NamedTuple):
    """`parent[node]`, in the order the topology was walked, and the events on the edge into each node."""

    parent: dict[str, str | None]
    edge_events: dict[str, tuple[Event, ...]]


def get_path(tree: CloneTree, node: str) -> list[str]:
    """Root to `node`, inclusive.

    Args:
        tree: the tree.
        node: the node.

    Returns:
        The nodes.
    """
    path = [node]
    while (up := tree.parent[path[-1]]) is not None:
        path.append(up)
    return path[::-1]


def get_lineage_events(tree: CloneTree, node: str) -> tuple[Event, ...]:
    """Every event on the path to `node`, in the order they arose.

    Args:
        tree: the tree.
        node: the node.

    Returns:
        The events.
    """
    return tuple(e for n in get_path(tree, node) for e in tree.edge_events[n])


def draw_topology(leaves: list[str], rng: np.random.Generator) -> dict[str, list[str]]:
    """An unrooted binary topology on `leaves` by stepwise insertion: three leaves meet at node 0, then each further
    leaf subdivides an edge chosen uniformly, edges sorted by their endpoints' names.

    Args:
        leaves: at least 3 distinct names.
        rng: the tree's generator.

    Returns:
        Each node's neighbours, in a preorder walk from node 0; internal nodes named `n<id>`.
    """
    adjacency: dict[int | str, list[int | str]] = {0: list(leaves[:3])}
    adjacency |= {leaf: [0] for leaf in leaves[:3]}
    for internal, leaf in enumerate(leaves[3:], start=1):
        edges: dict[frozenset[int | str], tuple[int | str, int | str]] = {}
        for a, near in adjacency.items():
            for b in near:
                edges.setdefault(frozenset((a, b)), (a, b))
        ordered = sorted(edges.values(), key=lambda edge: (str(edge[0]), str(edge[1])))
        first, second = ordered[int(rng.integers(len(ordered)))]
        adjacency[first].remove(second)
        adjacency[second].remove(first)
        adjacency[internal] = [first, second, leaf]
        adjacency[first].append(internal)
        adjacency[second].append(internal)
        adjacency[leaf] = [internal]

    def get_name(node: int | str) -> str:
        return node if isinstance(node, str) else f"n{node}"

    neighbours: dict[str, list[str]] = {}

    def walk(node: int | str, up: int | str | None) -> None:
        for down in adjacency[node]:
            if down != up:
                neighbours.setdefault(get_name(node), []).append(get_name(down))
                neighbours.setdefault(get_name(down), []).append(get_name(node))
                walk(down, node)

    walk(0, None)
    return neighbours


def draw_parents(simulation: Simulation, rng: np.random.Generator) -> dict[str, str | None]:
    """Each node's parent, rooted at normal: the trunk node is `founder` unless it is a clone or `[cna] cnas` is
    stated; ancestors `ancestor_k`.

    Args:
        simulation: the simulation.
        rng: the tree's generator.

    Returns:
        `parent[node]`, normal's None.
    """
    tumour = get_tumour_clones(simulation)
    if simulation.cna.mode == "shared.unique" or len(tumour) < 2:
        return {"normal": None, "founder": "normal"} | dict.fromkeys(tumour, "founder")
    neighbours = draw_topology(["normal", *tumour], rng)
    parent: dict[str, str | None] = {"normal": None}
    frontier = ["normal"]
    while frontier:
        node = frontier.pop()
        for other in neighbours[node]:
            if other not in parent:
                parent[other] = node
                frontier.append(other)
    names = {n: n for n in parent if n in {"normal", *tumour}}
    names |= {n: f"ancestor_{k}" for k, n in enumerate(sorted(n for n in parent if n not in names))}
    trunk = next(n for n, p in parent.items() if p == "normal")
    if simulation.cna.cnas is None:
        names[trunk] = trunk if trunk in tumour else "founder"
    return {names[n]: None if p is None else names[p] for n, p in parent.items()}


def draw_tree_clone_count(simulation: Simulation, rng: np.random.Generator) -> int:
    """A truth's tumour clone count `k`, `[cna] n_clones` stated or drawn; given `[cna] cnas` and a uniform law,
    uniform on the law's counts whose `2k - 1` edges below normal fit `cnas`.

    Args:
        simulation: the simulation.
        rng: the tree's generator.

    Returns:
        The count.
    """
    law, cnas = simulation.cna.n_clones, simulation.cna.cnas
    if not isinstance(law, StatedLaw):
        return law
    if cnas is None:
        return draw_clone_count(law, rng)
    feasible = [k for k in range(int(law.values["minimum"]), int(law.values["maximum"]) + 1) if 2 * k - 1 <= cnas]
    return feasible[int(rng.integers(len(feasible)))]


def draw_edge_counts(cnas: int, edges: int, rng: np.random.Generator) -> list[int]:
    """Each edge's CNAs, `1 + Poisson(lambda)`, redrawn until they total `cnas`.

    Args:
        cnas: `[cna] cnas`, at least `edges`.
        edges: the edges below normal, `2k - 1`.
        rng: the tree's generator.

    Returns:
        One count per edge.

    Raises:
        ValueError: where no draw in `get_placements()` totals `cnas`.
    """
    # NB lambda = (n - m) / m sets the expected total to n: acceptance about 1 / sqrt(2 pi (n - m)), 1 at n = m.
    # NB hd_tree's least is 0.16 (n = 9, m = 3): 1,000 draws all fail with probability 0.84^1000 < 1e-75.
    rate = (cnas - edges) / edges
    for _ in range(get_placements()):
        counts = 1 + rng.poisson(rate, size=edges)
        if int(counts.sum()) == cnas:
            return [int(c) for c in counts]
    raise ValueError(f"[cna] cnas: no edge counts in {get_placements()} draws total {cnas}")


def draw_placement(simulation: Simulation, rng: np.random.Generator) -> tuple[str, int, int]:
    """A chromosome by length, an event length, then a start uniform on the chromosome.

    Args:
        simulation: the simulation.
        rng: the tree's generator.

    Returns:
        `(chromosome, start, end)`, chromosome `1`-`22`.
    """
    lengths = np.asarray(simulation.chromosome_lengths, dtype=np.float64)
    index = int(rng.choice(lengths.size, p=lengths / lengths.sum()))
    span = min(draw_event_length(simulation.cna.length, rng), int(lengths[index]))
    start = int(rng.integers(0, int(lengths[index]) - span + 1))
    return str(index + 1), start, start + span


def get_lineage_states(lineage: tuple[Event, ...], chromosome: str, start: int, end: int) -> list[tuple[int, int]]:
    """The lineage's `(A, B)` on each piece of `[start, end)` its events cut it into: the last event covering a piece.

    Args:
        lineage: the events, in the order they arose.
        chromosome: the span's chromosome.
        start: the span's start.
        end: the span's end.

    Returns:
        One `(A, B)` per piece.
    """
    on = [e for e in lineage if e.chromosome == chromosome]
    cuts = {start, end} | {p for e in on for p in (e.start, e.end) if start < p < end}
    states = []
    for left, right in itertools.pairwise(sorted(cuts)):
        state = (1, 1)
        for e in on:
            if e.start <= left and right <= e.end:
                state = (e.a, e.b)
        states.append(state)
    return states


def get_admissible_states(states: tuple[tuple[int, int], ...], pieces: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The states an irreversible event may plant over `pieces`: a haplotype at 0 on any piece stays 0, and the state
    differs from the lineage's on at least one piece.

    Args:
        states: `[cna] states`.
        pieces: the lineage's states over the span.

    Returns:
        The admissible states, in `states`' order.
    """
    return [
        (a, b)
        for a, b in states
        if all((pa > 0 or a == 0) and (pb > 0 or b == 0) for pa, pb in pieces) and any((a, b) != p for p in pieces)
    ]


def draw_events(
    simulation: Simulation, count: int, lineage: tuple[Event, ...], rng: np.random.Generator
) -> tuple[Event, ...]:
    """`count` events: a placement, then a state uniform on `[cna] states`, or, under irreversible LOH, on those
    admissible over the lineage and the events before it.

    Args:
        simulation: the simulation.
        count: the events.
        lineage: the events above this edge.
        rng: the tree's generator.

    Returns:
        The events.

    Raises:
        ValueError: where no placement in `get_placements()` admits a state.
    """
    states = simulation.cna.states
    events: list[Event] = []
    irreversible = simulation.cna.loh == "irreversible"
    for _ in range(count):
        for _ in range(get_placements()):
            chromosome, start, end = draw_placement(simulation, rng)
            pieces = get_lineage_states((*lineage, *events), chromosome, start, end) if irreversible else []
            allowed = get_admissible_states(states, pieces) if irreversible else list(states)
            if allowed:
                break
        else:
            raise ValueError(f"[cna] loh = irreversible: no admissible state in {get_placements()} placements")
        a, b = allowed[int(rng.integers(len(allowed)))]
        events.append(Event(chromosome, start, end, int(a), int(b)))
    return tuple(events)


def get_explicit_tree(simulation: Simulation) -> CloneTree:
    """`[[cna.node]]` rooted at normal with `[[cna.event]]` on each node's edge, in the order stated.

    Args:
        simulation: an explicit simulation.

    Returns:
        The tree.

    Raises:
        ValueError: under irreversible LOH, where an event regains a haplotype its lineage lost or changes nothing.
    """
    parent: dict[str, str | None] = {"normal": None} | dict(simulation.cna.nodes)
    tree = CloneTree(parent, dict.fromkeys(parent, ()))
    # NB root first, so an event is checked against every event above it.
    for node, event in sorted(simulation.cna.events, key=lambda stated: len(get_path(tree, stated[0]))):
        pieces = get_lineage_states(get_lineage_events(tree, node), event.chromosome, event.start, event.end)
        if simulation.cna.loh == "irreversible" and not get_admissible_states(((event.a, event.b),), pieces):
            raise ValueError(f"[[cna.event]] {event} on {node}: inadmissible under loh = irreversible over {pieces}")
        tree.edge_events[node] = (*tree.edge_events[node], event)
    return tree


def draw_tree(simulation: Simulation, rng: np.random.Generator) -> CloneTree:
    """The clones' tree and the events on each edge, edges in name order, or root first under irreversible LOH.

    Args:
        simulation: the simulation.
        rng: the tree's generator.

    Returns:
        The tree.
    """
    cna = simulation.cna
    if cna.mode == "explicit":
        return get_explicit_tree(simulation)
    parent = draw_parents(simulation, rng)
    leaves = set(get_tumour_clones(simulation))
    edges = sorted(n for n in parent if parent[n] is not None)
    if cna.mode == "felsenstein":
        rate = (float(cna.expected_cnas) - len(leaves)) / len(edges)
        counts = {node: int(rng.poisson(rate)) + (node in leaves) for node in edges}
    elif cna.mode == "shared.unique":
        counts = {node: cna.shared if parent[node] == "normal" else cna.unique for node in edges}
    elif cna.cnas is not None:
        counts = dict(zip(edges, draw_edge_counts(cna.cnas, len(edges), rng), strict=True))
    else:
        counts = {
            node: cna.trunk if parent[node] == "normal" else (cna.per_leaf if node in leaves else cna.per_internal)
            for node in edges
        }
    tree = CloneTree(parent, {"normal": ()})
    order = sorted(counts) if cna.loh == "reversible" else sorted(counts, key=lambda n: (len(get_path(tree, n)), n))
    for node in order:
        lineage = () if cna.loh == "reversible" else get_lineage_events(tree, parent[node] or "normal")
        tree.edge_events[node] = draw_events(simulation, counts[node], lineage, rng)
    return tree


def get_clone_copies(
    tree: CloneTree, clones: tuple[str, ...], chromosome: np.ndarray, position: np.ndarray
) -> np.ndarray:
    """Each locus's `(A, B)` per clone: the last event on the clone's path covering it, `(1, 1)` where none does.

    Args:
        tree: the tree.
        clones: normal first, then the tumour clones.
        chromosome: (loci,) `1`-`22`.
        position: (loci,) bp.

    Returns:
        (loci, clones, 2) int64.
    """
    copies = np.ones((position.size, len(clones), 2), dtype=np.int64)
    query = np.asarray(chromosome).astype(str)
    for label, clone in enumerate(clones):
        for event in get_lineage_events(tree, clone) if clone != "normal" else ():
            covered = (query == event.chromosome) & (position >= event.start) & (position < event.end)
            copies[covered, label] = (event.a, event.b)
    return copies


def get_truth_profile(tree: CloneTree, clones: tuple[str, ...], chromosome_lengths: tuple[int, ...]) -> pd.DataFrame:
    """`truth_acn_profile.tsv`: `chr start end` per segment the events cut the genome into, then `A`, `B` per clone,
    normal last.

    Args:
        tree: the tree.
        clones: normal first, then the tumour clones.
        chromosome_lengths: chr1 first.

    Returns:
        The profile.
    """
    rows = []
    for index, length in enumerate(chromosome_lengths, start=1):
        cuts = {0, int(length)}
        for clone in clones[1:]:
            cuts |= {p for e in get_lineage_events(tree, clone) if e.chromosome == str(index) for p in (e.start, e.end)}
        rows += [(index, s, e) for s, e in itertools.pairwise(sorted(cuts))]
    profile = pd.DataFrame(rows, columns=["chr", "start", "end"])
    middle = ((profile["start"] + profile["end"]) // 2).to_numpy()
    copies = get_clone_copies(tree, clones, profile["chr"].astype(str).to_numpy(), middle)
    for label in [*range(1, len(clones)), 0]:
        profile[f"{clones[label]}_A_copy"] = copies[:, label, 0]
        profile[f"{clones[label]}_B_copy"] = copies[:, label, 1]
    return profile


def get_tree_table(tree: CloneTree) -> pd.DataFrame:
    """`truth_tree.tsv`: one row per edge, then one per event with the edge it arose on.

    Args:
        tree: the tree.

    Returns:
        The table.
    """
    columns = ["node", "parent", "chr", "start", "end", "A", "B"]
    edges = pd.DataFrame([{"node": n, "parent": p} for n, p in tree.parent.items()], columns=["node", "parent"])
    events = pd.DataFrame(
        [
            {
                "node": n,
                "parent": tree.parent[n],
                "chr": e.chromosome,
                "start": e.start,
                "end": e.end,
                "A": e.a,
                "B": e.b,
            }
            for n in tree.parent
            for e in tree.edge_events[n]
        ],
        columns=columns,
    )
    return pd.concat([edges, events], ignore_index=True)[columns]


def get_clone_tree(table: pd.DataFrame) -> CloneTree:
    """`truth_tree.tsv` read back as the tree it was written from (`get_tree_table`).

    Args:
        table: the table.

    Returns:
        The tree.
    """
    edges = table[table["chr"].isna()]
    parent = {str(r.node): None if pd.isna(r.parent) else str(r.parent) for r in edges.itertuples()}
    events = table.dropna(subset=["chr"])
    edge_events = {
        node: tuple(
            Event(str(r.chr), int(r.start), int(r.end), int(r.A), int(r.B))
            for r in events[events["node"] == node].itertuples()
        )
        for node in parent
    }
    return CloneTree(parent, edge_events)
