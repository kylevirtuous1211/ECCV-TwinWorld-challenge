"""The labelled occupancy volume that the Gold Coast term actually rewards.

`score_local.py` sends Gold Coast only to `confusion`, so the semantic term is a
truth-driven nearest-neighbour lookup inside 10 cm with **no precision term**: a
predicted point that no truth point is near costs exactly nothing. What maximises
it is therefore not a surface but a labelled *covering set*, and the question is
which set of points covers the most truth for a fixed budget.

Two facts decide it, and both are measured in `RESEARCH_LOG.md` T2.

**The lattice.** Filling a region is not covering it: the worst-covered point of a
filled region sits at the lattice's covering radius, `s*sqrt(3)/2` for simple
cubic and `a*sqrt(5)/4` for body-centred cubic. BCC is the thinnest covering
lattice in three dimensions (Bambah 1954, the lattice A3*), so at a fixed covering
guarantee it costs 1.86 times fewer points than simple cubic - 349 points per
cubic metre against 649 for a guaranteed 10 cm match.

**The thickness is worth almost nothing.** Coarsening the lattice to buy a thicker
shell loses, monotonically in the covering radius, because the truth is a sheet
rather than a filling: 3 to 17% of it lies between 30 cm and a metre from our
surface, while the coarsening that pays for that reach costs 40 to 70 points of
covering. The shipped recipe is therefore the finest lattice that fits the budget
with the thinnest shell that closes the gap, `bcc a=0.19` dilated by its eight
nearest neighbours.

    from twinworld.lattice import build
    nodes, labels = build(points, labels,
                          {"lattice": "bcc", "spacing": 0.19, "shell": 0.17})
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from twinworld.metrics import SEMANTIC_CLASS_INDICES

def read_labelled(source) -> tuple[np.ndarray, np.ndarray]:
    """A labelled cloud, from a path or from an open binary stream.

    The stream form is what lets a cloud be read straight out of a built
    submission zip, which is where the shipped Gold Coast surfaces come from:
    `make_submission.py --cap-cloud` thins them during packing, so the surface the
    volume is built on only exists inside the archive.
    """
    data = PlyData.read(str(source) if isinstance(source, (str, Path)) else source)["vertex"].data
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    labels = np.asarray(data["classification"]).astype(np.uint8)
    return points, labels


# ---------------------------------------------------------------------- lattices

def sc_nodes(points: np.ndarray, spacing: float) -> np.ndarray:
    """Occupied simple-cubic cells, as integer indices."""
    return np.unique(np.floor(points / spacing).astype(np.int64), axis=0)


def sc_positions(index: np.ndarray, spacing: float) -> np.ndarray:
    return (index + 0.5) * spacing


SC_NEIGHBOURS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                          (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int64)
# Face neighbours generate the lattice, so growing with six offsets reaches the
# same set as growing with twenty-six and costs a quarter of the memory. It just
# needs more rounds: a cell at Euclidean distance `r` is at Manhattan distance at
# most `r*sqrt(3)`, which is where the step counts below come from.


def bcc_nodes(points: np.ndarray, spacing: float) -> np.ndarray:
    """Nearest body-centred-cubic node of each point, as integer half-unit indices.

    In units of `a/2` the BCC lattice is exactly the integer triples whose three
    coordinates share a parity, so a node is `(a/2) * index` with index all-even
    or all-odd. The nearest node to a point is one of two candidates - round to
    the nearest all-even triple, and round to the nearest all-odd one - which is
    what makes the assignment exact rather than a search.
    """
    half = points / (spacing / 2.0)
    even = 2 * np.rint(half / 2.0)
    odd = 2 * np.rint((half - 1.0) / 2.0) + 1.0
    pick = (((half - even) ** 2).sum(axis=1) <= ((half - odd) ** 2).sum(axis=1))
    index = np.where(pick[:, None], even, odd).astype(np.int64)
    return np.unique(index, axis=0)


BCC_NEIGHBOURS = np.array(
    [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=np.int64)
# The eight body-diagonal offsets are the nearest neighbours and they generate the
# lattice on their own - two of them compose to an axis step of two half-units -
# so the axis neighbours are redundant for growth. In half-units a diagonal step
# changes every coordinate by exactly one, which makes the reachable set after
# `k` steps the Chebyshev ball of radius `k`, and that is the bound used below.


def bcc_positions(index: np.ndarray, spacing: float) -> np.ndarray:
    return index * (spacing / 2.0)


LATTICES = {
    #                                                     covering  nodes per   cells
    #  name    nodes       positions       neighbours     radius/u  spacing^3   per unit
    "sc": (sc_nodes, sc_positions, SC_NEIGHBOURS, np.sqrt(3) / 2, 1.0, 1.0),
    "bcc": (bcc_nodes, bcc_positions, BCC_NEIGHBOURS, np.sqrt(5) / 4, 2.0, 0.5),
}
# The fourth field is the covering radius in units of `spacing` and the fifth the
# node count per `spacing^3`; together they say that BCC covers to 10 cm for 349
# points per cubic metre where simple cubic needs 649. The sixth is the size of
# one growth step in units of `spacing`, which is what turns a shell radius into a
# number of rounds.


def _encode(index: np.ndarray, origin: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Integer cell coordinates as one int64 key, so `unique` is a 1-D sort.

    `np.unique(..., axis=0)` lexsorts a three-column array and is several times
    slower and several times heavier than sorting one column of int64. The growth
    below runs `unique` on hundreds of millions of rows, so this is the difference
    between minutes and hours.
    """
    shifted = index - origin
    return (shifted[:, 0] * dims[1] + shifted[:, 1]) * dims[2] + shifted[:, 2]


def _decode(key: np.ndarray, origin: np.ndarray, dims: np.ndarray) -> np.ndarray:
    z = key % dims[2]
    rest = key // dims[2]
    y = rest % dims[1]
    x = rest // dims[1]
    return np.stack([x, y, z], axis=1) + origin


def grow(seeds: np.ndarray, neighbours: np.ndarray, steps: int) -> np.ndarray:
    """`steps` rounds of neighbour expansion, deduplicated after every round.

    Deduplicating each round is what keeps this affordable: the working set is
    bounded by the size of the region being filled rather than by the number of
    offsets raised to the number of steps.
    """
    if steps <= 0:
        return seeds
    origin = seeds.min(axis=0) - steps - 1
    dims = seeds.max(axis=0) + steps + 2 - origin
    if int(dims[0]) * int(dims[1]) * int(dims[2]) >= 2 ** 62:
        raise SystemExit("cell grid too large to key into one int64")
    current = np.unique(_encode(seeds, origin, dims))
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    jumps = neighbours @ strides
    for _ in range(steps):
        current = np.unique(np.concatenate([current] + [current + j for j in jumps]))
    return _decode(current, origin, dims)


# ---------------------------------------------------------------------- builders

def thin_surface(points: np.ndarray, labels: np.ndarray, voxel: float):
    keys = np.floor(points / voxel).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    index = np.sort(index)
    return points[index], labels[index]


def legacy_volume(points: np.ndarray, labels: np.ndarray, spacing: float,
                  dilate: int, shape: str) -> tuple[np.ndarray, np.ndarray]:
    """Exactly what `label_volume.py` builds, so the null is the null.

    Reproduced here rather than imported because the two differ in nothing that
    matters and a null measured with a different code path is not a null.
    """
    keys = np.floor(points / spacing).astype(np.int64)
    seed, cell = np.unique(keys, axis=0, return_inverse=True)
    cell = np.asarray(cell).ravel().astype(np.int64)
    counts = np.bincount(cell * 5 + labels, minlength=len(seed) * 5)
    seed_label = counts.reshape(len(seed), 5).argmax(axis=1).astype(np.uint8)

    steps = range(-dilate, dilate + 1)
    cube = np.array([(a, b, c) for a in steps for b in steps for c in steps])
    offsets = cube if shape == "box" else cube[np.abs(cube).sum(axis=1) <= dilate]
    origin = seed.min(axis=0) - dilate - 1
    dims = seed.max(axis=0) + dilate + 2 - origin
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    key = _encode(seed, origin, dims)
    grown = np.unique(np.concatenate([key + j for j in offsets @ strides]))
    grown = _decode(grown, origin, dims)

    _, nearest = cKDTree(seed).query(grown, workers=-1)
    return (grown + 0.5) * spacing, seed_label[nearest]


def offset_ball(lattice: str, spacing: float, radius: float,
                squash: float = 1.0) -> np.ndarray:
    """Lattice index offsets whose world displacement is inside the shell.

    `squash` divides the vertical component before the test, so `squash < 1`
    stretches the shell vertically and `squash > 1` flattens it. The
    misregistration this shell exists to absorb is horizontal - it is measured in
    xy and the classes it damages are wall and window, not ground and roof - so a
    shell that spends its points sideways is spending them where the error is.
    """
    unit = spacing if lattice == "sc" else spacing / 2.0
    reach = int(np.ceil(radius / unit)) + 1
    steps = np.arange(-reach, reach + 1)
    grid = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
    if lattice == "bcc":
        parity = grid % 2
        grid = grid[(parity[:, 0] == parity[:, 1]) & (parity[:, 1] == parity[:, 2])]
    world = grid * unit
    scaled = world.copy()
    scaled[:, 2] *= squash
    return grid[(scaled ** 2).sum(axis=1) <= radius ** 2 + 1e-12]


def build(points: np.ndarray, labels: np.ndarray, spec: dict,
          verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """A candidate cloud, from a labelled surface and a specification.

    Three kinds. `surface` thins the surface and stops. `dilate` reproduces
    `label_volume.py` exactly, which is what the two nulls need. `shell` puts down
    a lattice and grows every occupied node by the offsets inside the shell.

    The growth is done entirely in cell space against an explicit offset ball
    rather than by filtering candidates against a tree built on the surface. The
    two differ by less than one cell and the cell-space version is an order of
    magnitude cheaper, which is what makes a grid of variants affordable at all.
    """
    kind = spec.get("kind", "shell")
    if kind == "union":
        # A coarse lattice buys thickness, but past `a = 0.179` its covering radius
        # is outside the 10 cm match radius, so the region it fills is only
        # partly readable. A finer lattice with a thin shell restores the
        # guarantee near the surface, where our geometry is already right, and
        # the coarse one hedges the offset further out. Overlap between the two
        # is paid for twice and is the price of the two scales.
        blocks = [build(points, labels, part, verbose) for part in spec["parts"]]
        return (np.concatenate([b[0] for b in blocks], axis=0),
                np.concatenate([b[1] for b in blocks], axis=0))
    if kind == "surface":
        thin = spec.get("thin")
        return (points, labels) if not thin else thin_surface(points, labels, thin)
    if kind == "dilate":
        return legacy_volume(points, labels, float(spec["spacing"]),
                             int(spec["dilate"]), spec.get("shape", "cross"))

    name = spec.get("lattice", "sc")
    spacing = float(spec["spacing"])
    to_nodes, to_positions, _, _, _, _ = LATTICES[name]
    squash = float(spec.get("squash", 1.0))

    seeds = to_nodes(points, spacing)
    seed_label = nearest_label(to_positions(seeds, spacing), points, labels)

    shells = dict(spec.get("class_shell") or {})
    default_shell = float(spec.get("shell", 0.0))
    per_class = {c: float(shells.get(c, shells.get(str(c), default_shell)))
                 for c in SEMANTIC_CLASS_INDICES}

    # Distinct radii are grown separately, so a class that wants a thicker shell
    # costs only its own seeds rather than everything's.
    parts, part_labels = [], []
    for radius in sorted(set(per_class.values())):
        of_radius = np.isin(seed_label, [c for c, r in per_class.items() if r == radius])
        block = seeds[of_radius]
        if not len(block):
            continue
        offsets = offset_ball(name, spacing, radius, squash)
        origin = block.min(axis=0) - offsets.max() - 1
        dims = block.max(axis=0) + offsets.max() + 2 - origin
        strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
        key = _encode(block, origin, dims)
        grown = _decode(np.unique(np.concatenate(
            [key + jump for jump in offsets @ strides])), origin, dims)
        parts.append(grown)
        part_labels.append(radius)
        if verbose:
            print(f"    {name} s={spacing:.3f} r={radius:.2f} squash={squash}: "
                  f"{len(block):,} seeds x {len(offsets)} offsets -> {len(grown):,}",
                  flush=True)

    if len(parts) > 1:
        stack = np.concatenate(parts, axis=0)
        origin = stack.min(axis=0) - 1
        dims = stack.max(axis=0) + 2 - origin
        grown = _decode(np.unique(_encode(stack, origin, dims)), origin, dims)
    else:
        grown = parts[0]
    nodes = to_positions(grown, spacing)
    node_labels = nearest_label(nodes, to_positions(seeds, spacing), seed_label)

    union_thin = spec.get("union_thin")
    if union_thin:
        kept, kept_labels = thin_surface(points, labels, union_thin)
        nodes = np.concatenate([nodes, kept], axis=0)
        node_labels = np.concatenate([node_labels, kept_labels], axis=0)
    return nodes, node_labels


def nearest_label(query: np.ndarray, reference: np.ndarray,
                  reference_labels: np.ndarray) -> np.ndarray:
    _, index = cKDTree(reference).query(query, workers=-1)
    return np.asarray(reference_labels)[index]


# ----------------------------------------------------------------------- scoring
