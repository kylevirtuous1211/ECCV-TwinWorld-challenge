"""Per-point geometry that a semantic label can be predicted from.

Measured on the two labelled Gold Coast scenes, a classifier over these
features reaches pooled mIoU 0.453 across scenes, against 0.112 for calling
everything wall and 0.349 for the obvious hand-written rules. Two features
carry almost all of it - height above ground and verticality - and the ones
that sound equally reasonable do not:

  height above ground   permutation importance +0.204
  verticality (coarse)  +0.181
  verticality (fine)    +0.097
  planarity (coarse)    +0.045
  everything else       +0.012 or less, planarity itself exactly 0.000

That last one is worth keeping in mind before adding features: ground is the
*least* planar of the flat classes here, being landscaped rather than paved, so
"ground is a large flat region" is false on this data. Wall is the large flat
region.

What the geometry cannot do is separate wall from window. They sit at the same
height with the same orientation, and a classifier given only geometry calls
69% of window points wall. That is a ceiling, not a tuning problem: with
window and other at zero, perfect ground, wall and roof caps mIoU at 0.60.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

UP = np.array([0.0, 0.0, 1.0])
GROUND_TILE = 3.0
GROUND_WINDOW = 21.0
GROUND_PERCENTILE = 2.0
NEIGHBOURS = 30
COARSE_VOXEL = 0.15
COARSE_NEIGHBOURS = 100

# The neighbour block a single chunk is allowed to hold. 2 GiB leaves the
# eigendecomposition's temporaries room on top and still keeps the query batches
# large enough that the tree's own overhead does not show.
CHUNK_BYTES = 2 << 30

FEATURE_NAMES = ("height_above_ground", "verticality", "verticality_coarse",
                 "planarity", "planarity_coarse", "linearity")

# Appended when a colour is supplied. Kept separate because they are only
# available on reconstructed clouds: the ground-truth clouds carry x, y, z and a
# classification and nothing else, so a classifier fitted on ground truth could
# never have used them. See label_cloud.py --fit-on.
#
# The three raw channels alone carry brightness and hue mixed together, and a
# tree has to spend splits separating them. The four derived columns do that
# separation directly: two chromaticity coordinates, which are hue at any
# exposure, an excess-green index, which is the standard vegetation
# discriminator and is aimed straight at `other` - the class with the largest
# remaining gap, 0.293 against a ceiling of 0.656, and mostly plants - and
# intensity, which is what is left once hue is removed.
COLOUR_FEATURE_NAMES = ("red", "green", "blue",
                        "chroma_red", "chroma_green", "excess_green", "intensity",
                        "excess_green_region", "intensity_region", "intensity_spread")

# The side of the cell those last three are pooled over. Larger than
# COARSE_VOXEL because what they are for is different: the geometric coarse
# scale describes the structure a surface belongs to, and this one describes the
# *material* a patch is made of. A leaf is green and so is the tree around it; a
# brick is mottled and so is the wall.
REGION_VOXEL = 0.60


def region_statistics(points: np.ndarray, values: np.ndarray,
                      voxel: float = REGION_VOXEL) -> tuple[np.ndarray, np.ndarray]:
    """The mean and the spread of `values` over each point's REGION_VOXEL cell.

    A groupby on voxel keys rather than a neighbour query, because this is a
    material statistic and does not need to respect the surface: a 60 cm cell
    holds enough points for a mean to mean something, and the query it replaces
    would cost another KD-tree over fourteen million points.
    """
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    total = np.bincount(inverse, weights=values)
    mean = total / counts
    variance = np.clip(np.bincount(inverse, weights=values ** 2) / counts - mean ** 2, 0, None)
    return mean[inverse], np.sqrt(variance)[inverse]


def colour_features(colour: np.ndarray, points: np.ndarray | None = None) -> np.ndarray:
    """The raw channels, the four that separate hue from brightness, and three
    that describe the patch a point sits in rather than the point itself.

    The regional columns are only produced when `points` is given, because they
    are a property of a neighbourhood and there is no honest default for one.
    """
    total = np.clip(colour.sum(axis=1), 1e-6, None)
    chroma_red = colour[:, 0] / total
    chroma_green = colour[:, 1] / total
    excess_green = 2.0 * colour[:, 1] - colour[:, 0] - colour[:, 2]
    intensity = total / 3.0
    matrix = np.concatenate(
        [colour, np.stack([chroma_red, chroma_green, excess_green, intensity], axis=1)],
        axis=1)
    if points is None:
        return matrix

    region_green, _ = region_statistics(points, excess_green)
    region_intensity, intensity_spread = region_statistics(points, intensity)
    return np.concatenate(
        [matrix, np.stack([region_green, region_intensity, intensity_spread], axis=1)],
        axis=1)


def feature_names(with_colour: bool = False) -> tuple[str, ...]:
    return FEATURE_NAMES + COLOUR_FEATURE_NAMES if with_colour else FEATURE_NAMES


class SemanticsError(ValueError):
    pass


def ground_elevation(points: np.ndarray, tile: float = GROUND_TILE,
                     window: float = GROUND_WINDOW,
                     percentile: float = GROUND_PERCENTILE):
    """A ground surface over the xy plane, as a lookup from position to elevation.

    A low percentile of z per tile is not enough on its own: on a building
    footprint every point in the tile is roof, so the "ground" follows the roof
    up and every roof point reports a height above ground of nearly zero. A
    morphological opening - erode then dilate, over a window wider than the
    buildings - pushes the estimate down to the surrounding terrain and then
    restores the terrain's own slope, which a plain erosion would flatten.

    Validated against the labels on both Gold Coast scenes: ground points land
    at a median of 0.25 and 0.47 m, roof points at 29.2 and 30.0 m.
    """
    if len(points) == 0:
        raise SemanticsError("no points to fit a ground surface to")
    origin = points[:, :2].min(axis=0)
    index = np.floor((points[:, :2] - origin) / tile).astype(np.int64)
    shape = index.max(axis=0) + 1

    flat = index[:, 0] * shape[1] + index[:, 1]
    order = np.argsort(flat, kind="stable")
    flat_sorted, z_sorted = flat[order], points[order, 2]
    starts = np.searchsorted(flat_sorted, np.arange(shape[0] * shape[1]), side="left")
    ends = np.searchsorted(flat_sorted, np.arange(shape[0] * shape[1]), side="right")

    grid = np.full(shape[0] * shape[1], np.nan)
    for cell, (start, end) in enumerate(zip(starts, ends)):
        if end > start:
            grid[cell] = np.percentile(z_sorted[start:end], percentile)
    grid = grid.reshape(shape)

    span = max(1, int(round(window / tile)))
    grid = _morphological(grid, span, np.nanmin)      # erode: reach past buildings
    grid = _morphological(grid, span, np.nanmax)      # dilate: give the terrain back
    return origin, tile, grid


def _morphological(grid: np.ndarray, span: int, reduce) -> np.ndarray:
    """A square-window min or max over a grid that has holes in it."""
    padded = np.pad(grid, span, mode="edge")
    out = np.empty_like(grid)
    for row in range(grid.shape[0]):
        window = padded[row:row + 2 * span + 1]
        for column in range(grid.shape[1]):
            block = window[:, column:column + 2 * span + 1]
            out[row, column] = np.nan if np.all(np.isnan(block)) else reduce(block)
    return out


def height_above_ground(points: np.ndarray, ground) -> np.ndarray:
    origin, tile, grid = ground
    index = np.clip(np.floor((points[:, :2] - origin) / tile).astype(np.int64),
                    0, np.array(grid.shape) - 1)
    elevation = grid[index[:, 0], index[:, 1]]
    elevation = np.where(np.isnan(elevation), np.nanmin(grid), elevation)
    return points[:, 2] - elevation


def local_shape(points: np.ndarray, query: np.ndarray, neighbours: int) -> tuple:
    """Verticality, planarity and linearity from local PCA.

    Verticality is |n . up| where n is the smallest eigenvector: 1 for a
    horizontal surface, 0 for a vertical one.

    Done in chunks, because the whole-array form is what stops this scaling. The
    neighbour block is one float64 triple per query point per neighbour, so a
    14.5M-point cloud at the coarse scale's 100 neighbours materialises 35 GB in
    one allocation and peaks near 95 GB once the index array and the covariance
    temporaries are counted. That fits on a 251 GB box and a 28M-point cloud - a
    Gold Coast scene at 4 cm - does not. Chunking makes the peak independent of
    the cloud, at no cost in the result: the blocks are per-query-point and never
    interact.
    """
    tree = cKDTree(points)
    neighbours = min(neighbours, len(points))
    query = np.asarray(query)
    verticality = np.empty(len(query))
    planarity = np.empty(len(query))
    linearity = np.empty(len(query))

    for start in range(0, len(query), _chunk_size(neighbours)):
        stop = min(start + _chunk_size(neighbours), len(query))
        _, index = tree.query(query[start:stop], k=neighbours, workers=-1)
        # k=1 comes back one-dimensional, which would drop the neighbour axis.
        blocks = points[index.reshape(stop - start, neighbours)]
        centred = blocks - blocks.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centred, centred) / centred.shape[1]

        values, vectors = np.linalg.eigh(covariance)      # ascending
        verticality[start:stop] = np.abs(vectors[:, :, 0] @ UP)

        total = values.sum(axis=1)
        total = np.where(total <= 0, 1e-12, total)
        planarity[start:stop] = (values[:, 1] - values[:, 0]) / total
        linearity[start:stop] = (values[:, 2] - values[:, 1]) / total

    return verticality, planarity, linearity


def _chunk_size(neighbours: int) -> int:
    """How many query points to hold neighbours for at once, by bytes not count.

    Keyed on the neighbour count so the fine and coarse scales get the same
    working-set size rather than the same number of rows, which is the thing
    that actually decides whether this fits.
    """
    return max(1, CHUNK_BYTES // (neighbours * 3 * 8))


def features(points: np.ndarray, ground=None,
             colour: np.ndarray | None = None) -> np.ndarray:
    """The feature matrix, one row per point, in `feature_names` order.

    Two neighbourhood scales, because they disagree usefully: the fine one
    describes the local surface and the coarse one describes the structure it
    belongs to, and the coarse verticality is the second most important feature
    of the set.

    `colour` appends the rendered RGB, and it is the first feature here that is
    not a function of the point positions. That matters because the module
    docstring records a limit geometry cannot pass: wall and window sit at the
    same height with the same orientation, so a classifier given only geometry
    calls 69% of window points wall. Nothing built out of eigenvalues fixes
    that. An image does.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise SemanticsError(f"points must be N by 3, got {points.shape}")

    ground = ground if ground is not None else ground_elevation(points)
    above = height_above_ground(points, ground)

    verticality, planarity, linearity = local_shape(points, points, NEIGHBOURS)

    keys = np.floor(points / COARSE_VOXEL).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    coarse_points = points[np.sort(first)]
    verticality_coarse, planarity_coarse, _ = local_shape(
        coarse_points, points, min(COARSE_NEIGHBOURS, len(coarse_points)))

    matrix = np.stack([above, verticality, verticality_coarse,
                       planarity, planarity_coarse, linearity], axis=1)
    if colour is None:
        return matrix

    colour = np.asarray(colour, dtype=np.float64)
    if colour.shape != (len(points), 3):
        raise SemanticsError(
            f"colour must be one RGB triple per point, got {colour.shape} for "
            f"{len(points)} points")
    return np.concatenate([matrix, colour_features(colour, points)], axis=1)
