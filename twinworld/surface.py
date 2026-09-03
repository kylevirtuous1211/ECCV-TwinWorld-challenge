"""A surface prior over a sparse cloud, and uniform samples from it.

The sparse COLMAP cloud occupies 12k to 33k of the ground truth's 4.7M to 11.9M
two-centimetre voxels, so its recall at 5 cm is 0.002 to 0.019 while its
precision at 20 cm is already 0.69 to 0.91. Under a metric with 5, 10 and 20 cm
thresholds that is not a noise problem, it is an absence problem: there is no
surface between the points.

Draping a triangulation over the points and sampling it densely invents that
surface. The interpolation is not evidence - nothing observed the space between
two COLMAP points - but measured on the four TUM dev scenes it takes
geometry_score from 0.1603 to 0.2861, because recall is worth far more than the
precision it costs. See `scripts/ablate_surface_prior.py`, which produces those
numbers.

The triangulation is 2.5D, over the xy footprint. One height per ground position
is what a downward-looking survey can support, and it needs no dependency beyond
scipy. It cannot represent an overhang, and on a facade it produces steep sliver
triangles rather than a vertical wall - which turns out to cover the facade well
enough at these thresholds.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

DEFAULT_SPACING = 0.03
MAX_SAMPLES = 40_000_000


class SurfaceError(ValueError):
    pass


def triangulate(points: np.ndarray) -> np.ndarray:
    """Triangles over the xy footprint, as indices into `points`."""
    if len(points) < 3:
        raise SurfaceError(f"a triangulation needs at least 3 points, got {len(points)}")
    return Delaunay(points[:, :2]).simplices


def edge_lengths(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The longest 3D edge of each triangle."""
    corners = points[triangles]
    return np.stack([
        np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1),
        np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1),
        np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1),
    ], axis=1).max(axis=1)


def filter_by_edge(points: np.ndarray, triangles: np.ndarray, limit: float) -> np.ndarray:
    """Keep only triangles that span less than `limit`.

    This is the difference between filling a hole and inventing a wall: a long
    edge in a sparse cloud is not a small gap in a surface, it is two surfaces
    the reconstruction never connected. Measured, gating below about 2 m costs
    more coverage than it saves precision, so the useful range is metres rather
    than the centimetres the phrase "high-confidence gap" suggests.
    """
    if not np.isfinite(limit):
        return triangles
    return triangles[edge_lengths(points, triangles) <= limit]


def sample_surface(points: np.ndarray, triangles: np.ndarray,
                   spacing: float = DEFAULT_SPACING, seed: int = 0,
                   max_samples: int = MAX_SAMPLES) -> np.ndarray:
    """Uniform-density samples over the triangles, roughly `spacing` apart.

    The density is set by spacing rather than by a total budget on purpose. A
    shared budget is distributed by area, so a handful of triangles spanning the
    whole scene absorb it and starve the real surface - which makes variants
    look different when only their sampling differed.
    """
    if len(triangles) == 0:
        return np.zeros((0, 3))
    corners = points[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    wanted = np.maximum(1, np.round(areas / (spacing ** 2)).astype(np.int64))
    if wanted.sum() > max_samples:
        wanted = np.maximum(1, (wanted * (max_samples / wanted.sum())).astype(np.int64))

    index = np.repeat(np.arange(len(triangles)), wanted)
    generator = np.random.default_rng(seed)
    u = generator.random(len(index))
    v = generator.random(len(index))
    folded = u + v > 1.0
    u[folded], v[folded] = 1.0 - u[folded], 1.0 - v[folded]

    origin = corners[index, 0]
    return origin + u[:, None] * (corners[index, 1] - origin) \
                  + v[:, None] * (corners[index, 2] - origin)


def drape(points: np.ndarray, spacing: float = DEFAULT_SPACING,
          edge_limit: float = float("inf"), seed: int = 0) -> np.ndarray:
    """Triangulate a sparse cloud and return dense samples of the result."""
    triangles = filter_by_edge(points, triangulate(points), edge_limit)
    return sample_surface(points, triangles, spacing=spacing, seed=seed)
