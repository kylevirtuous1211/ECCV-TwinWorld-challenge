"""Turn rendered depth maps into the surface cloud the geometry metric wants.

A trained Gaussian scene knows where the surface is, but it stores that
knowledge as blobs whose centres are wherever the optimiser put them. Rendering
depth and unprojecting it asks the scene the question the metric asks: along
this ray, where does the surface sit. The answer is on the surface by
construction, which is what `twinworld/pointcloud.py` warns Gaussian centres are
not.

Two filters decide what survives, and they pull in opposite directions.

  alpha gate            `RGB+ED` divides accumulated depth by alpha, so a pixel
                        no Gaussian covered reports a distance of order 1e9.
                        This gate is not a quality choice, it is required for
                        the arithmetic to mean anything.

  multi-view agreement  a point is kept only if other views, looking from
                        elsewhere, put a surface at the same place. This removes
                        floaters, at the cost of removing thinly-observed real
                        surface with them.

The second is left off by default. On this benchmark the F-score thresholds are
5, 10 and 20 cm against a 2 cm-dense ground truth, and coverage is worth far
more than precision - measured on the TUM dev scenes, driving precision to 1.0
moves scene_000's F-score from 0.196 to 0.202 while driving recall to 1.0 moves
it to 0.804. So agreement filtering has to earn its place against a measurement
rather than be assumed to help.

Everything here is numpy, so it runs and is tested in the plain venv with no
CUDA anywhere near it. Only the depth rendering upstream needs a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twinworld.metrics import voxel_downsample

DEFAULT_ALPHA_MIN = 0.5
DEFAULT_VOXEL = 0.02


class FusionError(ValueError):
    pass


@dataclass
class View:
    """A camera, reduced to what unprojection needs."""
    view_matrix: np.ndarray      # 4x4 world to camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def rotation(self) -> np.ndarray:
        return self.view_matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.view_matrix[:3, 3]


def unproject(depth: np.ndarray, alpha: np.ndarray, view: View,
              alpha_min: float = DEFAULT_ALPHA_MIN,
              near: float = 0.01, far: float = 1e6,
              image: np.ndarray | None = None):
    """World-space points for the pixels this view is confident about.

    The rasteriser's depth is the camera-space z of the surface, not the
    distance along the ray, so the pixel offsets scale by z directly. Pixel
    centres sit at index + 0.5, matching the projection the rasteriser inverted.

    Pass `image` and the pixel each point came from is sampled from it and
    returned alongside. That correspondence was always computed here and thrown
    away, and it is the only thing standing between the fused cloud and a
    per-point colour - which is the one feature that can separate wall from
    window, a pair `semantics.py` proves geometry cannot.
    """
    if depth.shape != alpha.shape:
        raise FusionError(f"depth {depth.shape} and alpha {alpha.shape} differ")
    if depth.shape != (view.height, view.width):
        raise FusionError(
            f"depth is {depth.shape} but the camera says {(view.height, view.width)}")
    if image is not None and image.shape[:2] != depth.shape:
        raise FusionError(
            f"image is {image.shape[:2]} but the depth is {depth.shape}")

    keep = (alpha >= alpha_min) & np.isfinite(depth) & (depth > near) & (depth < far)
    rows, columns = np.nonzero(keep)
    if len(rows) == 0:
        empty = np.zeros((0, 3))
        return (empty, empty) if image is not None else empty

    z = depth[rows, columns].astype(np.float64)
    x = (columns + 0.5 - view.cx) * z / view.fx
    y = (rows + 0.5 - view.cy) * z / view.fy
    camera = np.stack([x, y, z], axis=1)

    # world = R^T (camera - t), the inverse of the world-to-camera transform.
    points = (camera - view.translation) @ view.rotation
    if image is None:
        return points
    return points, image[rows, columns].astype(np.float64)


def project(points: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates and camera-space depth for world points."""
    camera = points @ view.rotation.T + view.translation
    z = camera[:, 2]
    safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    u = camera[:, 0] / safe * view.fx + view.cx
    v = camera[:, 1] / safe * view.fy + view.cy
    return u, v, z


def agreement_counts(points: np.ndarray, depths: list[np.ndarray],
                     alphas: list[np.ndarray], views: list[View],
                     relative_tolerance: float = 0.01,
                     alpha_min: float = DEFAULT_ALPHA_MIN,
                     chunk: int = 2_000_000) -> np.ndarray:
    """How many views independently place a surface where each point sits.

    A view only gets to vote where it has an opinion: the point must fall in
    frame, in front of the camera, on a pixel the view covered. Points outside
    a view's frustum are neither confirmed nor refuted by it, which is why this
    counts agreements rather than measuring a disagreement rate - the latter
    would punish surface that only one camera could ever see.
    """
    counts = np.zeros(len(points), dtype=np.int16)
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        for depth, alpha, view in zip(depths, alphas, views):
            u, v, z = project(block, view)
            columns = np.floor(u).astype(np.int64)
            rows = np.floor(v).astype(np.int64)
            inside = (z > 0) & (columns >= 0) & (columns < view.width) \
                & (rows >= 0) & (rows < view.height)
            if not inside.any():
                continue
            index = np.nonzero(inside)[0]
            rendered = depth[rows[index], columns[index]]
            covered = alpha[rows[index], columns[index]] >= alpha_min
            close = np.abs(rendered - z[index]) <= relative_tolerance * np.abs(z[index])
            counts[start + index[covered & close & np.isfinite(rendered)]] += 1
    return counts


def fuse(depths: list[np.ndarray], alphas: list[np.ndarray], views: list[View],
         alpha_min: float = DEFAULT_ALPHA_MIN,
         voxel_size: float = DEFAULT_VOXEL,
         min_views: int = 1,
         relative_tolerance: float = 0.01,
         far: float = 1e6,
         images: list[np.ndarray] | None = None):
    """Every view's depth, unprojected, deduplicated, and optionally cross-checked.

    `min_views=1` keeps everything the alpha gate admitted; anything higher
    demands that many independent views agree, this one included.

    Pass `images` - one per view, same size as its depth - and the colour of the
    pixel each surviving point came from comes back with the points. Every step
    from here is index-preserving, so the colour rides along rather than being
    re-derived: deduplication picks representative rows and the agreement test is
    a mask over them.

    A point several views saw keeps whichever view's colour survived
    deduplication, and colour is view-dependent. Averaging over the contributing
    views would be better and is not free, so this is the cheap version and the
    measurement should be read as a lower bound.
    """
    if not (len(depths) == len(alphas) == len(views)):
        raise FusionError(
            f"got {len(depths)} depths, {len(alphas)} alphas and {len(views)} views")
    if not depths:
        raise FusionError("fusing needs at least one view")
    if images is not None and len(images) != len(depths):
        raise FusionError(f"got {len(images)} images for {len(depths)} views")

    if images is None:
        clouds = [unproject(depth, alpha, view, alpha_min=alpha_min, far=far)
                  for depth, alpha, view in zip(depths, alphas, views)]
        colours = None
    else:
        pairs = [unproject(depth, alpha, view, alpha_min=alpha_min, far=far, image=image)
                 for depth, alpha, view, image in zip(depths, alphas, views, images)]
        clouds = [cloud for cloud, _ in pairs]
        colours = [colour for _, colour in pairs]

    filled = [index for index, cloud in enumerate(clouds) if len(cloud)]
    points = (np.concatenate([clouds[i] for i in filled], axis=0) if filled
              else np.zeros((0, 3)))
    if colours is not None:
        colours = (np.concatenate([colours[i] for i in filled], axis=0) if filled
                   else np.zeros((0, 3)))
    if len(points) == 0:
        return (points, colours) if colours is not None else points

    # Deduplicate before cross-checking: the metric voxelises at 2 cm anyway, so
    # the discarded points could not have scored, and the agreement test is the
    # expensive step.
    if voxel_size > 0:
        points, kept = voxel_downsample(points, voxel_size, return_index=True)
        if colours is not None:
            colours = colours[kept]
    if min_views > 1:
        counts = agreement_counts(points, depths, alphas, views,
                                  relative_tolerance=relative_tolerance,
                                  alpha_min=alpha_min)
        agreed = counts >= min_views
        points = points[agreed]
        if colours is not None:
            colours = colours[agreed]
    return (points, colours) if colours is not None else points
