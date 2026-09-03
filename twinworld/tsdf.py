"""Fuse depth in a volume instead of unprojecting it, to thin the surface.

Unprojecting every view's depth independently (`twinworld/fusion.py`) puts a
point wherever any view thought there was one, so where views disagree by a few
centimetres the result is a shell several voxels thick rather than a sheet.
Measured on scene_000: 2,760 m2 of predicted surface lands within 5 cm of the
truth while covering only 654 m2 of it, a ratio of about four to one. That
thickness is precision, and precision is what the fused cloud is now short of -
with 25M points the cloud is no longer starved of coverage, so the balance that
made hallucinated surface a bargain has moved.

A truncated signed distance volume averages the views' disagreement instead of
keeping all of it: each view votes on the signed distance to the surface near
its own depth reading, and the zero crossing lands between the votes. Marching
cubes then returns one sheet. This is what 2DGS does for its own surface
extraction, and it is deliberately not what this repository did first, because
on the sparse cloud a volumetric prior mostly buys hallucinated volume.

open3d is only installed in the reconstruction environment, so it is imported
inside the functions and the tests that need it skip elsewhere.
"""

from __future__ import annotations

import numpy as np

from twinworld.fusion import DEFAULT_ALPHA_MIN, View

DEFAULT_VOXEL_LENGTH = 0.04
DEFAULT_TRUNC_FACTOR = 4.0
DEFAULT_SPACING = 0.03


class TsdfError(RuntimeError):
    pass


def _inside(depth: np.ndarray, view: View,
            bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Which pixels see a surface inside the box, per pixel, without unprojecting twice."""
    rows, columns = np.mgrid[0:view.height, 0:view.width]
    z = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
    camera = np.stack([(columns + 0.5 - view.cx) * z / view.fx,
                       (rows + 0.5 - view.cy) * z / view.fy, z], axis=-1)
    world = (camera - view.translation) @ view.rotation
    lower, upper = bounds
    return np.all((world >= lower) & (world <= upper), axis=-1)


def _require_open3d():
    try:
        import open3d
    except ImportError as error:                      # pragma: no cover
        raise TsdfError(
            "TSDF fusion needs open3d, which lives in the reconstruction "
            "environment - run this from `source scripts/env.sh`") from error
    return open3d


def integrate(depths: list[np.ndarray], alphas: list[np.ndarray], views: list[View],
              voxel_length: float = DEFAULT_VOXEL_LENGTH,
              trunc_factor: float = DEFAULT_TRUNC_FACTOR,
              alpha_min: float = DEFAULT_ALPHA_MIN,
              depth_trunc: float = 1000.0,
              bounds: tuple[np.ndarray, np.ndarray] | None = None):
    """Integrate every view's depth into one volume and march its zero crossing.

    Pixels the alpha gate rejects are written as zero, which open3d reads as
    "no measurement" - the same gate the point fusion applies, for the same
    reason: `RGB+ED` divides by alpha, so an uncovered pixel carries a distance
    of order 1e9.

    `bounds` is the difference between minutes and hours. A radial limit is not
    enough: on scene_000 the real content lies within 110 m of the cameras, but
    hundreds of thousands of pixels report surface scattered out to 300 m, and
    every one of them allocates blocks. Unprojection shrugs that off because the
    evaluator crops before scoring; a volume pays for it up front. Cropping to
    the box the scene actually occupies is the same crop, applied early.
    """
    o3d = _require_open3d()
    if not (len(depths) == len(alphas) == len(views)):
        raise TsdfError(
            f"got {len(depths)} depths, {len(alphas)} alphas and {len(views)} views")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=trunc_factor * voxel_length,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

    for depth, alpha, view in zip(depths, alphas, views):
        keep = (alpha >= alpha_min) & np.isfinite(depth) & (depth > 0) & (depth < depth_trunc)
        if bounds is not None:
            keep &= _inside(depth, view, bounds)
        usable = np.where(keep, depth, 0.0).astype(np.float32)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            view.width, view.height, view.fx, view.fy, view.cx, view.cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((view.height, view.width, 3), np.uint8)),
            o3d.geometry.Image(np.ascontiguousarray(usable)),
            depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, np.ascontiguousarray(view.view_matrix))

    return volume.extract_triangle_mesh()


def sample_mesh(mesh, spacing: float = DEFAULT_SPACING, seed: int = 0) -> np.ndarray:
    """Uniform points over a mesh, dense enough that no 2 cm voxel is missed.

    This goes through `surface.sample_surface` rather than open3d's own
    `sample_points_uniformly`, which takes no seed in this build and so would
    make every export irreproducible. It also keeps one implementation of
    "uniform samples over triangles" in the codebase, shared with the draped
    mesh, so the two surface variants are sampled identically and differ only
    in where their triangles came from.
    """
    from twinworld.surface import sample_surface

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(triangles) == 0 or len(vertices) == 0:
        return np.zeros((0, 3))
    return sample_surface(vertices, triangles, spacing=spacing, seed=seed)


def fuse(depths: list[np.ndarray], alphas: list[np.ndarray], views: list[View],
         voxel_length: float = DEFAULT_VOXEL_LENGTH,
         trunc_factor: float = DEFAULT_TRUNC_FACTOR,
         alpha_min: float = DEFAULT_ALPHA_MIN,
         spacing: float = DEFAULT_SPACING,
         depth_trunc: float = 1000.0,
         bounds: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Depth maps in, one sheet of surface samples out.

    The two limits are not details. A volume allocates blocks wherever a reading
    says there is surface, so a stray at 900 m in a 45 m scene costs tens of
    gigabytes and hours, and a radial limit alone still admits everything
    scattered inside that radius. Pass `bounds` as well.
    """
    mesh = integrate(depths, alphas, views, voxel_length=voxel_length,
                     trunc_factor=trunc_factor, alpha_min=alpha_min,
                     depth_trunc=depth_trunc, bounds=bounds)
    return sample_mesh(mesh, spacing=spacing)
