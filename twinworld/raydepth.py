"""Read depth where the ray meets the disc, not where the disc's centre is.

2DGS represents a surface as oriented discs, and the depth a pixel should be
told is the distance to the point where its ray crosses the disc it hit. gsplat
1.5.3 reports something else: the depth channel is packed per Gaussian
(`rendering.py:1537` concatenates `depths` from the projection onto the colour
tensor), so every pixel a splat covers is told the splat's *centre* depth. A
disc tilted 60 degrees spans 1.73 of its own radius in depth and is reported
as a constant.

The intersection is not missing, it is discarded. `RasterizeToPixels2DGSFwd.cu`
computes `s`, the hit point in the disc's own uv frame, uses it for the Gaussian
weight, and then reads depth from the packed channel.

This is an open upstream bug - nerfstudio-project/gsplat issues 477 and 863,
with a fix in PR 932 that had not landed as of gsplat 1.5.3. Rather than patch
site-packages, this module recovers the intersection from what the library
already returns: the CUDA forward hands back `median_ids`, the splat that
carried transmittance through one half for each pixel, and `meta` exposes the
ray transforms and the flatten ids that index it.

Scope: this corrects the depth that is *read out*, which is what fusion
unprojects. It does not correct the distortion accumulator or the depth-derived
normal target used during training - both are computed inside the kernel from
the same packed channel and need the kernel change for that.
"""

from __future__ import annotations

import torch

# Anything closer to degenerate than this and the ray is parallel to the disc,
# so the intersection is off at infinity and the centre depth is the safer read.
MINIMUM_CROSS_Z = 1e-9


def quaternion_to_rotation(quats: torch.Tensor) -> torch.Tensor:
    """Rotation matrices from wxyz quaternions, normalised on the way through."""
    q = quats / quats.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


@torch.no_grad()
def median_gaussian_ids(meta: dict, count: int, width: int, height: int) -> torch.Tensor:
    """Which Gaussian set each pixel's median depth, or -1 where none did.

    Calls the same CUDA forward `rasterization_2dgs` just called, because that
    is the only place `median_ids` exists - the autograd wrapper saves it for
    the backward pass and does not return it. Colours are a single zero channel:
    the transmittance walk that picks the median depends on opacity and geometry
    alone, so the colour values cannot change which splat is chosen.

    Under `no_grad` because the result is a set of indices. Which splat carried
    transmittance through one half is a discrete choice, so there is nothing to
    differentiate here even when the caller is mid-training and everything else
    on the path does need a gradient.
    """
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    placeholder = torch.zeros((1, count, 1), device=meta["means2d"].device)
    outputs = _make_lazy_cuda_func("rasterize_to_pixels_2dgs_fwd")(
        meta["means2d"], meta["ray_transforms"], placeholder, meta["opacities"],
        meta["normals"], None, None, width, height, meta["tile_size"],
        meta["isect_offsets"], meta["flatten_ids"])
    return outputs[6][0]


def intersection_depth(means: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor,
                       view_matrix: torch.Tensor, width: int, height: int,
                       meta: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth of the ray-disc hit for the splat that set each pixel's median.

    Returns the depth map and a mask that is false wherever no splat set the
    median or the ray grazed its disc, so the caller can fall back rather than
    substitute a number produced by dividing by nothing.
    """
    device = means.device
    ids = median_gaussian_ids(meta, len(means), width, height)
    usable = ids >= 0
    gaussian = meta["flatten_ids"][ids.clamp(min=0).long()].long()

    rows, columns = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    pixel_x, pixel_y = columns + 0.5, rows + 0.5           # kernel samples pixel centres

    transforms = meta["ray_transforms"][0][gaussian]        # rows are u_M, v_M, w_M
    plane_u = pixel_x[..., None] * transforms[..., 2, :] - transforms[..., 0, :]
    plane_v = pixel_y[..., None] * transforms[..., 2, :] - transforms[..., 1, :]
    crossing = torch.cross(plane_u, plane_v, dim=-1)

    usable = usable & (crossing[..., 2].abs() > MINIMUM_CROSS_Z)
    denominator = torch.where(crossing[..., 2:3].abs() > MINIMUM_CROSS_Z,
                              crossing[..., 2:3], torch.ones_like(crossing[..., 2:3]))
    hit = crossing[..., :2] / denominator                   # (u, v) in units of the disc's sigma

    rotation = quaternion_to_rotation(quats[gaussian])
    radius = scales[gaussian][..., :2]
    point = (means[gaussian]
             + hit[..., 0:1] * radius[..., 0:1] * rotation[..., :, 0]
             + hit[..., 1:2] * radius[..., 1:2] * rotation[..., :, 1])

    camera = point @ view_matrix[:3, :3].T + view_matrix[:3, 3]
    return camera[..., 2], usable


def intersection_normal_map(means: torch.Tensor, quats: torch.Tensor,
                            scales: torch.Tensor, view_matrix: torch.Tensor,
                            intrinsics: torch.Tensor, width: int, height: int,
                            meta: dict, fallback: torch.Tensor) -> torch.Tensor:
    """The surface orientation implied by the hit depths, as 2DGS's normal target.

    2DGS ties each splat's own normal to the normal of the surface its depth map
    describes. With the centre depth that target is degenerate: the depth is
    constant across every pixel a splat covers, so the unprojected points lie on
    a plane perpendicular to the view direction and the finite-difference normal
    is the view direction itself. The loss then rotates splats to face the
    camera, which is the opposite of what it is for.

    Read at the hit instead and the target becomes what the paper intends. It is
    not circular, which is the obvious worry: the normal at a pixel is a finite
    difference across *neighbouring* pixels, and neighbours are frequently set by
    different splats, so the target measures the orientation the local ensemble
    agrees on. Where one splat covers the whole neighbourhood the term does go
    quiet, and it should - there is nothing there to bring into agreement.

    `fallback` supplies the depth wherever the ray grazed its disc and no hit
    could be recovered, so the map has no holes for the finite difference to
    smear across.

    Returns H by W by 3, squeezed the way `rasterization_2dgs` squeezes its own
    `normals_from_depth`, so this is a drop-in replacement rather than something
    that merely broadcasts against the same loss.
    """
    from gsplat.utils import depth_to_normal

    hit, usable = intersection_depth(means, quats, scales, view_matrix,
                                     width, height, meta)
    depth = torch.where(usable, hit, fallback)
    camera_to_world = torch.linalg.inv(view_matrix)
    return depth_to_normal(depth[None, ..., None], camera_to_world[None],
                           intrinsics[None]).squeeze(0)


def plane_offset(means: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor,
                 view_matrix: torch.Tensor, width: int, height: int,
                 meta: dict) -> torch.Tensor:
    """How far each recovered hit sits off its own disc's plane.

    A self-check rather than a product: the hit is constructed inside the disc's
    tangent frame, so it must satisfy (point - centre) . normal = 0 to floating
    point. Anything larger means the transform rows or the scale convention have
    been read wrong, which would otherwise show up only as a slightly worse
    F-score and be blamed on the method.
    """
    device = means.device
    ids = median_gaussian_ids(meta, len(means), width, height)
    gaussian = meta["flatten_ids"][ids.clamp(min=0).long()].long()

    rows, columns = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    transforms = meta["ray_transforms"][0][gaussian]
    plane_u = (columns + 0.5)[..., None] * transforms[..., 2, :] - transforms[..., 0, :]
    plane_v = (rows + 0.5)[..., None] * transforms[..., 2, :] - transforms[..., 1, :]
    crossing = torch.cross(plane_u, plane_v, dim=-1)
    denominator = torch.where(crossing[..., 2:3].abs() > MINIMUM_CROSS_Z,
                              crossing[..., 2:3], torch.ones_like(crossing[..., 2:3]))
    hit = crossing[..., :2] / denominator

    rotation = quaternion_to_rotation(quats[gaussian])
    radius = scales[gaussian][..., :2]
    offset = (hit[..., 0:1] * radius[..., 0:1] * rotation[..., :, 0]
              + hit[..., 1:2] * radius[..., 1:2] * rotation[..., :, 1])
    return (offset * rotation[..., :, 2]).sum(-1).abs()
