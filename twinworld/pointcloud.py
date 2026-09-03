"""Write the point cloud the evaluator wants, which is not the one 3DGS writes.

A 3DGS checkpoint is also called `point_cloud.ply` and also has x, y, z, so the
two are easy to confuse and nothing downstream complains if you do. They are
different objects:

  3DGS point_cloud.ply   Gaussian centres plus scale, rotation, opacity and SH
                         coefficients. The centres sit off the surface - a
                         Gaussian is a blob with extent, and its centre is
                         wherever the optimiser put it to make the render right.

  submission PLY         samples of the reconstructed surface, x y z only, plus
                         a classification for Gold Coast. The geometry metric
                         crops to the ground-truth bounding box, voxel-downsamples
                         at 2 cm and takes F-scores at 5, 10 and 20 cm, so it is
                         measuring distance to a surface and nothing else.

Submitting the first where the second is wanted is scored, not rejected, which
makes it the expensive mistake. So this module writes the submission cloud from
explicit coordinates and refuses to carry anything the evaluator ignores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from twinworld.metrics import voxel_downsample

VALID_CLASSIFICATION = (0, 1, 2, 3, 4, 255)


class PointCloudError(ValueError):
    """The points would not survive the evaluator, so they are refused here."""


def write_submission_ply(path: Path, points: np.ndarray,
                         classification: np.ndarray | None = None) -> int:
    """Write `points` as the submission cloud and return how many were written.

    Binary little-endian with float32 coordinates, which the rules recommend to
    keep the upload small, and uchar classification, which is what the format
    states.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise PointCloudError(f"points must be an N by 3 array, got {points.shape}")
    if len(points) == 0:
        raise PointCloudError("a submission cloud cannot be empty")
    if not np.isfinite(points).all():
        bad = int((~np.isfinite(points)).any(axis=1).sum())
        raise PointCloudError(f"{bad} of {len(points)} points are not finite")

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if classification is not None:
        classification = np.asarray(classification)
        if classification.shape != (len(points),):
            raise PointCloudError(
                f"classification must be one label per point, got {classification.shape} "
                f"for {len(points)} points")
        invalid = set(np.unique(classification).tolist()) - set(VALID_CLASSIFICATION)
        if invalid:
            raise PointCloudError(
                f"classification contains {sorted(invalid)}, allowed values are "
                f"{list(VALID_CLASSIFICATION)}")
        fields.append(("classification", "u1"))

    vertex = np.empty(len(points), dtype=fields)
    vertex["x"] = points[:, 0].astype(np.float32)
    vertex["y"] = points[:, 1].astype(np.float32)
    vertex["z"] = points[:, 2].astype(np.float32)
    if classification is not None:
        vertex["classification"] = classification.astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False, byte_order="<").write(str(path))
    return len(vertex)


def read_gaussian_centres(path: Path) -> np.ndarray:
    """The x, y, z of a 3DGS checkpoint, named so the caller knows what they are.

    Useful as an initialisation or a sanity check. Not a surface point cloud -
    see this module's docstring for why passing these straight to
    `write_submission_ply` is a scored mistake rather than a rejected one.
    """
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def read_submission_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """The x, y, z and, where it carries one, the classification of a submission cloud."""
    vertex = PlyData.read(str(path))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    labelled = "classification" in {property.name for property in vertex.properties}
    return points, np.asarray(vertex["classification"]) if labelled else None


def thin_unscored_ply(source: Path, destination: Path, voxel: float) -> tuple[int, int]:
    """Rewrite a submission cloud at `voxel` spacing, carrying labels through.

    Only for clouds this phase's reference data does not contain.
    `downsample_clouds.py` refuses to thin a labelled cloud and is right to: the
    surviving point keeps whichever label it happened to have rather than one
    predicted at the new scale. That objection is about a cloud that gets scored.
    This is for the ones that do not, where the only thing asked of a cloud is
    that it is present and valid, and where every point past that is evaluator
    load the submission cannot be paid for.

    Returns how many points were read and how many were written.
    """
    points, classification = read_submission_ply(source)
    kept, index = voxel_downsample(points, voxel, return_index=True)
    written = write_submission_ply(
        destination, kept, None if classification is None else classification[index])
    return len(points), written
