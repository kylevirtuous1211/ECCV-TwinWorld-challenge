"""Read the COLMAP reconstruction each scene ships with.

The parsing rule for images.txt lives in `dataset._is_image_record` and is
shared, because getting it wrong inflates counts silently rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twinworld.dataset import _is_image_record


@dataclass(frozen=True)
class Pose:
    name: str
    camera_id: int
    rotation: np.ndarray       # world to camera
    translation: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        """The camera position in world coordinates."""
        return -self.rotation.T @ self.translation

    @property
    def viewing_direction(self) -> np.ndarray:
        """Unit vector the camera looks along, in world coordinates."""
        return self.rotation.T @ np.array([0.0, 0.0, 1.0])


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_poses(path: Path) -> list[Pose]:
    poses: list[Pose] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not _is_image_record(parts):
            continue
        poses.append(Pose(
            name=parts[9],
            camera_id=int(parts[8]),
            rotation=quaternion_to_rotation(np.array([float(v) for v in parts[1:5]])),
            translation=np.array([float(v) for v in parts[5:8]]),
        ))
    return poses


def read_point_tracks(path: Path) -> list[list[int]]:
    """For each 3D point, the image ids that observe it.

    Track length is the honest measure of how much redundancy a multi-view
    method has to work with: a point on two images can be triangulated but
    nothing about it can be checked.
    """
    tracks: list[list[int]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        observations = parts[8:]
        tracks.append([int(observations[i]) for i in range(0, len(observations), 2)])
    return tracks
