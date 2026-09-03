"""What the challenge data contains, read from the data rather than hardcoded.

The scene list is not a constant here on purpose. Five TUM scenes and two Gold
Coast scenes ship with test poses but no reference photos and no ground truth -
they are the final-phase scenes - and a hardcoded list would quietly disagree
with the dataset the moment the organisers add or withhold one. Everything below
is derived from what is on disk, so a scene appearing or disappearing shows up as
a changed manifest instead of a rejected submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The submission uses different names for the datasets than the download does.
DATASET_DIRECTORIES = {
    "tum": "Data_TUM",
    "gold_coast": "Data_Goldcoast",
}

# Only Gold Coast point clouds carry semantic labels.
SEMANTIC_DATASET = "gold_coast"

SEMANTIC_CLASSES = {
    0: "ground",
    1: "wall",
    2: "roof",
    3: "window",
    4: "other",
    255: "ignore",
}


class DatasetError(RuntimeError):
    """The dataset on disk is not shaped the way the challenge describes."""


@dataclass(frozen=True)
class Frame:
    """One reference pose a submission has to render."""

    stem: str
    width: int
    height: int


@dataclass(frozen=True)
class Scene:
    dataset: str
    scene_id: str
    root: Path
    frames: tuple[Frame, ...]
    train_image_count: int
    has_reference_photos: bool
    has_ground_truth: bool

    @property
    def needs_classification(self) -> bool:
        return self.dataset == SEMANTIC_DATASET

    @property
    def is_withheld(self) -> bool:
        """No reference photos and no ground truth, so it is scored blind."""
        return not self.has_reference_photos and not self.has_ground_truth

    def submission_paths(self) -> list[str]:
        """Every path this scene must contribute to the zip, in zip order."""
        prefix = f"{self.dataset}/{self.scene_id}"
        paths = [f"{prefix}/rgb/{frame.stem}.png" for frame in self.frames]
        paths.append(f"{prefix}/3D_point_cloud/point_cloud.ply")
        return paths


def read_cameras(path: Path) -> dict[int, tuple[int, int]]:
    """COLMAP cameras.txt to {camera_id: (width, height)}."""
    cameras: dict[int, tuple[int, int]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cameras[int(parts[0])] = (int(parts[2]), int(parts[3]))
    if not cameras:
        raise DatasetError(f"{path} lists no cameras")
    return cameras


def _is_image_record(parts: list[str]) -> bool:
    """An image line ends in a filename; a POINTS2D line is numbers all the way.

    Neither of the obvious rules works. Stepping by two breaks because the test
    poses carry an *empty* POINTS2D line, which vanishes once blank lines are
    skipped and knocks the alternation out of phase. Testing whether field 8
    parses as an integer breaks the other way: POINTS2D is `X Y POINT3D_ID`
    repeated, so field 8 lands on a POINT3D_ID often enough that those lines
    slip through and get counted as images.

    What actually separates them is the last field. On an image record it is a
    filename, and on a POINTS2D record it is a number.
    """
    if len(parts) < 10:
        return False
    try:
        float(parts[-1])
    except ValueError:
        return True
    return False


def read_image_entries(path: Path) -> list[tuple[str, int]]:
    """COLMAP images.txt to [(name, camera_id)]."""
    entries: list[tuple[str, int]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not _is_image_record(parts):
            continue
        entries.append((parts[9], int(parts[8])))
    if not entries:
        raise DatasetError(f"{path} lists no images")
    return entries


def load_scene(dataset: str, scene_root: Path) -> Scene:
    sparse = scene_root / "test" / "sparse" / "0"
    cameras = read_cameras(sparse / "cameras.txt")
    entries = read_image_entries(sparse / "images.txt")

    frames = []
    for name, camera_id in entries:
        if camera_id not in cameras:
            raise DatasetError(
                f"{scene_root.name}: image {name} names camera {camera_id}, "
                f"which cameras.txt does not define")
        width, height = cameras[camera_id]
        frames.append(Frame(Path(name).stem, width, height))

    stems = [frame.stem for frame in frames]
    if len(set(stems)) != len(stems):
        raise DatasetError(f"{scene_root.name}: duplicate frame stems in test poses")

    train_images = scene_root / "train" / "images"
    reference_photos = scene_root / "test" / "images"
    return Scene(
        dataset=dataset,
        scene_id=scene_root.name,
        root=scene_root,
        frames=tuple(frames),
        train_image_count=len(list(train_images.glob("*"))) if train_images.is_dir() else 0,
        has_reference_photos=reference_photos.is_dir() and any(reference_photos.iterdir()),
        has_ground_truth=(scene_root / "3d_gt" / "point_cloud.ply").exists(),
    )


def load_scenes(dataset_root: Path) -> list[Scene]:
    """Every scene a complete submission must cover, in submission order."""
    dataset_root = Path(dataset_root)
    scenes: list[Scene] = []
    for dataset, directory in DATASET_DIRECTORIES.items():
        dataset_dir = dataset_root / directory
        if not dataset_dir.is_dir():
            raise DatasetError(f"{dataset_dir} is missing; is dataset_root correct?")
        found = sorted(dataset_dir.glob("scene_*"))
        if not found:
            raise DatasetError(f"{dataset_dir} contains no scene_* directories")
        scenes.extend(load_scene(dataset, scene_root) for scene_root in found)
    return scenes


def required_paths(scenes: list[Scene]) -> list[str]:
    """Every path the zip must contain for the submission to be scored."""
    return [path for scene in scenes for path in scene.submission_paths()]
