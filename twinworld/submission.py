"""Check a submission zip against the rules before it costs a submission slot.

The final phase allows one submission per day and ten in total, and a zip that
is missing any required file is rejected without being scored. So every rule the
challenge states in prose is checked here against the actual zip: the scorer's
verdict should never be the first time we learn the shape is wrong.

Checks are all read from the dataset, not from a hardcoded scene list, so they
stay true if the organisers change what is withheld.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

from twinworld.dataset import Scene, load_scenes, required_paths

VALID_CLASSIFICATION = {0, 1, 2, 3, 4, 255}

# The scorer is killed by clouds above some size and the limit is bracketed, not
# published: 13.00M, 13.84M and 11.39M points in a single cloud have all been
# scored, and 15.11M and about 17.1M have both been killed with a null exit code
# and no log. So warn from the top of the survivable range rather than from the
# bottom of the fatal one. It is not the archive: 1457 MB has scored where 901 MB
# was killed, and 128.7M points across an archive have scored where 76.2M were
# killed. Only what one phase reads from one scene separates the cases.
# See "The final phase" in HANDOVER.md.
LARGEST_SCORED_CLOUD = 13_840_000

# A 3DGS checkpoint is also a .ply with x, y, z, so it passes every structural
# check and then scores badly for a reason no error message would explain: its
# points are Gaussian centres, not surface samples, and the evaluator measures
# distance to a surface. These are the properties that give it away.
GAUSSIAN_PARAMETER_PREFIXES = ("f_dc_", "f_rest_", "scale_", "rot_")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_frames: int = 0
    checked_clouds: int = 0
    points: dict[str, int] = field(default_factory=dict)

    @property
    def largest_cloud(self) -> int:
        return max(self.points.values(), default=0)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"checked {self.checked_frames} frames and {self.checked_clouds} point clouds"]
        if self.points:
            total = sum(self.points.values())
            biggest = max(self.points, key=lambda name: self.points[name])
            lines.append(f"  {total:,} points, largest cloud {self.points[biggest]:,} "
                         f"({biggest})")
        for warning in self.warnings:
            lines.append(f"  warn  {warning}")
        for error in self.errors:
            lines.append(f"  ERROR {error}")
        lines.append("PASS - safe to upload" if self.ok else f"FAIL - {len(self.errors)} error(s)")
        return "\n".join(lines)


def _check_png(data: bytes, name: str, scene: Scene, report: Report) -> None:
    expected = {frame.stem: (frame.width, frame.height) for frame in scene.frames}
    stem = Path(name).stem
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as error:  # noqa: BLE001 - any decode failure is a submission error
        report.errors.append(f"{name}: not a readable image ({error})")
        return
    if image.format != "PNG":
        report.errors.append(f"{name}: is {image.format}, the rules require PNG")
    if image.size != expected[stem]:
        report.errors.append(
            f"{name}: is {image.size[0]}x{image.size[1]}, "
            f"the reference frame is {expected[stem][0]}x{expected[stem][1]}")
    report.checked_frames += 1


def _check_ply(data: bytes, name: str, scene: Scene, report: Report) -> None:
    try:
        ply = PlyData.read(io.BytesIO(data))
    except Exception as error:  # noqa: BLE001
        report.errors.append(f"{name}: not a readable PLY ({error})")
        return
    if "vertex" not in ply:
        report.errors.append(f"{name}: has no vertex element")
        return
    vertex = ply["vertex"]
    properties = {p.name for p in vertex.properties}
    missing = {"x", "y", "z"} - properties
    if missing:
        report.errors.append(f"{name}: missing coordinate propert(ies) {sorted(missing)}")
        return

    gaussian = sorted(p for p in properties if p.startswith(GAUSSIAN_PARAMETER_PREFIXES))
    if gaussian and "opacity" in properties:
        report.errors.append(
            f"{name}: this is a 3DGS checkpoint, not a surface point cloud - it carries "
            f"{len(gaussian) + 1} Gaussian parameters ({', '.join(gaussian[:3])}, ..., opacity). "
            f"Its points are Gaussian centres, which sit off the surface the evaluator "
            f"measures against. Export a separate clean cloud instead of submitting this one")
        return
    if len(vertex) == 0:
        report.errors.append(f"{name}: contains no points")
        return

    coordinates = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    if not np.isfinite(coordinates).all():
        count = int((~np.isfinite(coordinates)).any(axis=1).sum())
        report.errors.append(f"{name}: {count} point(s) have non-finite coordinates")

    if scene.needs_classification:
        if "classification" not in properties:
            report.errors.append(
                f"{name}: {scene.dataset} point clouds require a classification property")
        else:
            labels = vertex["classification"]
            if labels.dtype != np.uint8:
                report.warnings.append(
                    f"{name}: classification is {labels.dtype}, the format states uchar")
            values = set(np.unique(labels).tolist())
            invalid = values - VALID_CLASSIFICATION
            if invalid:
                report.errors.append(
                    f"{name}: classification contains {sorted(invalid)}, "
                    f"allowed values are {sorted(VALID_CLASSIFICATION)}")
            if values == {255}:
                report.warnings.append(f"{name}: every point is ignore(255), so it scores zero")
    elif "classification" in properties:
        report.warnings.append(
            f"{name}: carries a classification the scorer ignores for {scene.dataset}")

    extra = properties - {"x", "y", "z", "classification"}
    if extra:
        report.warnings.append(
            f"{name}: carries {sorted(extra)}, which the scorer ignores and which inflate the upload")
    report.points[f"{scene.dataset}/{scene.scene_id}"] = len(coordinates)
    if len(coordinates) > LARGEST_SCORED_CLOUD:
        report.warnings.append(
            f"{name}: {len(coordinates):,} points, above the {LARGEST_SCORED_CLOUD:,} that "
            f"the scorer is known to survive. Archives whose largest cloud was 15.1M were "
            f"killed with no exit code and no log; thin this one before spending a day on it")
    report.checked_clouds += 1


def _wrapping_directory(present: set[str], by_key: dict[tuple[str, str], Scene]) -> str | None:
    """The single extra top-level folder every entry sits under, if there is one."""
    if not present:
        return None
    tops = {path.split("/")[0] for path in present}
    if len(tops) != 1:
        return None
    wrapper = tops.pop()
    if any(key[0] == wrapper for key in by_key):
        return None  # a legitimate dataset folder, not a wrapper
    stripped = {path.split("/", 1)[1] for path in present if "/" in path}
    datasets = {dataset for dataset, _ in by_key}
    if any(path.split("/")[0] in datasets for path in stripped):
        return wrapper
    return None


def validate(zip_path: Path, dataset_root: Path) -> Report:
    scenes = load_scenes(Path(dataset_root))
    by_key = {(scene.dataset, scene.scene_id): scene for scene in scenes}
    report = Report()

    with zipfile.ZipFile(zip_path) as archive:
        present = {info.filename for info in archive.infolist() if not info.is_dir()}

        # Zipping the parent directory instead of its contents puts everything one
        # level down. That fails every required path at once, so say what actually
        # happened rather than printing a wall of "missing".
        wrapper = _wrapping_directory(present, by_key)
        if wrapper is not None:
            report.errors.append(
                f"everything sits under an extra top-level folder {wrapper!r}; "
                f"the rules require the dataset folders at the top level - "
                f"zip the contents, not the directory")
            return report

        for path in required_paths(scenes):
            if path not in present:
                report.errors.append(f"missing required file: {path}")

        for path in sorted(present):
            parts = path.split("/")
            if len(parts) < 2 or (parts[0], parts[1]) not in by_key:
                report.warnings.append(
                    f"unexpected entry ignored by the scorer: {path}")
                continue
            scene = by_key[(parts[0], parts[1])]
            if path not in set(scene.submission_paths()):
                report.warnings.append(f"unexpected entry ignored by the scorer: {path}")
                continue
            data = archive.read(path)
            if path.endswith(".png"):
                _check_png(data, path, scene, report)
            elif path.endswith(".ply"):
                _check_ply(data, path, scene, report)

    return report
