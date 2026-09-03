"""The official metrics, reimplemented so we can measure without spending a slot.

The development phase allows three submissions a day and the final phase ten in
total, which is nowhere near enough to use the leaderboard as a measurement
instrument. Six scenes ship with ground truth - TUM 000 to 003 and Gold Coast
009 and 010 - so those can be scored here as often as we like.

Everything follows the challenge's stated definitions literally:

  geometry   crop both clouds to the ground-truth bounding box plus 20 cm,
             voxel-downsample at 2 cm, then precision/recall F-scores at 5, 10
             and 20 cm. The reported figure is the mean of the three.

  semantics  each ground-truth point takes the label of the nearest submitted
             point within 10 cm, unmatched points count as errors, per-class IoU
             is pooled across scenes and mIoU is the mean over the five classes.

These are our reading of the rules, not the organisers' code, so treat them as a
consistent yardstick for comparing our own attempts rather than as a predictor
of the leaderboard's absolute numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

BOUNDING_BOX_PADDING = 0.20
VOXEL_SIZE = 0.02
FSCORE_THRESHOLDS = (0.05, 0.10, 0.20)
SEMANTIC_MATCH_RADIUS = 0.10
SEMANTIC_CLASS_INDICES = (0, 1, 2, 3, 4)
IGNORE_LABEL = 255


@dataclass
class GeometryScore:
    per_threshold: dict[float, float]
    precision: dict[float, float]
    recall: dict[float, float]
    predicted_points: int
    reference_points: int

    @property
    def fscore(self) -> float:
        return float(np.mean(list(self.per_threshold.values())))


def voxel_downsample(points: np.ndarray, voxel_size: float = VOXEL_SIZE,
                     return_index: bool = False):
    """One representative point per occupied voxel, matching Open3D's behaviour.

    `return_index` hands back which original rows survived, so anything measured
    per point - a colour, a label, a confidence - can be carried through by the
    same indexing rather than recomputed against the thinned cloud.
    """
    if len(points) == 0:
        return (points, np.zeros(0, dtype=np.int64)) if return_index else points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    index = np.sort(index)
    return (points[index], index) if return_index else points[index]


def crop_to_box(points: np.ndarray, reference: np.ndarray,
                padding: float = BOUNDING_BOX_PADDING) -> np.ndarray:
    lower = reference.min(axis=0) - padding
    upper = reference.max(axis=0) + padding
    inside = np.all((points >= lower) & (points <= upper), axis=1)
    return points[inside]


def geometry_fscore(predicted: np.ndarray, reference: np.ndarray) -> GeometryScore:
    """F-score of a predicted cloud against ground truth, as the challenge defines it."""
    predicted = crop_to_box(np.asarray(predicted, dtype=np.float64), np.asarray(reference))
    reference = np.asarray(reference, dtype=np.float64)
    predicted = voxel_downsample(predicted)
    reference = voxel_downsample(reference)

    if len(predicted) == 0 or len(reference) == 0:
        zeros = {t: 0.0 for t in FSCORE_THRESHOLDS}
        return GeometryScore(zeros, dict(zeros), dict(zeros), len(predicted), len(reference))

    # Precision looks from the prediction to the truth, recall the other way.
    to_reference, _ = cKDTree(reference).query(predicted, workers=-1)
    to_predicted, _ = cKDTree(predicted).query(reference, workers=-1)

    per_threshold, precisions, recalls = {}, {}, {}
    for threshold in FSCORE_THRESHOLDS:
        precision = float((to_reference <= threshold).mean())
        recall = float((to_predicted <= threshold).mean())
        denominator = precision + recall
        per_threshold[threshold] = 0.0 if denominator == 0 else 2 * precision * recall / denominator
        precisions[threshold] = precision
        recalls[threshold] = recall

    return GeometryScore(per_threshold, precisions, recalls, len(predicted), len(reference))


def transfer_labels(predicted_points: np.ndarray, predicted_labels: np.ndarray,
                    reference_points: np.ndarray,
                    radius: float = SEMANTIC_MATCH_RADIUS) -> np.ndarray:
    """Each reference point takes its nearest predicted label, or 255 past the radius.

    Bounded at the radius, which is not an approximation: anything past it is
    discarded on the next line anyway, and the bound lets the tree abandon each
    hopeless reference point instead of finding its true nearest neighbour first.
    That matters because this is the slowest step in the repository - a Gold
    Coast scene asks it for 30M reference points against a 14M-point tree - and
    it runs on every scoring path.

    A reference point with nothing inside the radius comes back with an index of
    len(predicted_points), which is why the labels are padded rather than
    indexed directly: the out-of-range index is scipy's way of saying "no
    neighbour", and it would otherwise raise.
    """
    predicted_points = np.asarray(predicted_points)
    if len(predicted_points) == 0:
        return np.full(len(reference_points), IGNORE_LABEL, dtype=np.uint8)

    _, index = cKDTree(predicted_points).query(
        np.asarray(reference_points), distance_upper_bound=radius, workers=-1)
    padded = np.append(np.asarray(predicted_labels).astype(np.uint8), IGNORE_LABEL)
    return padded[index]


def coverage(predicted_points: np.ndarray, reference_points: np.ndarray,
             reference_labels: np.ndarray,
             radius: float = SEMANTIC_MATCH_RADIUS) -> np.ndarray:
    """Per class, how many ground-truth points have any prediction near them.

    Returned as a 5 by 2 tally of covered and total, so scenes can be pooled by
    adding them, the way `pooled_miou` pools confusions.

    This is the semantic score's ceiling and it is worth computing every time,
    because it splits the two failures the mIoU alone cannot tell apart. A
    ground-truth point with a prediction inside the match radius would be correct
    under perfect labelling; one without is unmatched however good the labels
    are, and contributes a false negative to its own class and nothing anywhere
    else. So per-class coverage is exactly the IoU a perfect classifier reaches,
    what is left between it and the real score belongs to the classifier, and
    this repository has already once concluded "coverage bound" from a number
    that was really measuring a registration error.
    """
    size = len(SEMANTIC_CLASS_INDICES)
    tally = np.zeros((size, 2), dtype=np.int64)
    if len(predicted_points) == 0:
        near = np.zeros(len(reference_points), dtype=bool)
    else:
        distance, _ = cKDTree(np.asarray(predicted_points)).query(
            np.asarray(reference_points), distance_upper_bound=radius, workers=-1)
        near = np.isfinite(distance)

    labels = np.asarray(reference_labels)
    for index in SEMANTIC_CLASS_INDICES:
        of_class = labels == index
        tally[index] = (int(near[of_class].sum()), int(of_class.sum()))
    return tally


def pooled_ceiling(tallies: list[np.ndarray]) -> tuple[float, dict[int, float]]:
    """The mIoU a perfect labelling of the submitted cloud would reach."""
    total = np.sum(tallies, axis=0)
    per_class = {index: (float(total[index, 0] / total[index, 1])
                         if total[index, 1] else float("nan"))
                 for index in SEMANTIC_CLASS_INDICES}
    present = [value for value in per_class.values() if not np.isnan(value)]
    return (float(np.mean(present)) if present else 0.0), per_class


def confusion(reference_labels: np.ndarray, predicted_labels: np.ndarray) -> np.ndarray:
    """A 5 by 6 tally: ground-truth class by predicted class, plus an unmatched column.

    The extra column is what makes "unmatched points count as errors" come out
    right. An unmatched ground-truth point has to reduce its own class's recall
    without inflating any other class's false positives, and a plain 5 by 5
    matrix has nowhere to put it - folding it onto a real column would blame a
    class the prediction never named.

    Ground-truth points labelled 255 are excluded entirely; they are the
    challenge's own ignore label.
    """
    size = len(SEMANTIC_CLASS_INDICES)
    matrix = np.zeros((size, size + 1), dtype=np.int64)

    scored = np.isin(reference_labels, SEMANTIC_CLASS_INDICES)
    reference = np.asarray(reference_labels)[scored].astype(np.int64)
    predicted = np.asarray(predicted_labels)[scored].astype(np.int64)

    column = np.where(np.isin(predicted, SEMANTIC_CLASS_INDICES), predicted, size)
    np.add.at(matrix, (reference, column), 1)
    return matrix


def pooled_miou(confusions: list[np.ndarray]) -> tuple[float, dict[int, float]]:
    """mIoU over the five classes, pooled across scenes rather than averaged per scene.

    Pooling matters: a class that appears in one scene and not another would be
    weighted quite differently the other way round.
    """
    total = np.sum(confusions, axis=0)
    per_class: dict[int, float] = {}
    for index in SEMANTIC_CLASS_INDICES:
        true_positive = total[index, index]
        false_negative = total[index, :].sum() - true_positive
        # Only the five real columns can be a false positive; the unmatched
        # column belongs to whichever row it came from and is already counted.
        false_positive = total[:, index].sum() - true_positive
        union = true_positive + false_negative + false_positive
        per_class[index] = float(true_positive / union) if union else float("nan")

    present = [value for value in per_class.values() if not np.isnan(value)]
    return (float(np.mean(present)) if present else 0.0), per_class
