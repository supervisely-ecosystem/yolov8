from typing import List, Sequence, Tuple


def split_keypoints_by_confidence(
    point_labels: Sequence[str],
    coordinates: Sequence[Tuple[float, float]],
    scores: Sequence[float],
    point_threshold: float,
    keep_all_keypoints: bool,
) -> Tuple[List[str], List[Tuple[float, float]], List[bool]]:
    """
    Decide which template keypoints to emit in a prediction, and which of those should
    be marked disabled.

    A point scoring below ``point_threshold`` is dropped entirely, unless
    ``keep_all_keypoints`` is set — in that case every point is kept, and the
    low-confidence ones are flagged disabled instead of being removed from the graph.
    This preserves the same node set as manually annotated ground truth, where a
    not-visible keypoint stays in the graph as a disabled node rather than being
    deleted.

    :return: three equal-length lists (labels, coordinates, disabled) to build a
        ``PredictionKeypoints`` DTO from.
    """
    labels, coords, disabled = [], [], []
    for label, coordinate, score in zip(point_labels, coordinates, scores):
        if score >= point_threshold:
            labels.append(label)
            coords.append(coordinate)
            disabled.append(False)
        elif keep_all_keypoints:
            labels.append(label)
            coords.append(coordinate)
            disabled.append(True)
    return labels, coords, disabled


def count_visible(disabled: Sequence[bool]) -> int:
    """Number of entries in ``disabled`` that are False (i.e. visible/enabled points)."""
    return sum(1 for is_disabled in disabled if not is_disabled)
