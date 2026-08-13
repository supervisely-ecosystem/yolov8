from typing import List, Sequence, Tuple


def select_visible_indices(
    scores: Sequence[float],
    point_threshold: float,
    keep_all_keypoints: bool,
) -> Tuple[List[int], List[bool]]:
    """
    Decide which keypoints to emit in a prediction, and which of those should be
    marked disabled -- without touching coordinates.

    A point scoring below ``point_threshold`` is dropped entirely, unless
    ``keep_all_keypoints`` is set -- in that case every point is kept, and the
    low-confidence ones are flagged disabled instead of being removed from the graph.
    This preserves the same node set as manually annotated ground truth, where a
    not-visible keypoint stays in the graph as a disabled node rather than being
    deleted.

    Deliberately only returns indices/flags rather than filtering coordinates
    directly: coordinate extraction can involve a GPU->CPU sync (``.cpu().numpy()``)
    per point, so callers should only pay that cost for points that are actually
    kept, exactly as before this option existed.

    :param scores: confidence score per keypoint, in template order.
    :param point_threshold: minimum score for a point to count as visible.
    :param keep_all_keypoints: if True, keep every point (flagging low-confidence
        ones as disabled) instead of dropping them.
    :return: ``(indices, disabled)`` -- same length, in ascending/original order.
        ``indices`` are positions into ``scores`` (and any other same-length,
        same-order per-point sequence, e.g. labels/coordinates) that should be kept.
    """
    indices, disabled = [], []
    for i, score in enumerate(scores):
        if score >= point_threshold:
            indices.append(i)
            disabled.append(False)
        elif keep_all_keypoints:
            indices.append(i)
            disabled.append(True)
    return indices, disabled


def count_visible(disabled: Sequence[bool]) -> int:
    """Number of entries in ``disabled`` that are False (i.e. visible/enabled points)."""
    return sum(1 for is_disabled in disabled if not is_disabled)
