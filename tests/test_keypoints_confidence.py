"""
Unit tests for src.keypoints_confidence — the pure post-processing logic behind the
Serve YOLO "keep_all_keypoints" setting.

Deliberately dependency-light (stdlib only): does not import yolov8.py, ultralytics,
torch, cv2 or supervisely, so it can run in any Python environment without setting up
the full serve app stack.

Run with:  python -m pytest tests/test_keypoints_confidence.py -v
(from the repo root, with "serve" on sys.path — see conftest.py)
"""

import pytest

from src.keypoints_confidence import count_visible, split_keypoints_by_confidence

LABELS = ["nose_end", "back_base", "front_right_elbow", "belly_bottom", "breast"]
COORDS = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]


def test_all_above_threshold_are_included_and_enabled():
    labels, coords, disabled = split_keypoints_by_confidence(
        LABELS, COORDS, [0.9, 0.8, 0.95, 0.5, 0.6], point_threshold=0.1, keep_all_keypoints=False
    )
    assert labels == LABELS
    assert coords == COORDS
    assert disabled == [False] * len(LABELS)


def test_default_behavior_drops_low_confidence_points_unchanged():
    """keep_all_keypoints=False must reproduce the pre-existing behavior exactly:
    sub-threshold points are dropped, not retained as disabled."""
    scores = [0.9, 0.05, 0.95, 0.02, 0.6]
    labels, coords, disabled = split_keypoints_by_confidence(
        LABELS, COORDS, scores, point_threshold=0.1, keep_all_keypoints=False
    )
    assert labels == ["nose_end", "front_right_elbow", "breast"]
    assert coords == [(1.0, 1.0), (3.0, 3.0), (5.0, 5.0)]
    assert disabled == [False, False, False]


def test_keep_all_keypoints_marks_low_confidence_points_disabled_instead_of_dropping():
    scores = [0.9, 0.05, 0.95, 0.02, 0.6]
    labels, coords, disabled = split_keypoints_by_confidence(
        LABELS, COORDS, scores, point_threshold=0.1, keep_all_keypoints=True
    )
    # all 24 (here: 5) labels are always present, in the original order
    assert labels == LABELS
    assert coords == COORDS
    assert disabled == [False, True, False, True, False]


def test_keep_all_keypoints_with_all_points_confident_has_none_disabled():
    scores = [0.9, 0.8, 0.95, 0.5, 0.6]
    labels, coords, disabled = split_keypoints_by_confidence(
        LABELS, COORDS, scores, point_threshold=0.1, keep_all_keypoints=True
    )
    assert labels == LABELS
    assert disabled == [False] * len(LABELS)


def test_keep_all_keypoints_with_all_points_unconfident_marks_all_disabled():
    scores = [0.0, 0.01, 0.02, 0.0, 0.09]
    labels, coords, disabled = split_keypoints_by_confidence(
        LABELS, COORDS, scores, point_threshold=0.1, keep_all_keypoints=True
    )
    assert labels == LABELS
    assert coords == COORDS
    assert disabled == [True] * len(LABELS)


def test_without_keep_all_keypoints_all_points_unconfident_yields_nothing():
    scores = [0.0, 0.01, 0.02, 0.0, 0.09]
    labels, coords, disabled = split_keypoints_by_confidence(
        LABELS, COORDS, scores, point_threshold=0.1, keep_all_keypoints=False
    )
    assert labels == []
    assert coords == []
    assert disabled == []


@pytest.mark.parametrize("keep_all_keypoints", [False, True])
def test_score_exactly_at_threshold_counts_as_visible(keep_all_keypoints):
    # ">= point_threshold" -> a point exactly at the threshold is confident/enabled,
    # regardless of keep_all_keypoints
    labels, coords, disabled = split_keypoints_by_confidence(
        ["a"], [(0.0, 0.0)], [0.1], point_threshold=0.1, keep_all_keypoints=keep_all_keypoints
    )
    assert labels == ["a"]
    assert disabled == [False]


def test_empty_input_returns_empty_lists():
    labels, coords, disabled = split_keypoints_by_confidence(
        [], [], [], point_threshold=0.1, keep_all_keypoints=True
    )
    assert (labels, coords, disabled) == ([], [], [])


def test_output_lists_are_always_same_length_as_each_other():
    for keep_all_keypoints in (False, True):
        labels, coords, disabled = split_keypoints_by_confidence(
            LABELS, COORDS, [0.9, 0.05, 0.95, 0.02, 0.6], 0.1, keep_all_keypoints
        )
        assert len(labels) == len(coords) == len(disabled)


def test_count_visible():
    assert count_visible([]) == 0
    assert count_visible([False, False, False]) == 3
    assert count_visible([True, True, True]) == 0
    assert count_visible([False, True, False, True, True]) == 2
