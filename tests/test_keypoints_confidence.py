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

from src.keypoints_confidence import count_visible, select_visible_indices

SCORES = [0.9, 0.05, 0.95, 0.02, 0.6]


def test_all_above_threshold_are_included_and_enabled():
    indices, disabled = select_visible_indices(
        [0.9, 0.8, 0.95, 0.5, 0.6], point_threshold=0.1, keep_all_keypoints=False
    )
    assert indices == [0, 1, 2, 3, 4]
    assert disabled == [False] * 5


def test_default_behavior_drops_low_confidence_points_unchanged():
    """keep_all_keypoints=False must reproduce the pre-existing behavior exactly:
    sub-threshold points are dropped, not retained as disabled."""
    indices, disabled = select_visible_indices(
        SCORES, point_threshold=0.1, keep_all_keypoints=False
    )
    assert indices == [0, 2, 4]  # only the points scoring >= 0.1
    assert disabled == [False, False, False]


def test_keep_all_keypoints_marks_low_confidence_points_disabled_instead_of_dropping():
    indices, disabled = select_visible_indices(
        SCORES, point_threshold=0.1, keep_all_keypoints=True
    )
    # every index is present, in original order
    assert indices == [0, 1, 2, 3, 4]
    assert disabled == [False, True, False, True, False]


def test_keep_all_keypoints_with_all_points_confident_has_none_disabled():
    indices, disabled = select_visible_indices(
        [0.9, 0.8, 0.95, 0.5, 0.6], point_threshold=0.1, keep_all_keypoints=True
    )
    assert indices == [0, 1, 2, 3, 4]
    assert disabled == [False] * 5


def test_keep_all_keypoints_with_all_points_unconfident_marks_all_disabled():
    scores = [0.0, 0.01, 0.02, 0.0, 0.09]
    indices, disabled = select_visible_indices(
        scores, point_threshold=0.1, keep_all_keypoints=True
    )
    assert indices == [0, 1, 2, 3, 4]
    assert disabled == [True] * 5


def test_without_keep_all_keypoints_all_points_unconfident_yields_nothing():
    scores = [0.0, 0.01, 0.02, 0.0, 0.09]
    indices, disabled = select_visible_indices(
        scores, point_threshold=0.1, keep_all_keypoints=False
    )
    assert indices == []
    assert disabled == []


@pytest.mark.parametrize("keep_all_keypoints", [False, True])
def test_score_exactly_at_threshold_counts_as_visible(keep_all_keypoints):
    # ">= point_threshold" -> a point exactly at the threshold is confident/enabled,
    # regardless of keep_all_keypoints
    indices, disabled = select_visible_indices(
        [0.1], point_threshold=0.1, keep_all_keypoints=keep_all_keypoints
    )
    assert indices == [0]
    assert disabled == [False]


def test_empty_input_returns_empty_lists():
    indices, disabled = select_visible_indices(
        [], point_threshold=0.1, keep_all_keypoints=True
    )
    assert (indices, disabled) == ([], [])


def test_output_lists_are_always_same_length_as_each_other():
    for keep_all_keypoints in (False, True):
        indices, disabled = select_visible_indices(SCORES, 0.1, keep_all_keypoints)
        assert len(indices) == len(disabled)


def test_indices_are_a_subset_in_ascending_original_order():
    for keep_all_keypoints in (False, True):
        indices, _ = select_visible_indices(SCORES, 0.1, keep_all_keypoints)
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)  # no duplicates
        assert all(0 <= i < len(SCORES) for i in indices)  # always in-bounds


def test_count_visible():
    assert count_visible([]) == 0
    assert count_visible([False, False, False]) == 3
    assert count_visible([True, True, True]) == 0
    assert count_visible([False, True, False, True, True]) == 2
