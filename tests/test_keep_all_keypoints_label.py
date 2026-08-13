"""
SDK-level tests for the "keep_all_keypoints" feature: verifies that a
PredictionKeypoints DTO carrying a `.disabled` list is turned into a GraphNodes Label
with the correct per-node `disabled` flag, that this validates against a project meta
template requiring all nodes, and that the disabled flag round-trips through
to_json/from_json exactly like a manually-hidden (Ctrl) keypoint would.

Only needs the `supervisely` package installed — no ultralytics/torch/cv2, no running
model, no GUI. This intentionally mirrors (rather than imports) the small
PredictionKeypoints -> GraphNodes conversion in YOLOv8Model._create_label
(serve/src/yolov8.py), since importing that module pulls in the full ultralytics/torch
stack that this feature's core logic does not actually depend on.

Run with:  python -m pytest tests/test_keep_all_keypoints_label.py -v
"""

from typing import List, Optional

import pytest
import supervisely as sly
from supervisely.nn.prediction_dto import PredictionKeypoints

# The exact 24 pig-anatomy keypoint labels from the reported customer template
# (see investigation.md), used here as a realistic, non-trivial node set.
PIG_TEMPLATE_LABELS = [
    "nose_end", "belly_bottom", "right_ear_base", "left_ear_base", "back_hanging_paw",
    "right_ear_end", "front_right_thigh", "front_left_paw", "front_right_paw",
    "left_ear_end", "front_right_elbow", "back_end", "front_left_elbow",
    "back_hanging_knee", "back_on_track_thigh", "back_base", "back_hanging_thigh",
    "back_middle", "tail_end", "tail_base", "front_left_thigh", "back_on_track_knee",
    "back_on_track_paw", "breast",
]
assert len(PIG_TEMPLATE_LABELS) == 24


def dto_to_nodes(dto: PredictionKeypoints) -> List[sly.Node]:
    """Mirrors the PredictionKeypoints branch of YOLOv8Model._create_label
    (serve/src/yolov8.py) exactly, without importing that module. Bounds-safe on
    purpose: a `.disabled` list shorter than `.labels`/`.coordinates` must not raise,
    it should just default the missing entries to enabled."""
    disabled_flags: Optional[List[bool]] = getattr(dto, "disabled", None) or []
    nodes = []
    for i, (node_label, coordinate) in enumerate(zip(dto.labels, dto.coordinates)):
        x, y = coordinate
        is_disabled = bool(disabled_flags[i]) if i < len(disabled_flags) else False
        nodes.append(sly.Node(label=node_label, row=y, col=x, disabled=is_disabled))
    return nodes


def make_full_pig_meta():
    template = sly.geometry.graph.KeypointsTemplate()
    for i, label in enumerate(PIG_TEMPLATE_LABELS):
        template.add_point(label=label, row=i, col=i)
    obj_class = sly.ObjClass(
        "new pig on track keypoints", sly.GraphNodes, geometry_config=template
    )
    return sly.ProjectMeta(obj_classes=[obj_class]), obj_class


def test_legacy_dto_without_disabled_attr_builds_all_visible_nodes():
    """A DTO with no `.disabled` attribute (i.e. every other DTO producer in the
    codebase, and this DTO before this feature existed) must behave exactly as before:
    all constructed nodes are enabled."""
    dto = PredictionKeypoints(
        "new pig on track keypoints",
        PIG_TEMPLATE_LABELS[:3],
        [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
    )
    nodes = dto_to_nodes(dto)
    assert len(nodes) == 3
    assert all(not n.disabled for n in nodes)


def test_dto_with_disabled_list_produces_matching_disabled_nodes():
    labels = PIG_TEMPLATE_LABELS[:4]
    coords = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    disabled = [False, True, False, True]
    dto = PredictionKeypoints("new pig on track keypoints", labels, coords)
    dto.disabled = disabled
    nodes = dto_to_nodes(dto)
    assert [n.disabled for n in nodes] == disabled
    # disabled nodes still carry a real location, same as a manually Ctrl-hidden point
    assert [(n.location.row, n.location.col) for n in nodes] == [
        (y, x) for x, y in coords
    ]


def test_all_24_nodes_present_with_mixed_disabled_validates_against_full_template():
    """This is the customer's exact ask: keep_all_keypoints=True should produce a
    label with all 24 template nodes (some disabled), and it must validate cleanly
    against a project meta that declares all 24 -- no
    "Graph contains nodes not declared in the template" error."""
    meta, obj_class = make_full_pig_meta()
    coords = [(float(i), float(i)) for i in range(24)]
    # simulate: first 10 keypoints confidently detected, remaining 14 below threshold
    disabled = [False] * 10 + [True] * 14
    dto = PredictionKeypoints("new pig on track keypoints", PIG_TEMPLATE_LABELS, coords)
    dto.disabled = disabled
    nodes = dto_to_nodes(dto)
    assert len(nodes) == 24

    # Label.__init__ validates the geometry against obj_class.geometry_config
    # internally -- it must NOT raise "Graph contains nodes not declared in
    # the template" here, unlike the original customer-reported bug.
    label = sly.Label(sly.GraphNodes(nodes), obj_class)

    geometry_json = label.geometry.to_json()
    assert set(geometry_json["nodes"].keys()) == set(PIG_TEMPLATE_LABELS)
    disabled_in_json = {
        lbl for lbl, node in geometry_json["nodes"].items() if node.get("disabled")
    }
    assert disabled_in_json == set(PIG_TEMPLATE_LABELS[10:])


def test_default_partial_label_also_validates_against_full_template():
    """Regression guard: today's default behavior (drop sub-threshold points, so
    fewer than 24 nodes are present) must keep validating fine too -- a label using a
    subset of a template's nodes is always valid; validate() only rejects nodes that
    are NOT declared in the template."""
    meta, obj_class = make_full_pig_meta()
    labels = PIG_TEMPLATE_LABELS[:5]
    coords = [(float(i), float(i)) for i in range(5)]
    dto = PredictionKeypoints("new pig on track keypoints", labels, coords)
    nodes = dto_to_nodes(dto)
    assert len(nodes) == 5

    label = sly.Label(sly.GraphNodes(nodes), obj_class)  # must not raise

    geometry_json = label.geometry.to_json()
    assert set(geometry_json["nodes"].keys()) == set(labels)
    assert not any(node.get("disabled") for node in geometry_json["nodes"].values())


def test_disabled_flag_round_trips_through_json_like_a_manual_annotation():
    """Node.to_json only writes "disabled" when True (see supervisely/geometry/graph.py);
    a manually Ctrl-hidden keypoint is stored the same way. Confirms our nodes are
    indistinguishable from a manually annotated disabled node after round-tripping."""
    node_visible = sly.Node(label="breast", row=1, col=1, disabled=False)
    node_hidden = sly.Node(label="tail_end", row=2, col=2, disabled=True)

    visible_json = node_visible.to_json()
    hidden_json = node_hidden.to_json()
    assert "disabled" not in visible_json
    assert hidden_json["disabled"] is True

    restored_visible = sly.geometry.graph.Node.from_json(visible_json)
    restored_hidden = sly.geometry.graph.Node.from_json(hidden_json)
    assert restored_visible.disabled is False
    assert restored_hidden.disabled is True


def test_disabled_list_shorter_than_labels_does_not_crash():
    """Defensive guard: if `.disabled` is ever malformed (too short) relative to
    `.labels`/`.coordinates`, building nodes must not raise -- missing entries
    default to enabled rather than throwing IndexError."""
    labels = PIG_TEMPLATE_LABELS[:5]
    coords = [(float(i), float(i)) for i in range(5)]
    dto = PredictionKeypoints("new pig on track keypoints", labels, coords)
    dto.disabled = [True]  # only 1 entry for 5 labels
    nodes = dto_to_nodes(dto)  # must not raise
    assert len(nodes) == 5
    assert [n.disabled for n in nodes] == [True, False, False, False, False]


def test_disabled_explicitly_none_or_empty_behaves_like_missing():
    labels = PIG_TEMPLATE_LABELS[:2]
    coords = [(1.0, 1.0), (2.0, 2.0)]
    for disabled_value in (None, []):
        dto = PredictionKeypoints("new pig on track keypoints", labels, coords)
        dto.disabled = disabled_value
        nodes = dto_to_nodes(dto)  # must not raise
        assert [n.disabled for n in nodes] == [False, False]


def test_disabled_count_below_two_visible_would_have_skipped_instance_pre_feature():
    """Sanity-check for the count_visible() gate used in yolov8.py: an instance with
    zero or one confident point should not have produced a keypoints annotation
    before this feature, and with keep_all_keypoints=True it must still be gated the
    same way (based on *visible* count, not total node count)."""
    from src.keypoints_confidence import count_visible

    all_disabled_but_one = [True] * 23 + [False]
    assert count_visible(all_disabled_but_one) == 1  # still gated out (needs > 1)

    two_visible = [True] * 22 + [False, False]
    assert count_visible(two_visible) == 2  # gated in
