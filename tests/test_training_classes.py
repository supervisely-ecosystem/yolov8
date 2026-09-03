import unittest

from train.src.training_classes import resolve_training_class_names


class _GeometryType:
    def __init__(self, name):
        self._name = name

    def geometry_name(self):
        return self._name


class _ObjClass:
    def __init__(self, name, geometry_name):
        self.name = name
        self.geometry_type = _GeometryType(geometry_name)


class ResolveTrainingClassNamesTest(unittest.TestCase):
    def test_pose_estimation_requires_graph_class(self):
        obj_classes = [
            _ObjClass("person", "polygon"),
            _ObjClass("mask", "bitmap"),
        ]

        with self.assertRaisesRegex(
            ValueError,
            "Project has no classes compatible with 'pose estimation'.*graph class",
        ):
            resolve_training_class_names(obj_classes, "pose estimation")

    def test_pose_estimation_returns_only_graph_classes(self):
        obj_classes = [
            _ObjClass("person", "rectangle"),
            _ObjClass("skeleton", "graph"),
        ]

        class_names = resolve_training_class_names(obj_classes, "pose estimation")

        self.assertEqual(class_names, ["skeleton"])

    def test_instance_segmentation_returns_mask_classes(self):
        obj_classes = [
            _ObjClass("box", "rectangle"),
            _ObjClass("person", "polygon"),
            _ObjClass("mask", "bitmap"),
        ]

        class_names = resolve_training_class_names(
            obj_classes, "instance segmentation"
        )

        self.assertEqual(class_names, ["person", "mask"])

    def test_object_detection_accepts_all_project_classes(self):
        obj_classes = [
            _ObjClass("person", "polygon"),
            _ObjClass("mask", "bitmap"),
        ]

        class_names = resolve_training_class_names(obj_classes, "object detection")

        self.assertEqual(class_names, ["person", "mask"])


if __name__ == "__main__":
    unittest.main()
