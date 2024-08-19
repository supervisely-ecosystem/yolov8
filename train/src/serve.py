import os
from typing import Any, Dict, List, Literal, Union

import cv2
import supervisely as sly
import torch
from src.keypoints_template import dict_to_template, human_template
from supervisely.nn.inference import CheckpointInfo
from supervisely.nn.prediction_dto import (
    PredictionBBox,
    PredictionKeypoints,
    PredictionMask,
)
from ultralytics import YOLO


class YOLOv8ModelBM(sly.nn.inference.ObjectDetection):

    def load_model_meta(self, model_source: str, weights_save_path: str):
        self.class_names = list(self.model.names.values())
        if self.task_type == "pose estimation":
            if model_source == "Pretrained models":
                self.keypoints_template = human_template
            elif model_source == "Custom models":
                weights_dict = torch.load(weights_save_path)
                geometry_data = weights_dict["geometry_config"]
                if "nodes_order" not in geometry_data:
                    geometry_config = geometry_data
                    self.keypoints_template = dict_to_template(geometry_config)
                else:
                    self.nodes_order = geometry_data["nodes_order"]
                    self.cls2config = {}
                    for key, value in geometry_data["configs"].items():
                        self.cls2config[key] = value
        if self.task_type == "object detection":
            obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in self.class_names]
        elif self.task_type == "instance segmentation":
            obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in self.class_names]
        elif self.task_type == "pose estimation":
            if len(self.class_names) == 1:
                obj_classes = [
                    sly.ObjClass(name, sly.GraphNodes, geometry_config=self.keypoints_template)
                    for name in self.class_names
                ]
            elif len(self.class_names) > 1:
                obj_classes = [
                    sly.ObjClass(
                        name,
                        sly.GraphNodes,
                        geometry_config=dict_to_template(self.cls2config[name]),
                    )
                    for name in self.class_names
                ]
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        self._get_confidence_tag_meta()

    def load_model(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        model_source: Literal["Pretrained models", "Custom models"],
        task_type: Literal["object detection", "instance segmentation", "pose estimation"],
        checkpoint_name: str,
        checkpoint_url: str,
    ):
        """
        Load model method is used to deploy model.

        :param model_source: Specifies whether the model is pretrained or custom.
        :type model_source: Literal["Pretrained models", "Custom models"]
        :param device: The device on which the model will be deployed.
        :type device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        :param task_type: The type of computer vision task the model is designed for.
        :type task_type: Literal["object detection", "instance segmentation", "pose estimation"]
        :param checkpoint_name: The name of the checkpoint from which the model is loaded.
        :type checkpoint_name: str
        :param checkpoint_url: The URL where the model can be downloaded.
        :type checkpoint_url: str
        """
        self.task_type = task_type
        local_weights_path = os.path.join(self.model_dir, checkpoint_name)
        if not sly.fs.file_exists(local_weights_path):
            self.download(
                src_path=checkpoint_url,
                dst_path=local_weights_path,
            )

        self.model = YOLO(local_weights_path)
        if device.startswith("cuda"):
            if device == "cuda":
                self.device = 0
            else:
                self.device = int(device[-1])
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.load_model_meta(model_source, local_weights_path)
        self.checkpoint_info = CheckpointInfo(checkpoint_name, "yolov8", model_source)  # TODO name

    def get_info(self):
        info = super().get_info()
        info["task type"] = self.task_type
        info["videos_support"] = True
        info["async_video_inference_support"] = True
        info["tracking_on_videos_support"] = True
        if self.task_type == "pose estimation":
            info["detector_included"] = True
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    def _create_label(self, dto: Union[PredictionMask, PredictionBBox, PredictionKeypoints]):
        if self.task_type == "object detection":
            obj_class = self.model_meta.get_obj_class(dto.class_name)
            if obj_class is None:
                raise KeyError(
                    f"Class {dto.class_name} not found in model classes {self.get_classes()}"
                )
            geometry = sly.Rectangle(*dto.bbox_tlbr)
            tags = []
            if dto.score is not None:
                tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
            label = sly.Label(geometry, obj_class, tags)
        elif self.task_type == "instance segmentation":
            obj_class = self.model_meta.get_obj_class(dto.class_name)
            if obj_class is None:
                raise KeyError(
                    f"Class {dto.class_name} not found in model classes {self.get_classes()}"
                )
            if isinstance(dto, PredictionMask):
                if not dto.mask.any():  # skip empty masks
                    sly.logger.debug(f"Mask of class {dto.class_name} is empty and will be skipped")
                    return None
                geometry = sly.Bitmap(dto.mask, extra_validation=False)
            elif isinstance(dto, PredictionBBox):
                geometry = sly.Rectangle(*dto.bbox_tlbr)
            tags = []
            if dto.score is not None:
                tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
            label = sly.Label(geometry, obj_class, tags)
        elif self.task_type == "pose estimation":
            obj_class = self.model_meta.get_obj_class(dto.class_name)
            if obj_class is None:
                raise KeyError(
                    f"Class {dto.class_name} not found in model classes {self.get_classes()}"
                )
            nodes = []
            for label, coordinate in zip(dto.labels, dto.coordinates):
                x, y = coordinate
                nodes.append(sly.Node(label=label, row=y, col=x))
            label = sly.Label(sly.GraphNodes(nodes), obj_class)
        return label

    def node_id_to_point(self, keypoints):
        nid2point = {}
        for node, keypoint in zip(self.nodes_order, keypoints):
            nid2point[node] = keypoint
        return nid2point

    def _to_dto(
        self, prediction, settings: dict
    ) -> List[Union[PredictionMask, PredictionBBox, PredictionKeypoints]]:
        """Converts YOLO Results to a List of Prediction DTOs."""
        dtos = []
        if self.task_type == "object detection":
            boxes_data = prediction.boxes.data
            for box in boxes_data:
                left, top, right, bottom, confidence, cls_index = (
                    int(box[0]),
                    int(box[1]),
                    int(box[2]),
                    int(box[3]),
                    float(box[4]),
                    int(box[5]),
                )
                bbox = [top, left, bottom, right]
                dtos.append(PredictionBBox(self.class_names[cls_index], bbox, confidence))
        elif self.task_type == "instance segmentation":
            boxes_data = prediction.boxes.data
            if prediction.masks:
                masks = prediction.masks.data
                for box, mask in zip(boxes_data, masks):
                    left, top, right, bottom, confidence, cls_index = (
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3]),
                        float(box[4]),
                        int(box[5]),
                    )
                    mask = mask.cpu().numpy()
                    dtos.append(PredictionMask(self.class_names[cls_index], mask, confidence))
        elif self.task_type == "pose estimation":
            boxes_data = prediction.boxes.data
            keypoints_data = prediction.keypoints.data
            point_threshold = settings.get("point_threshold", 0.1)
            if len(self.class_names) == 1:
                point_labels = self.keypoints_template.point_names
                if len(point_labels) == 17:
                    point_labels.append("fictive")
                for box, keypoints in zip(boxes_data, keypoints_data):
                    left, top, right, bottom, confidence, cls_index = (
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3]),
                        float(box[4]),
                        int(box[5]),
                    )
                    included_labels, included_point_coordinates = [], []
                    if keypoints_data.shape[-1] == 3:
                        point_coordinates, point_scores = (
                            keypoints[:, :2],
                            keypoints[:, 2],
                        )
                        for j, (point_coordinate, point_score) in enumerate(
                            zip(point_coordinates, point_scores)
                        ):
                            if point_score >= point_threshold and point_labels[j] != "fictive":
                                included_labels.append(point_labels[j])
                                included_point_coordinates.append(point_coordinate.cpu().numpy())
                    elif keypoints_data.shape[-1] == 2:
                        for j, point_coordinate in enumerate(keypoints):
                            included_labels.append(point_labels[j])
                            included_point_coordinates.append(point_coordinate.cpu().numpy())
                    if len(included_labels) > 1:
                        dtos.append(
                            sly.nn.PredictionKeypoints(
                                self.class_names[cls_index],
                                included_labels,
                                included_point_coordinates,
                            )
                        )
            elif len(self.class_names) > 1:
                for box, keypoints in zip(boxes_data, keypoints_data):
                    left, top, right, bottom, confidence, cls_index = (
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3]),
                        float(box[4]),
                        int(box[5]),
                    )
                    keypoints_template = self.cls2config[self.class_names[cls_index]]
                    point_labels = [
                        value["label"] for value in keypoints_template["nodes"].values()
                    ]
                    node_ids = list(keypoints_template["nodes"].keys())
                    nid2point = self.node_id_to_point(keypoints)
                    included_labels, included_point_coordinates = [], []
                    for j, node_id in enumerate(node_ids):
                        kpt = nid2point[node_id]
                        point_coordinate, point_score = kpt[:2], kpt[2]
                        if point_score >= point_threshold:
                            included_labels.append(point_labels[j])
                            included_point_coordinates.append(point_coordinate.cpu().numpy())
                    if len(included_labels) > 1:
                        dtos.append(
                            sly.nn.PredictionKeypoints(
                                self.class_names[cls_index],
                                included_labels,
                                included_point_coordinates,
                            )
                        )
        return dtos

    def predict_batch(
        self, images_np: List, settings: Dict[str, Any]
    ) -> List[List[Union[PredictionMask, PredictionBBox, PredictionKeypoints]]]:
        retina_masks = self.task_type == "instance segmentation"
        predictions = self.model(
            source=[cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np],
            conf=settings["conf"],
            iou=settings["iou"],
            half=settings["half"],
            device=self.device,
            max_det=settings["max_det"],
            agnostic_nms=settings["agnostic_nms"],
            retina_masks=retina_masks,
        )
        return [self._to_dto(prediction, settings) for prediction in predictions]

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[Union[PredictionMask, PredictionBBox, PredictionKeypoints]]:
        retina_masks = self.task_type == "instance segmentation"
        predictions = self.model(
            source=image_path,
            conf=settings["conf"],
            iou=settings["iou"],
            half=settings["half"],
            device=self.device,
            max_det=settings["max_det"],
            agnostic_nms=settings["agnostic_nms"],
            retina_masks=retina_masks,
        )
        return self._to_dto(predictions[0], settings)
