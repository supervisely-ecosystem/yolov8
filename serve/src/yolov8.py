import os
from typing import Any, Dict, Generator, List, Union, Literal
from threading import Event

import numpy as np
import cv2
import torch

from ultralytics import YOLO

import supervisely as sly
from supervisely.nn.inference import TaskType, CheckpointInfo, RuntimeType
from supervisely.app.widgets import (
    PretrainedModelsSelector,
    RadioTabs,
    CustomModelsSelector,
)
from supervisely.nn.prediction_dto import (
    PredictionBBox,
    PredictionKeypoints,
    PredictionMask,
)

from supervisely.nn.artifacts.yolov8 import YOLOv8
from src.keypoints_template import dict_to_template, human_template
from src.models import yolov8_models
import src.workflow as w


class YOLOv8Model(sly.nn.inference.ObjectDetection):
    team_id = sly.env.team_id()
    in_train = False

    def initialize_custom_gui(self):
        """Create custom GUI layout for model selection. This method is called once when the application is started."""
        self.pretrained_models_table = PretrainedModelsSelector(yolov8_models)
        yolov8_artifacts = YOLOv8(self.team_id)
        custom_checkpoints = yolov8_artifacts.get_list()
        self.custom_models_table = CustomModelsSelector(
            self.team_id,
            custom_checkpoints,
            show_custom_checkpoint_path=True,
            custom_checkpoint_task_types=[
                "object detection",
                "instance segmentation",
                "pose estimation",
            ],
        )

        self.model_source_tabs = RadioTabs(
            titles=["Pretrained models", "Custom models"],
            descriptions=[
                "Publicly available models",
                "Models trained by you in Supervisely",
            ],
            contents=[self.pretrained_models_table, self.custom_models_table],
        )
        return self.model_source_tabs

    def get_params_from_gui(self) -> dict:
        model_source = self.model_source_tabs.get_active_tab()
        self.device = self.gui.get_device()
        if model_source == "Pretrained models":
            model_params = self.pretrained_models_table.get_selected_model_params()
        elif model_source == "Custom models":
            model_params = self.custom_models_table.get_selected_model_params()

        self.task_type = model_params.get("task_type")
        deploy_params = {
            "device": self.device,
            **model_params,
        }

        # -------------------------------------- Add Workflow Input -------------------------------------- #
        if not self.in_train:
            w.workflow_input(self.api, model_params)
        # ----------------------------------------------- - ---------------------------------------------- #
        
        return deploy_params

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
            obj_classes = [
                sly.ObjClass(name, sly.Rectangle) for name in self.class_names
            ]
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
        task_type: Literal[
            "object detection", "instance segmentation", "pose estimation"
        ],
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
        self.device = device
        self.runtime = RuntimeType.PYTORCH
        self.task_type = task_type

        local_weights_path = os.path.join(self.model_dir, checkpoint_name)
        if not sly.fs.file_exists(local_weights_path):
            self.download(
                src_path=checkpoint_url,
                dst_path=local_weights_path,
            )

        self.model = YOLO(local_weights_path)
        self.model.to(self.device)
        self.load_model_meta(model_source, local_weights_path)

        # Set checkpoint info
        train_args = self.model.ckpt["train_args"]
        if model_source == "Pretrained models":
            custom_checkpoint_path = None
            checkpoint_name = os.path.basename(train_args["model"]).split(".")[0]
        else:
            custom_checkpoint_path = checkpoint_url
            file_id = self.api.file.get_info_by_path(self.team_id, checkpoint_url).id
            checkpoint_url = self.api.file.get_url(file_id)
        # model_name, architecture = parse_model_name(checkpoint_name)
        self.checkpoint_info = CheckpointInfo(
            checkpoint_name=checkpoint_name,  # TODO: checkpoint_name is not correct for custom models
            architecture="YOLO",
            model_source=model_source,
            checkpoint_url=checkpoint_url,
            custom_checkpoint_path=custom_checkpoint_path,
        )

        # This will disable logs from YOLO
        # import logging
        # if sly.logger.isEnabledFor(logging.DEBUG):
        #     self.model.overrides['verbose'] = True
        # else:
        #     self.model.overrides['verbose'] = False

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

    def _create_label(
        self, dto: Union[PredictionMask, PredictionBBox, PredictionKeypoints]
    ):
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
                    sly.logger.debug(
                        f"Mask of class {dto.class_name} is empty and will be skipped"
                    )
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
                dtos.append(
                    PredictionBBox(self.class_names[cls_index], bbox, confidence)
                )
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
                    dtos.append(
                        PredictionMask(self.class_names[cls_index], mask, confidence)
                    )
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

    def predict_video(
        self, video_path: str, settings: Dict[str, Any], stop: Event
    ) -> Generator:
        retina_masks = self.task_type == "instance segmentation"
        predictions_generator = self.model(
            source=video_path,
            stream=True,
            conf=settings["conf"],
            iou=settings["iou"],
            half=settings["half"],
            device=self.device,
            max_det=settings["max_det"],
            agnostic_nms=settings["agnostic_nms"],
            retina_masks=retina_masks,
        )
        for prediction in predictions_generator:
            if stop.is_set():
                predictions_generator.close()
                return
            yield self._to_dto(prediction, settings)
    
    def predict_benchmark(self, images_np: List[np.ndarray], settings: Dict):
        # RGB to BGR
        images_np = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np]
        retina_masks = self.task_type == "instance segmentation"
        predictions = self.model(
            source=images_np,
            conf=settings["conf"],
            iou=settings["iou"],
            half=settings["half"],
            device=self.device,
            max_det=settings["max_det"],
            agnostic_nms=settings["agnostic_nms"],
            retina_masks=retina_masks,
        )
        n = len(predictions)
        first_benchmark = predictions[0].speed
        # YOLOv8 returns avg time per image, so we need to multiply it by the number of images
        benchmark = {
            "preprocess": first_benchmark["preprocess"] * n,
            "inference": first_benchmark["inference"] * n,
            "postprocess": first_benchmark["postprocess"] * n,
        }
        predictions = [self._to_dto(prediction, settings) for prediction in predictions]
        return predictions, benchmark


def parse_model_name(model_name: str):
    v8 = int(model_name[5])
    postfix = model_name[6:].split("-")
    variant = postfix[0].upper().strip()
    if len(postfix) > 1:
        task = postfix[1]
        name = f"YOLOv{v8}-{variant}"
    else:
        name = f"YOLOv{v8}-{variant}"
    architecture = f"YOLOv{v8}"
    return name, architecture

