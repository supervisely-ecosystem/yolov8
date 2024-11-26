from src.monkey_patching_fix import monkey_patching_fix

# Monkey patching to avoid a problem with installing wrong onnxruntime requirements
# issue: https://github.com/ultralytics/ultralytics/issues/5093
monkey_patching_fix()

import os
from typing import Any, Dict, Generator, List, Union, Literal
from threading import Event
import re

import numpy as np
import cv2
import torch

from ultralytics import YOLO
import yaml

import supervisely as sly
from supervisely.nn.inference import (
    TaskType,
    CheckpointInfo,
    RuntimeType,
    ModelSource,
    ModelPrecision,
)
from supervisely.app.widgets import (
    PretrainedModelsSelector,
    RadioTabs,
    CustomModelsSelector,
    SelectString,
    Field,
    Container,
    Checkbox,
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
    TENSORRT_MAX_BATCH_SIZE = 1

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
        self.runtime_select = SelectString(
            [
                RuntimeType.PYTORCH,
                RuntimeType.ONNXRUNTIME,
                RuntimeType.TENSORRT,
            ]
        )
        runtime_field = Field(
            self.runtime_select,
            "Runtime",
            "The model will be exported to the selected runtime for efficient inference (exporting to TensorRT may take about a minute).",
        )
        self.fp16_checkbox = Checkbox("Enable FP16", False)
        fp16_field = Field(
            self.fp16_checkbox,
            "FP16 precision",
            "Export model with FP16 precision. This will reduce GPU memory usage and increase inference speed. "
            "Usually, the accuracy of predictions remains the same.",
        )
        fp16_field.hide()
        layout = Container([self.model_source_tabs, runtime_field, fp16_field])

        @self.runtime_select.value_changed
        def on_runtime_changed(value):
            if value == RuntimeType.TENSORRT:
                fp16_field.show()
            else:
                fp16_field.hide()

        return layout

    def get_params_from_gui(self) -> dict:
        model_source = self.model_source_tabs.get_active_tab()
        device = self.gui.get_device()
        if model_source == "Pretrained models":
            model_params = self.pretrained_models_table.get_selected_model_params()
        elif model_source == "Custom models":
            model_params = self.custom_models_table.get_selected_model_params()
        runtime = self.runtime_select.get_value()
        fp16 = self.fp16_checkbox.is_checked()
        deploy_params = {
            "device": device,
            "runtime": runtime,
            "fp16": fp16,
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
            self.general_class_names = list(self.model.names.values())
            bbox_class_names = [f"{cls}_bbox" for cls in self.class_names]
            kpt_class_names = self.class_names
            self.class_names = bbox_class_names + kpt_class_names
            if model_source == "Pretrained models":
                self.keypoints_template = human_template
            elif model_source == "Custom models":
                model = YOLO(weights_save_path)
                weights_dict = model.ckpt
                geometry_data = weights_dict["geometry_config"]
                if "nodes_order" not in geometry_data:
                    geometry_config = geometry_data
                    self.keypoints_template = dict_to_template(geometry_config)
                    self.nodes_order = []
                    for key, value in geometry_config["nodes"].items():
                        label = value["label"]
                        self.nodes_order.append(label)
                    self.cls2config = {}
                    for cls in list(self.model.names.values()):
                        self.cls2config[cls] = geometry_config
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
            self.general_class_names = list(self.model.names.values())
            bbox_class_names = [f"{cls}_bbox" for cls in self.class_names]
            mask_class_names = self.class_names
            self.class_names = bbox_class_names + mask_class_names
            bbox_obj_classes = [
                sly.ObjClass(name, sly.Rectangle) for name in bbox_class_names
            ]
            mask_obj_classes = [
                sly.ObjClass(name, sly.Bitmap) for name in mask_class_names
            ]
            obj_classes = bbox_obj_classes + mask_obj_classes
        elif self.task_type == "pose estimation":
            if self.class_names == ["person_bbox", "person"]:  # human pose estimation
                obj_classes = [
                    sly.ObjClass("person_bbox", sly.Rectangle),
                    sly.ObjClass(
                        "person",
                        sly.GraphNodes,
                        geometry_config=self.keypoints_template,
                    ),
                ]
            elif self.class_names != ["person_bbox", "person"]:
                bbox_obj_classes = [
                    sly.ObjClass(name, sly.Rectangle) for name in bbox_class_names
                ]
                kpt_obj_classes = []
                for name in self.general_class_names:
                    kpt_obj_classes.append(
                        sly.ObjClass(
                            name,
                            sly.GraphNodes,
                            geometry_config=dict_to_template(self.cls2config[name]),
                        )
                    )
                obj_classes = bbox_obj_classes + kpt_obj_classes
        self._model_meta = sly.ProjectMeta(
            obj_classes=sly.ObjClassCollection(obj_classes)
        )
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
        runtime: str,
        fp16: bool = False,
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
        :param runtime: The runtime used for inference. Supported runtimes are PyTorch, ONNXRuntime, and TensorRT.
        :type runtime: str
        :param fp16: If True, the model will be loaded with FP16 precision.
        :type fp16: bool
        """
        self.device = device
        self.runtime = runtime
        self.task_type = task_type
        self.model_source = model_source
        self.model_precision = ModelPrecision.FP16 if fp16 else ModelPrecision.FP32

        # Remove old checkpoint_info.yaml if exists
        sly.fs.silent_remove(os.path.join(self.model_dir, "checkpoint_info.yaml"))

        local_weights_path = os.path.join(self.model_dir, checkpoint_name)
        if not sly.fs.file_exists(local_weights_path):
            self.download(
                src_path=checkpoint_url,
                dst_path=local_weights_path,
            )

        if runtime == RuntimeType.PYTORCH:
            self.model = self._load_pytorch(local_weights_path)
        elif runtime == RuntimeType.ONNXRUNTIME:
            self._check_onnx_device(device)
            self.model = self._load_onnx(local_weights_path)
        elif runtime == RuntimeType.TENSORRT:
            self._check_tensorrt_device(device)
            self.model = self._load_tensorrt(local_weights_path)
            self.max_batch_size = self.TENSORRT_MAX_BATCH_SIZE

        self.load_model_meta(model_source, local_weights_path)

        # Set checkpoint info
        if model_source == "Pretrained models":
            custom_checkpoint_path = None
            checkpoint_name = os.path.splitext(checkpoint_name)[0]
            model_name, architecture = parse_model_name(checkpoint_name)
        else:
            custom_checkpoint_path = checkpoint_url
            file_id = self.api.file.get_info_by_path(self.team_id, checkpoint_url).id
            checkpoint_url = self.api.file.get_url(file_id)
            model_name, architecture = self._try_extract_checkpoint_info(self.model)
        self.checkpoint_info = CheckpointInfo(
            checkpoint_name=checkpoint_name,
            model_name=model_name,
            architecture=architecture,
            checkpoint_url=checkpoint_url,
            custom_checkpoint_path=custom_checkpoint_path,
            model_source=model_source,
        )

        # This will disable logs from YOLO
        # self.model.overrides['verbose'] = False

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
        if self.task_type == "object detection" or dto.class_name.endswith("_bbox"):
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
        elif self.task_type == "instance segmentation" and not dto.class_name.endswith(
            "_bbox"
        ):
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
            tags = []
            if dto.score is not None:
                tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
            label = sly.Label(geometry, obj_class, tags)
        elif self.task_type == "pose estimation" and not dto.class_name.endswith(
            "_bbox"
        ):
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

    def node_label_to_point(self, keypoints):
        n_label2point = {}
        for node, keypoint in zip(self.nodes_order, keypoints):
            n_label2point[node] = keypoint
        return n_label2point

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
                    mask_class_name = self.general_class_names[cls_index]
                    dtos.append(PredictionMask(mask_class_name, mask, confidence))
                    bbox_class_name = self.general_class_names[cls_index] + "_bbox"
                    bbox = [top, left, bottom, right]
                    dtos.append(PredictionBBox(bbox_class_name, bbox, confidence))
        elif self.task_type == "pose estimation":
            boxes_data = prediction.boxes.data
            keypoints_data = prediction.keypoints.data
            point_threshold = settings.get("point_threshold", 0.1)
            if self.class_names == [
                "person_bbox",
                "person",
            ]:  # human pose estimation
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
                            if (
                                point_score >= point_threshold
                                and point_labels[j] != "fictive"
                            ):
                                included_labels.append(point_labels[j])
                                included_point_coordinates.append(
                                    point_coordinate.cpu().numpy()
                                )
                    elif keypoints_data.shape[-1] == 2:
                        for j, point_coordinate in enumerate(keypoints):
                            included_labels.append(point_labels[j])
                            included_point_coordinates.append(
                                point_coordinate.cpu().numpy()
                            )
                    if len(included_labels) > 1:
                        kpt_class_name = self.general_class_names[cls_index]
                        dtos.append(
                            sly.nn.PredictionKeypoints(
                                kpt_class_name,
                                included_labels,
                                included_point_coordinates,
                            )
                        )
                        bbox_class_name = self.general_class_names[cls_index] + "_bbox"
                        bbox = [top, left, bottom, right]
                        dtos.append(PredictionBBox(bbox_class_name, bbox, confidence))
            else:
                for box, keypoints in zip(boxes_data, keypoints_data):
                    left, top, right, bottom, confidence, cls_index = (
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3]),
                        float(box[4]),
                        int(box[5]),
                    )
                    keypoints_template = self.cls2config[
                        self.general_class_names[cls_index]
                    ]
                    point_labels = [
                        value["label"] for value in keypoints_template["nodes"].values()
                    ]
                    node_labels = [
                        node["label"] for node in keypoints_template["nodes"].values()
                    ]
                    n_label2point = self.node_label_to_point(keypoints)
                    included_labels, included_point_coordinates = [], []
                    for j, node_label in enumerate(node_labels):
                        kpt = n_label2point[node_label]
                        point_coordinate, point_score = kpt[:2], kpt[2]
                        if point_score >= point_threshold:
                            included_labels.append(point_labels[j])
                            included_point_coordinates.append(
                                point_coordinate.cpu().numpy()
                            )
                    if len(included_labels) > 1:
                        kpt_class_name = self.general_class_names[cls_index]
                        dtos.append(
                            sly.nn.PredictionKeypoints(
                                kpt_class_name,
                                included_labels,
                                included_point_coordinates,
                            )
                        )
                        bbox_class_name = self.general_class_names[cls_index] + "_bbox"
                        bbox = [top, left, bottom, right]
                        dtos.append(PredictionBBox(bbox_class_name, bbox, confidence))
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
        with sly.nn.inference.Timer() as timer:
            predictions = [
                self._to_dto(prediction, settings) for prediction in predictions
            ]
        to_dto_time = timer.get_time()
        benchmark["postprocess"] += to_dto_time
        return predictions, benchmark

    def _load_pytorch(self, weights_path: str):
        model = YOLO(weights_path)
        model.to(self.device)
        return model

    def _load_runtime(self, weights_path: str, format: str, **kwargs):
        if weights_path.endswith(".pt"):
            exported_weights_path = weights_path.replace(".pt", f".{format}")
            if self.model_precision == ModelPrecision.FP16:
                exported_weights_path = exported_weights_path.replace(
                    f".{format}", f"_fp16.{format}"
                )
            model = None
            if not os.path.exists(exported_weights_path):
                sly.logger.info(f"Exporting model to '{format}' format...")
                if self.gui is not None:
                    bar = self.gui.download_progress(
                        message=f"Exporting model to '{format}' format...", total=1
                    )
                    self.gui.download_progress.show()
                model = YOLO(weights_path)
                model.export(format=format, **kwargs)
                if self.model_precision == ModelPrecision.FP16:
                    # rename after YOLO's export
                    os.rename(
                        weights_path.replace(".pt", f".{format}"), exported_weights_path
                    )
                    sly.fs.silent_remove(weights_path.replace(".pt", f".onnx"))
                if self.gui is not None:
                    bar.update(1)
                    self.gui.download_progress.hide()
            checkpoint_info_path = os.path.join(
                os.path.dirname(weights_path), "checkpoint_info.yaml"
            )
            if self.model_source == ModelSource.CUSTOM and not os.path.exists(
                checkpoint_info_path
            ):
                # save custom checkpoint_info in yaml, as it will be lost after exporting
                if model is None:
                    model = YOLO(weights_path)
                self._dump_yaml_checkpoint_info(model, os.path.dirname(weights_path))
        else:
            exported_weights_path = weights_path

        task_type_map = {
            "object detection": "detect",
            "instance segmentation": "segment",
            "pose estimation": "pose",
        }
        
        model = YOLO(exported_weights_path, task=task_type_map[self.task_type])
        return model

    def _load_onnx(self, weights_path: str):
        return self._load_runtime(weights_path, "onnx", dynamic=True)

    def _load_tensorrt(self, weights_path: str):
        return self._load_runtime(
            weights_path,
            "engine",
            dynamic=False,  # batch size is 1
            # batch=self.TENSORRT_MAX_BATCH_SIZE,
            half=self.model_precision == ModelPrecision.FP16,
        )

    def _check_onnx_device(self, device: str):
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if device.startswith("cuda") and "CUDAExecutionProvider" not in providers:
            raise ValueError(
                f"Selected {device} device, but CUDAExecutionProvider is not available"
            )
        elif device == "cpu" and "CPUExecutionProvider" not in providers:
            raise ValueError(
                f"Selected {device} device, but CPUExecutionProvider is not available"
            )

    def _check_tensorrt_device(self, device: str):
        if "cuda" not in device:
            raise ValueError(
                f"Selected {device} device, but TensorRT only supports CUDA devices"
            )

    def _try_extract_checkpoint_info(self, model: YOLO):
        model_name = None
        architecture = None
        # 1. Extract from checkpoint_info.yaml
        try:
            yaml_path = os.path.join(self.model_dir, "checkpoint_info.yaml")
            if os.path.exists(yaml_path):
                with open(yaml_path, "r") as f:
                    info = yaml.safe_load(f)
                model_name = info["model_name"]
                architecture = info["architecture"]
                return model_name, architecture
        except Exception as e:
            pass

        # 2. Extract from sly_metadata for custom checkpoints
        try:
            model_name = model.ckpt["sly_metadata"]["model_name"]
            architecture = get_arch_from_model_name(model_name)
            assert architecture is not None
            return model_name, architecture
        except Exception as e:
            pass

        # 3. Fallback: trying to extract from train_args for old custom checkpoints
        try:
            train_args = model.ckpt["train_args"]
            model_name = os.path.basename(train_args["model"]).split(".")[0]
            if "yolov" in model_name:
                model_name = model_name.replace("yolov", "YOLOv")
            architecture = get_arch_from_model_name(model_name)
            assert architecture is not None
            return model_name, architecture
        except Exception as e:
            sly.logger.warn(
                f"Failed to extract model_name and architecture from train_args. {repr(e)}",
                exc_info=True,
            )
        return model_name, architecture

    def _dump_yaml_checkpoint_info(self, model: YOLO, dir_path: str):
        model_name, architecture = self._try_extract_checkpoint_info(model)
        info = {
            "model_name": model_name,
            "architecture": architecture,
        }
        yaml_path = os.path.join(dir_path, "checkpoint_info.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(info, f)
        return yaml_path


def parse_model_name(checkpoint_name: str):
    # yolov8n
    if "11" in checkpoint_name:
        p = r"yolo(\d+)(\w)"
    else:
        p = r"yolov(\d+)(\w)"
    match = re.match(p, checkpoint_name.lower())
    version = match.group(1)
    variant = match.group(2)
    if "11" in checkpoint_name:
        model_name = f"YOLO{version}{variant}"
        architecture = f"YOLO{version}"
    else:
        model_name = f"YOLOv{version}{variant}"
        architecture = f"YOLOv{version}"
    return model_name, architecture


def get_arch_from_model_name(model_name: str):
    # yolov8n-det
    p = r"yolov(\d+)"
    match = re.match(p, model_name.lower())
    if match:
        return f"YOLOv{match.group(1)}"
