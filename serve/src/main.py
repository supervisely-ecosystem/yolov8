import supervisely as sly
from supervisely.nn.prediction_dto import (
    PredictionMask,
    PredictionBBox,
    PredictionKeypoints,
)
from ultralytics import YOLO
import torch
import os
from dotenv import load_dotenv
from src.keypoints_template import human_template, dict_to_template
from typing import List, Any, Dict, Union
from supervisely.app.widgets import (
    TrainedModelsSelector,
    Container,
    RadioTabs,
    PretrainedModelsSelector,
)
from src.models import yolov8_models


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()
team_id = sly.env.team_id()


class YOLOv8Model(sly.nn.inference.ObjectDetection):
    def init_gui(self):
        self.pretrained_models_table = PretrainedModelsSelector(yolov8_models)
        custom_models = sly.nn.checkpoints.yolov8.get_list(api, team_id)
        self.custom_models_table = TrainedModelsSelector(team_id, custom_models)
        self.model_tab_select = RadioTabs(
            titles=["Pretrained models", "Custom models"],
            descriptions=["Publicly available models", "Models trained by you in Supervisely"],
            contents=[self.pretrained_models_table, self.custom_models_table],
        )

        return Container([self.model_tab_select])

    def update_gui(self, is_model_deployed: bool):
        if is_model_deployed:
            self.pretrained_models_table.disable()
            self.custom_models_table.disable()
            self.model_tab_select.disable()
        else:
            self.pretrained_models_table.enable()
            self.custom_models_table.enable()
            self.model_tab_select.enable()

    def set_params_to_ui(self, deploy_params):
        model_tab = deploy_params["model_tab"]
        self.model_tab_select.set_active_tab(deploy_params["model_tab"])
        if model_tab == "Pretrained models":
            self.pretrained_models_table.set_active_task_type(deploy_params["task_type"])
            self.pretrained_models_table.set_active_row(deploy_params["model_id"])
        else:
            self.custom_models_table.set_active_row(deploy_params["model_id"])
            model_row = self.custom_models_table.get_selected_row()
            model_row.checkpoints_selector.set_value(deploy_params["checkpoint_name"])

    def get_params_from_ui(self) -> dict:
        # "Pretrained models" | "Custom models"
        model_tab = self.model_tab_select.get_active_tab()

        # "object detection" | "instance segmentation" | "pose estimation"
        self.task_type = self.pretrained_models_table.get_selected_task_type()
        self.device = self.gui.get_device()

        if model_tab == "Pretrained models":
            selected_model_id = self.pretrained_models_table.get_selected_row_index()
            selected_model = self.pretrained_models_table.get_selected_row()["Model"]
            if selected_model.endswith("det"):
                selected_model = selected_model[:-4]
            model_filename = f"{selected_model.lower()}.pt"
            weights_url = (
                f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_filename}"
            )
        elif model_tab == "Custom models":
            selected_model_id = self.custom_models_table.get_selected_row_index()
            selected_model = self.custom_models_table.get_selected_row()
            weights_url = selected_model.get_selected_checkpoint_path()
            model_filename = selected_model.get_selected_checkpoint_name()

        deploy_params = {
            "device": self.device,
            "model_tab": model_tab,
            "task_type": self.task_type,
            "model_id": selected_model_id,
            "checkpoint_name": model_filename,
            "model_url": weights_url,
        }
        return deploy_params

    def download_weights(self, model_dir: str, model_filename: str, weights_url: str) -> str:
        weights_dst_path = os.path.join(model_dir, model_filename)
        if not sly.fs.file_exists(weights_dst_path):
            self.download(
                src_path=weights_url,
                dst_path=weights_dst_path,
            )
        return weights_dst_path

    def load_model_meta(self, model_tab: str, weights_save_path: str):
        self.class_names = list(self.model.names.values())
        if self.task_type == "pose estimation":
            if model_tab == "Pretrained models":
                self.keypoints_template = human_template
            elif model_tab == "Custom models":
                weights_dict = torch.load(weights_save_path)
                geometry_config = weights_dict["geometry_config"]
                self.keypoints_template = dict_to_template(geometry_config)
        if self.task_type == "object detection":
            obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in self.class_names]
        elif self.task_type == "instance segmentation":
            obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in self.class_names]
        elif self.task_type == "pose estimation":
            obj_classes = [
                sly.ObjClass(name, sly.GraphNodes, geometry_config=self.keypoints_template)
                for name in self.class_names
            ]
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        self._get_confidence_tag_meta()

    def load_model(
        self,
        model_tab: str,
        device: str,
        task_type: str,
        model_id: int,  # used to set gui
        checkpoint_name: str,
        model_url: str,
    ):
        self.task_type = task_type
        local_weights_path = self.download_weights(self.model_dir, checkpoint_name, model_url)
        self.model = YOLO(local_weights_path)
        if device.startswith("cuda"):
            if device == "cuda":
                self.device = 0
            else:
                self.device = int(device[-1])
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.load_model_meta(model_tab, local_weights_path)

    # def load_on_device(
    #     self,
    #     model_dir,
    #     device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    #     started_via_api: bool = False,
    #     deploy_params: Dict[str, Any] = None,
    # ):
    #     if not started_via_api:
    #         model_tab = self.gui.get_model_tab()
    #         if model_tab == "Pretrained models":
    #             self.task_type = self.task_type_select.get_value()
    #             selected_model = self.gui.get_checkpoint_info()["Model"]
    #             if selected_model.endswith("det"):
    #                 selected_model = selected_model[:-4]
    #             model_filename = selected_model.lower() + ".pt"
    #             weights_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_filename}"
    #             weights_dst_path = os.path.join(model_dir, model_filename)
    #             if not sly.fs.file_exists(weights_dst_path):
    #                 self.download(
    #                     src_path=weights_url,
    #                     dst_path=weights_dst_path,
    #                 )
    #         elif model_tab == "Custom models":
    #             self.task_type = self.custom_task_type_select.get_value()
    #             custom_link = self.gui.get_custom_link()
    #             weights_file_name = os.path.basename(custom_link)
    #             weights_dst_path = os.path.join(model_dir, weights_file_name)
    #             if not sly.fs.file_exists(weights_dst_path):
    #                 self.download(
    #                     src_path=custom_link,
    #                     dst_path=weights_dst_path,
    #                 )
    #     else:
    #         # ------------------------ #
    #         model_tab = deploy_params["model_tab"]  # "pretrained" / "custom"
    #         self.task_type = deploy_params["task_type"]
    #         weights_file_name = deploy_params["checkpoint_name"]

    #         if model_tab == "Pretrained models":
    #             if weights_file_name.endswith("det"):
    #                 weights_file_name = weights_file_name[:-4]
    #             weights_file_name = weights_file_name.lower() + ".pt"
    #             weights_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{weights_file_name}"
    #             weights_dst_path = os.path.join(model_dir, weights_file_name)
    #             if not sly.fs.file_exists(weights_dst_path):
    #                 self.download(
    #                     src_path=weights_url,
    #                     dst_path=weights_dst_path,
    #                 )

    #         elif model_tab == "Custom models":
    #             weights_dst_path = os.path.join(model_dir, weights_file_name)
    #             custom_weights_path = deploy_params["model_url"]
    #             if not sly.fs.file_exists(weights_dst_path):
    #                 self.download(
    #                     src_path=custom_weights_path,
    #                     dst_path=weights_dst_path,
    #                 )
    #     # ------------------------ #

    #     self.model = YOLO(weights_dst_path)
    #     self.class_names = list(self.model.names.values())
    #     if device.startswith("cuda"):
    #         if device == "cuda":
    #             self.device = 0
    #         else:
    #             self.device = int(device[-1])
    #     else:
    #         self.device = "cpu"
    #     self.model.to(self.device)
    #     if self.task_type == "pose estimation":
    #         if model_tab == "Pretrained models":
    #             self.keypoints_template = human_template
    #         elif model_tab == "Custom models":
    #             weights_dict = torch.load(weights_dst_path)
    #             geometry_config = weights_dict["geometry_config"]
    #             self.keypoints_template = dict_to_template(geometry_config)
    #     if self.task_type == "object detection":
    #         obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in self.class_names]
    #     elif self.task_type == "instance segmentation":
    #         obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in self.class_names]
    #     elif self.task_type == "pose estimation":
    #         obj_classes = [
    #             sly.ObjClass(name, sly.GraphNodes, geometry_config=self.keypoints_template)
    #             for name in self.class_names
    #         ]
    #     self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
    #     self._get_confidence_tag_meta()

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
                geometry = sly.Bitmap(dto.mask)
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

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[Union[PredictionMask, PredictionBBox, PredictionKeypoints]]:
        input_image = sly.image.read(image_path)
        # RGB to BGR
        input_image = input_image[:, :, ::-1]
        if self.task_type == "instance segmentation":
            retina_masks = True
            # input_height, input_width = input_image.shape[:2]
            # resolution = 640
            # scaler = max(input_width, input_height) / resolution
            # resized_width = int(input_width / scaler)
            # resized_height = int(input_height / scaler)
            # input_image = input_image.transpose(2, 0, 1)
            # input_image = torch.from_numpy(input_image.copy())
            # input_image = torch.unsqueeze(input_image, 0)
            # input_image = torch.nn.functional.interpolate(
            #     input_image, (resized_height, resized_width), mode="nearest"
            # )
            # input_image = input_image.squeeze().numpy()
            # input_image = input_image.transpose(1, 2, 0)
        else:
            retina_masks = False
        predictions = self.model(
            source=input_image,
            conf=settings["conf"],
            iou=settings["iou"],
            half=settings["half"],
            device=self.device,
            max_det=settings["max_det"],
            agnostic_nms=settings["agnostic_nms"],
            retina_masks=retina_masks,
        )
        results = []
        if self.task_type == "object detection":
            boxes_data = predictions[0].boxes.data
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
                results.append(PredictionBBox(self.class_names[cls_index], bbox, confidence))
        elif self.task_type == "instance segmentation":
            boxes_data = predictions[0].boxes.data
            masks = predictions[0].masks.data
            for box, mask in zip(boxes_data, masks):
                left, top, right, bottom, confidence, cls_index = (
                    int(box[0]),
                    int(box[1]),
                    int(box[2]),
                    int(box[3]),
                    float(box[4]),
                    int(box[5]),
                )
                # mask = torch.unsqueeze(mask, 0)
                # mask = torch.unsqueeze(mask, 0)
                # mask = torch.nn.functional.interpolate(
                #     mask, (input_height, input_width), mode="nearest"
                # )
                # mask = mask.squeeze()
                mask = mask.cpu().numpy()
                results.append(PredictionMask(self.class_names[cls_index], mask, confidence))
        elif self.task_type == "pose estimation":
            boxes_data = predictions[0].boxes.data
            keypoints_data = predictions[0].keypoints.data
            point_labels = self.keypoints_template.point_names
            point_threshold = settings.get("point_threshold", 0.1)
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
                    point_coordinates, point_scores = keypoints[:, :2], keypoints[:, 2]
                    for j, (point_coordinate, point_score) in enumerate(
                        zip(point_coordinates, point_scores)
                    ):
                        if point_score >= point_threshold:
                            included_labels.append(point_labels[j])
                            included_point_coordinates.append(point_coordinate.cpu().numpy())
                elif keypoints_data.shape[-1] == 2:
                    for j, point_coordinate in enumerate(keypoints):
                        included_labels.append(point_labels[j])
                        included_point_coordinates.append(point_coordinate.cpu().numpy())
                if len(included_labels) > 1:
                    results.append(
                        sly.nn.PredictionKeypoints(
                            self.class_names[cls_index],
                            included_labels,
                            included_point_coordinates,
                        )
                    )
        return results


m = YOLOv8Model(
    use_gui=True,
    # custom_inference_settings=os.path.join(root_source_path, "serve", "custom_settings.yaml"),
)

if sly.is_production():
    m.serve()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    deploy_params = m.get_params_from_ui()
    m.load_model(**deploy_params)
    image_path = "./demo_data/image_01.jpg"
    settings = {
        "conf": 0.25,
        "iou": 0.7,
        "half": False,
        "max_det": 300,
        "agnostic_nms": False,
    }
    results = m.predict(image_path, settings=settings)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=5)
    print(f"predictions and visualization have been saved: {vis_path}")
