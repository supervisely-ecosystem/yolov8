import supervisely as sly
from supervisely.app.widgets import RadioGroup, Field
from supervisely.project.project_meta import ProjectMeta
from supervisely.annotation.obj_class import ObjClass
from supervisely.imaging.color import get_predefined_colors
from supervisely.nn.prediction_dto import PredictionMask, PredictionBBox, PredictionKeypoints
from ultralytics import YOLO
import torch
import os
from dotenv import load_dotenv
from pathlib import Path

from src.keypoints_template import human_template

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict, Union

load_dotenv("serve/local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
app_root_directory = str(Path(__file__).parents[2])
det_models_data_path = os.path.join(app_root_directory, "models", "det_models_data.json")
seg_models_data_path = os.path.join(app_root_directory, "models", "seg_models_data.json")
pose_models_data_path = os.path.join(app_root_directory, "models", "pose_models_data.json")
det_models_data = sly.json.load_json_file(det_models_data_path)
seg_models_data = sly.json.load_json_file(seg_models_data_path)
pose_models_data = sly.json.load_json_file(pose_models_data_path)


class YOLOv8Model(sly.nn.inference.ObjectDetection):
    def add_content_to_pretrained_tab(self, gui):
        task_type_items = [
            RadioGroup.Item(value="object detection"),
            RadioGroup.Item(value="instance segmentation"),
            RadioGroup.Item(value="pose estimation"),
        ]
        self.task_type_select = RadioGroup(items=task_type_items, direction="vertical")
        task_type_select_f = Field(
            content=self.task_type_select,
            title="Task type",
        )

        @self.task_type_select.value_changed
        def change_table(task_type):
            if task_type == "object detection":
                models_table_columns = [key for key in det_models_data[0].keys()]
                models_table_subtitles = [None] * len(models_table_columns)
                models_table_rows = []
                for element in det_models_data:
                    models_table_rows.append(list(element.values()))
                gui._models_table.set_data(
                    columns=models_table_columns,
                    rows=models_table_rows,
                    subtitles=models_table_subtitles,
                )
                gui._models = det_models_data
            elif task_type == "instance segmentation":
                models_table_columns = [key for key in seg_models_data[0].keys()]
                models_table_subtitles = [None] * len(models_table_columns)
                models_table_rows = []
                for element in seg_models_data:
                    models_table_rows.append(list(element.values()))
                gui._models_table.set_data(
                    columns=models_table_columns,
                    rows=models_table_rows,
                    subtitles=models_table_subtitles,
                )
                gui._models = seg_models_data
            elif task_type == "pose estimation":
                models_table_columns = [key for key in pose_models_data[0].keys()]
                models_table_subtitles = [None] * len(models_table_columns)
                models_table_rows = []
                for element in pose_models_data:
                    models_table_rows.append(list(element.values()))
                gui._models_table.set_data(
                    columns=models_table_columns,
                    rows=models_table_rows,
                    subtitles=models_table_subtitles,
                )
                gui._models = pose_models_data

        return task_type_select_f

    def add_content_to_custom_tab(self, gui):
        task_type_items = [
            RadioGroup.Item(value="object detection"),
            RadioGroup.Item(value="instance segmentation"),
            RadioGroup.Item(value="pose estimation"),
        ]
        self.custom_task_type_select = RadioGroup(items=task_type_items, direction="vertical")
        custom_task_type_select_f = Field(
            content=self.custom_task_type_select,
            title="Task type",
        )
        return custom_task_type_select_f

    def get_models(self):
        return det_models_data

    def load_on_device(
        self,
        model_dir,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        self.task_type = self.task_type_select.get_value()
        model_source = self.gui.get_model_source()
        if model_source == "Pretrained models":
            selected_model = self.gui.get_checkpoint_info()["Model"]
            if selected_model.endswith("det"):
                selected_model = selected_model[:-4]
            model_filename = selected_model.lower() + ".pt"
        self.model = YOLO(model_filename)
        self.class_names = list(self.model.names.values())
        if self.task_type_select.get_value() == "pose estimation":
            if model_source == "Pretrained models":
                self.keypoints_template = human_template
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_info(self):
        info = super().get_info()
        info["task type"] = self.task_type
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    @property
    def model_meta(self) -> ProjectMeta:
        if self._model_meta is None:
            if self.task_type in ["object detection", "instance segmentation"]:
                colors = get_predefined_colors(len(self.get_classes()))
                classes = []
                for name, rgb in zip(self.get_classes(), colors):
                    classes.append(ObjClass(name, self._get_obj_class_shape(), rgb))
                self._model_meta = ProjectMeta(classes)
                self._get_confidence_tag_meta()
            elif self.task_type == "pose estimation":
                classes = []
                for name in self.get_classes():
                    classes.append(
                        ObjClass(
                            name,
                            self._get_obj_class_shape(),
                            geometry_config=self.keypoints_template,
                        )
                    )
                self._model_meta = ProjectMeta(classes)
                self._get_confidence_tag_meta()
        return self._model_meta

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
            return label
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
        results = self.model(input_image)
        if self.task_type == "object detection":
            boxes = results[0].boxes
        elif self.task_type == "instance segmentation":
            masks = results[0].masks
        elif self.task_type == "pose estimation":
            keypoints = results[0].data.keypoints
        return results


m = YOLOv8Model(
    use_gui=True,
    custom_inference_settings=os.path.join(app_root_directory, "serve", "custom_settings.yaml"),
)

if sly.is_production():
    m.serve()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, settings={})
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=0)
    print(f"predictions and visualization have been saved: {vis_path}")
