import supervisely as sly
from supervisely.app.widgets import RadioGroup, Field
from ultralytics import YOLO
import torch
import os
from dotenv import load_dotenv
from pathlib import Path

# from serve.src.keypoints_templates import human_template

from keypoints_templates import human_template

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict

load_dotenv("serve/local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
app_root_directory = str(Path(__file__).parents[2])
det_models_data_path = os.path.join(app_root_directory, "models", "det_models_data.json")
seg_models_data_path = os.path.join(app_root_directory, "models", "seg_models_data.json")
pose_models_data_path = os.path.join(app_root_directory, "models", "pose_models_data.json")
det_models_data = sly.json.load_json_file(det_models_data_path)
seg_models_data = sly.json.load_json_file(seg_models_data_path)
pose_models_data = sly.json.load_json_file(pose_models_data_path)


class YOLOv8Model(sly.nn.inference.PoseEstimation):
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

        return task_type_select_f

    def get_models(self):
        # task_type = self.task_type_select.get_value()
        # if task_type == "object detection":
        #     return det_models_data
        # elif task_type == "instance segmentation":
        #     return seg_models_data
        # elif task_type == "pose estimation":
        #     return pose_models_data
        return det_models_data

    def load_on_device(
        self,
        model_dir,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        model_source = self.gui.get_model_source()
        if model_source == "Pretrained models":
            selected_model = self.gui.get_checkpoint_info()["Model"]
            if selected_model.endswith("det"):
                selected_model = selected_model[:-4]
            model_filename = selected_model.lower() + ".pt"
        model_filename = "yolov8n-pose.pt"
        self.model = YOLO(model_filename)
        self.class_names = ["person_keypoints"]
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self) -> List[str]:
        return self.class_names

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionKeypoints]:
        input_image = sly.image.read(image_path)
        results = self.model(input_image)
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
