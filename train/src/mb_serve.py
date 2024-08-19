import os
import sys
from pathlib import Path
from typing import Literal

import supervisely as sly
from dotenv import load_dotenv
from supervisely.nn.inference import CheckpointInfo
from ultralytics import YOLO

root_source_path = Path(__file__).parent.parent.parent
serve_path = root_source_path.joinpath("serve")
if not serve_path.exists():
    raise FileNotFoundError(f"Not found serve module for model benchmark evaluation: {serve_path}")

sys.path.insert(0, str(serve_path.resolve()))

from serve.src.main import YOLOv8Model

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

root_source_path = str(Path(__file__).parents[2])

api = sly.Api.from_env()
team_id = sly.env.team_id()


class YOLOv8ModelBM(YOLOv8Model):

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
        self.checkpoint_info = CheckpointInfo(checkpoint_name, "yolov8", model_source)  # TODO name
