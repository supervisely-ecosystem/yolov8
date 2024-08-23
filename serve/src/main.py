import os
from dotenv import load_dotenv

load_dotenv("local.env")
debug_session = bool(os.environ.get("DEBUG_SESSION", False)) # FIXME: move it to serve/yolov8 after resolving merge conflicts

from pathlib import Path
import torch
from dotenv import load_dotenv

import supervisely as sly


from src.yolov8 import YOLOv8Model
# if debug_session: # FIXME: move it to serve/yolov8 after resolving merge conflicts
#     import serve.src.workflow as w
#     from serve.src.keypoints_template import dict_to_template, human_template
#     from serve.src.models import yolov8_models
# else:
#     from src.keypoints_template import dict_to_template, human_template
#     from src.models import yolov8_models
#     import src.workflow as w



load_dotenv("supervisely.env")
root_source_path = str(Path(__file__).parents[2])
api = sly.Api.from_env()
team_id = sly.env.team_id()

m = YOLOv8Model(
    model_dir="app_data",
    use_gui=True,
    custom_inference_settings=os.path.join(
        root_source_path, "serve", "custom_settings.yaml"
    ),
)

if sly.is_production():
    m.serve()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    deploy_params = m.get_params_from_gui()
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
