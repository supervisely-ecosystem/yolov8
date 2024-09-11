import os
import sys
sys.path.append("serve")
from dotenv import load_dotenv
from pathlib import Path
import supervisely as sly
from serve.src.yolov8 import YOLOv8Model
from supervisely.nn import utils, TaskType

load_dotenv("local.env")
load_dotenv("supervisely.env")
root_source_path = str(Path(__file__).parents[1])

m = YOLOv8Model(
    model_dir="app_data",
    use_gui=False,
    custom_inference_settings=os.path.join(
        root_source_path, "serve", "custom_settings.yaml"
    ),
)

test_image_id = 31180449
deploy_params = {
    'device': 'cuda',
    'runtime': utils.RuntimeType.TENSORRT,
    'model_source': utils.ModelSource.PRETRAINED,
    'task_type': TaskType.OBJECT_DETECTION,
    'checkpoint_name': 'YOLOv8s-det (Open Images V7).pt'.lower(),
    'checkpoint_url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-oiv7.pt',
    'fp16': False,
}
m._load_model(deploy_params)
m._inference_image_id(m.api, {"image_id": test_image_id})
m._inference_batch_ids(m.api, {"batch_ids": [31450547, 31447888]})
m._inference_batch_ids(m.api, {"batch_ids": [31450547, 31447888]})