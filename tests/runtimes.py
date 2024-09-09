import os
import sys
sys.path.append("serve")
from dotenv import load_dotenv
from pathlib import Path
import supervisely as sly
from serve.src.yolov8 import YOLOv8Model
from supervisely.nn.inference import RuntimeType

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
    'device': 'cuda:0',
    'runtime': RuntimeType.TENSORRT,
    'model_source': 'Pretrained models',
    'task_type': 'object detection',
    'checkpoint_name': 'yolov8n-det (coco).pt',
    'checkpoint_url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8n.pt'
}
m._load_model(deploy_params)
# m._inference_image_id(m.api, {"image_id": test_image_id})
m._inference_batch_ids(m.api, {"batch_ids": [31450547, 31447888]})
m._inference_batch_ids(m.api, {"batch_ids": [31450547, 31447888]})