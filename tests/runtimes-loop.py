import os
import sys
sys.path.append("serve")
from tqdm import tqdm
import supervisely as sly
from dotenv import load_dotenv
from serve.src.yolov8 import YOLOv8Model
from serve.src.models import yolov8_models
from supervisely.nn.inference import RuntimeType


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


# Set custom model path
custom_model_url = "/yolov8_train_tmp/object detection/TACO-10 trainset/57436/weights/best_277.pt"
det_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt"
seg_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8n-seg.pt"
pose_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8n-pose.pt"

img_path = "tests/images/giraffe.jpg"
img_np = sly.image.read(img_path)

output_dir = "app_data/results"


models = [
    ("object detection", det_url),
    ("instance segmentation", seg_url),
    ("pose estimation", pose_url),
]

m = YOLOv8Model(
    model_dir="app_data",
    use_gui=False,
    custom_inference_settings="serve/custom_settings.yaml"
)

def ensure_taco():
    global img_np, img_path
    img_path = "app_data/taco.jpg"
    taco_img_id = 30402303
    if not os.path.exists(img_path):
        api = sly.Api()
        api.image.download_path(taco_img_id, img_path)
    img_np = sly.image.read(img_path)


def test_pretrained():
    for runtime in [RuntimeType.PYTORCH, RuntimeType.ONNXRUNTIME, RuntimeType.TENSORRT]:
        for task, url in tqdm(models):
            name = os.path.basename(url)
            m.load_model(
                device="cuda",
                model_source="Pretrained models",
                task_type=task,
                checkpoint_name=name,
                checkpoint_url=url,
                runtime=runtime,
            )
            settings = m.custom_inference_settings_dict
            # settings["conf"] = 0.45
            pred = m.predict_benchmark([img_np], settings)[0][0]
            m.visualize(pred, img_path, f"{output_dir}/result_{runtime}_{name}.jpg", thickness=3)


def test_custom():
    ensure_taco()
    task = "object detection"
    for runtime in [RuntimeType.PYTORCH, RuntimeType.ONNXRUNTIME, RuntimeType.TENSORRT]:
        m.load_model(
            device="cuda",
            model_source="Custom model",
            task_type=task,
            checkpoint_name="custom.pt",
            checkpoint_url=custom_model_url,
            runtime=runtime,
        )
        settings = m.custom_inference_settings_dict
        # settings["conf"] = 0.45
        pred = m.predict_benchmark([img_np], settings)[0][0]
        m.visualize(pred, img_path, f"{output_dir}/result_taco_{runtime}_{task}.jpg", thickness=5)


def assert_checkpoint_info(m: YOLOv8Model):
    info = m.checkpoint_info
    assert None not in [
        info.checkpoint_name,
        info.model_name,
        info.architecture,
    ], "Checkpoint info is not fully set"

test_pretrained()
test_custom()
