# Monkey patching to avoid a problem with installing onnxruntime requirements
# issue: https://github.com/ultralytics/ultralytics/issues/5093
import importlib
import ultralytics.nn.autobackend
import ultralytics.utils.checks as checks
check_requirements = checks.check_requirements  # save original function
def check_requirements_dont_install(*args, **kwargs):
    kwargs["install"] = False
    return check_requirements(*args, **kwargs)
checks.check_requirements = check_requirements_dont_install
importlib.reload(ultralytics.nn.autobackend)

from dotenv import load_dotenv
from pathlib import Path
from ultralytics import YOLO
import cv2
import torch

load_dotenv("local.env")
load_dotenv("supervisely.env")

image_paths = Path("tests/images").glob("*.jpg")
images = [cv2.imread(str(path)) for path in image_paths]  # BGR images
# images = [images[0]]
assert len(images) >= 2  # for batch testing

def compare_predictions(preds1: list, preds2: list, atol: float = 1e-2):
    for pred1, pred2 in zip(preds1, preds2):
        for box1, box2 in zip(pred1.boxes.data, pred2.boxes.data):
            assert torch.allclose(box1.cpu(), box2.cpu(), atol=atol), f"boxes are not close with tolerance atol={atol}. box1={box1}, box2={box2}"


def assert_device(preds: list, device: str):
    for pred in preds:
        assert pred.boxes.data.device.type == device, f"predictions are not on {device} device"


# PyTorch GPU (original)
device = "cuda"
model = YOLO("app_data/yolov8n-det (coco).pt")
preds_original = model(images, device=device)
assert len(preds_original) == len(images)
assert_device(preds_original, device)

# # ONNX GPU
# device = "cuda"
# model = YOLO("app_data/yolov8n-det (coco).onnx")
# preds_onnx = model(images, device=device)
# assert len(preds_onnx) == len(images)
# assert_device(preds_onnx, device)
# compare_predictions(preds_original, preds_onnx, atol=1e-2)

# # ONNX CPU
# device = "cpu"
# model = YOLO("app_data/yolov8n-det (coco).onnx")
# preds_onnx = model(images, device=device)
# assert len(preds_onnx) == len(images)
# assert_device(preds_onnx, device)
# compare_predictions(preds_original, preds_onnx, atol=1e-2)

# model.export(format="engine", dynamic=True, batch=16)

# TensorRT GPU
device = "cuda"
model = YOLO("app_data/yolov8n-det (coco).engine")
preds_trt = model(images, device=device)
assert len(preds_trt) == len(images)
assert_device(preds_trt, device)
compare_predictions(preds_original, preds_trt, atol=0.1)
