import sys, os
sys.path.append(os.path.abspath("./"))
from serve.src.monkey_patching_fix import monkey_patching_fix
monkey_patching_fix()

from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
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


MODEL_NAME = "yolov10s-det"

# PyTorch GPU (original)
device = "cuda"
model = YOLO(f"app_data/{MODEL_NAME}.pt")

# Export to TensorRT
os.remove(f"app_data/{MODEL_NAME}_fp16.engine")  # remove old engine
if not os.path.exists(f"app_data/{MODEL_NAME}_fp16.engine"):
    model.export(format="engine", dynamic=True, half=True, batch=4, imgsz=320)
    os.rename(f"app_data/{MODEL_NAME}.engine", f"app_data/{MODEL_NAME}_fp16.engine")

# PyTorch GPU inference
preds_original = model(images, device=device)
assert len(preds_original) == len(images)
assert_device(preds_original, device)

# ONNX GPU
# device = "cuda"
# model = YOLO(f"app_data/{MODEL_NAME}.onnx")
# preds_onnx = model(images, device=device)
# assert len(preds_onnx) == len(images)
# assert_device(preds_onnx, device)
# compare_predictions(preds_original, preds_onnx, atol=1e-2)

# ONNX CPU
# device = "cpu"
# model = YOLO(f"app_data/{MODEL_NAME}.onnx")
# preds_onnx = model(images, device=device)
# assert len(preds_onnx) == len(images)
# assert_device(preds_onnx, device)
# compare_predictions(preds_original, preds_onnx, atol=1e-2)

# TensorRT GPU FP16
device = "cuda"
model = YOLO(f"app_data/{MODEL_NAME}_fp16.engine")
preds_trt = model(images, device=device)
assert len(preds_trt) == len(images)
assert_device(preds_trt, device)
compare_predictions(preds_original, preds_trt, atol=0.5)
