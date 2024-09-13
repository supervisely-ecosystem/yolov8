import sys, os
sys.path.append(os.path.abspath("./"))
from serve.src.monkey_patching_fix import monkey_patching_fix
monkey_patching_fix()

from dotenv import load_dotenv
from tqdm import tqdm
from ultralytics import YOLO

load_dotenv("local.env")
load_dotenv("supervisely.env")


def run_speedtest(model: YOLO, batch_size: int, device: str, imgsz: int = 640, n: int = 100):
    import time
    import numpy as np

    def get_input():
        return [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(batch_size)]
    
    warmup = 3
    times = np.zeros(n + warmup)
    pbar = tqdm(range(n + warmup))
    for i in pbar:
        dummy_input = get_input()
        start = time.time()
        model(dummy_input, device=device, verbose=False)
        times[i] = (time.time() - start) * 1000  # ms
        pbar.set_postfix(time=f"{times[i]/batch_size:.2f}ms")
    times = times[warmup:] / batch_size
    print(f"Avg. time: {times.mean():.2f} ms (per image), std: {times.std():.2f} ms")
    print(f"FPS: {1000 / times.mean():.2f}")
    print(f"imgsz={imgsz}, batch_size={batch_size}, device={device}")


MODEL_NAME = "yolov10n"


# export
model = YOLO(f"app_data/{MODEL_NAME}.pt")
model.export(format="engine", dynamic=True, batch=4, half=True)


# TensorRT GPU
print("TensorRT GPU FP32")
device = "cuda"
model = YOLO(f"app_data/{MODEL_NAME}_fp32.engine", task="detect")
print("TensorRT GPU")
run_speedtest(model, batch_size=4, device=device)


# TensorRT GPU
print("TensorRT GPU FP16")
device = "cuda"
model = YOLO(f"app_data/{MODEL_NAME}.engine", task="detect")
print("TensorRT GPU")
run_speedtest(model, batch_size=4, device=device)


# PyTorch GPU
device = "cuda"
model = YOLO(f"app_data/{MODEL_NAME}.pt", task="detect")
print("PyTorch GPU")
run_speedtest(model, batch_size=4, device=device)

# ONNX GPU
device = "cuda"
model = YOLO(f"app_data/{MODEL_NAME}.onnx", task="detect")
print("ONNX GPU")
run_speedtest(model, batch_size=4, device=device)

# TensorRT GPU
device = "cuda"
model = YOLO(f"app_data/{MODEL_NAME}.engine", task="detect")
print("TensorRT GPU")
run_speedtest(model, batch_size=4, device=device)


# PyTorch CPU
device = "cpu"
model = YOLO(f"app_data/{MODEL_NAME}.pt", task="detect")
print("PyTorch CPU")
run_speedtest(model, batch_size=1, device=device)

# ONNX CPU
device = "cpu"
model = YOLO(f"app_data/{MODEL_NAME}.onnx", task="detect")
print("ONNX CPU")
run_speedtest(model, batch_size=1, device=device)
