from src.monkey_patching_fix import monkey_patching_fix
# Monkey patching to avoid a problem with installing wrong onnxruntime requirements
# issue: https://github.com/ultralytics/ultralytics/issues/5093
monkey_patching_fix()

import os
from ultralytics import YOLO


def export_checkpoint(weights_path: str, format: str, fp16=False, dynamic=False, **kwargs):
    exported_weights_path = weights_path.replace(".pt", f".{format}")
    if fp16:
        exported_weights_path = exported_weights_path.replace(f".{format}", f"_fp16.{format}")
    model = YOLO(weights_path)
    model.export(format=format, half=fp16, dynamic=dynamic, **kwargs)
    if fp16:
        # add '_fp16' suffix after YOLO's export
        os.rename(weights_path.replace(".pt", f".{format}"), exported_weights_path)
        if format == "engine":
            # also rename onnx file
            os.rename(
                weights_path.replace(".pt", f".onnx"),
                exported_weights_path.replace(".engine", ".onnx")
            )
    return exported_weights_path