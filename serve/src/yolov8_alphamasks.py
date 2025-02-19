from src.monkey_patching_fix import monkey_patching_fix

monkey_patching_fix()

import os
from typing import Dict, List, Literal

import cv2
import numpy as np
from ultralytics import YOLO

import supervisely as sly
from supervisely.nn.inference import CheckpointInfo
from supervisely.nn.prediction_dto import PredictionAlphaMask


class AttentionMapModel(sly.nn.inference.InstanceProbabilitySegmentation):

    def load_model_meta(self, model_source: str, weights_save_path: str):
        self.classes = list(self.model.names.values())
        obj_classes = [sly.ObjClass(name, sly.AlphaMask) for name in self.classes]
        self._model_meta = sly.ProjectMeta(obj_classes=obj_classes)

    def load_model(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        model_source: Literal["Pretrained models", "Custom models"],
        task_type: Literal["instance segmentation"],
        checkpoint_name: str,
        checkpoint_url: str,
        runtime: str,
    ):
        self.device = device
        self.runtime = runtime
        self.task_type = task_type
        self.model_source = model_source

        # Remove old checkpoint_info.yaml if exists
        sly.fs.silent_remove(os.path.join(self.model_dir, "checkpoint_info.yaml"))
        local_weights_path = os.path.join(self.model_dir, checkpoint_name)
        if not sly.fs.file_exists(local_weights_path):
            self.download(src_path=checkpoint_url, dst_path=local_weights_path)

        self.model = YOLO(local_weights_path)
        self.model.to(self.device)
        self.load_model_meta(model_source, local_weights_path)

        self.checkpoint_info = CheckpointInfo(
            checkpoint_name=checkpoint_name,
            model_name="YOLOv8n-seg",
            architecture="YOLOv8",
            checkpoint_url=checkpoint_url,
            model_source="Pretrained models",
        )

    def _only_for_debug_method(self, mask):
        mask = mask.copy()
        mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return mask

    def predict_batch(self, images_np: List[np.ndarray], settings: Dict):
        """
        Predicts a batch of images.

        Returns a list of lists of PredictionAlphaMask objects (one list per image).

        PredictionAlphaMask object contains:
        - class_name: str
        - mask: np.ndarray numpy array normalized to [0, 255] with dtype=np.uint8
        - score: Optional[float]
        """
        # RGB to BGR
        images_np = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np]

        # Predict
        predictions = self.model(
            source=images_np,
            conf=settings["conf"],
            iou=settings["iou"],
            half=settings["half"],
            device=self.device,
            max_det=settings["max_det"],
            agnostic_nms=settings["agnostic_nms"],
            retina_masks=True,
        )

        # Convert predictions to PredictionAlphaMask objects for each image
        results = []
        for prediction in predictions:
            if not prediction.masks:
                continue
            temp_results = []
            for data, mask in zip(prediction.boxes.data, prediction.masks.data):
                # data: [x1, y1, x2, y2, confidence, cls_index]
                confidence, cls_index = list(map(int, data[4:]))
                mask_class_name = self.classes[cls_index]

                # mask: numpy array of shape (H, W) with values from 0 to 255
                mask = mask.cpu().numpy()
                mask = self._only_for_debug_method(mask)

                # create PredictionAlphaMask object
                dto = PredictionAlphaMask(mask_class_name, mask, confidence)
                temp_results.append(dto)
            results.append(temp_results)
        return results
