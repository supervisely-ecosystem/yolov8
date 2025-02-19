import os

from dotenv import load_dotenv
from src.yolov8_alphamasks import AttentionMapModel

import supervisely as sly

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

m = AttentionMapModel(model_dir="app_data",custom_inference_settings="serve/custom_settings.yaml")

m.serve()
deploy_params = {
    "device": "cpu",
    "model_source": "Pretrained models",
    "task_type": "instance segmentation",
    "checkpoint_name": "YOLOv8n-seg.pt",
    "checkpoint_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8n-seg.pt",
    "runtime": "PyTorch",
}
m._load_model(deploy_params)
# # image_path = "serve/demo_data/image_01.jpg"
# # settings = {
# #     "conf": 0.25,
# #     "iou": 0.7,
# #     "half": False,
# #     "max_det": 300,
# #     "agnostic_nms": False,
# # }
# # img = sly.image.read(image_path)
# # results = m.predict_batch([img], settings=settings)[0]
# # ann = m._predictions_to_annotation(img, results)
# # vis_path = f"serve/demo_data/image_01_prediction_{sly.rand_str(5)}.jpg"
# # ann.draw(img)
# # sly.image.write(vis_path, img)
# # sly.json.dump_json_file(ann.to_json(), f"serve/demo_data/image_01_prediction_{sly.rand_str(5)}.json")
# # print(f"predictions and visualization have been saved: {vis_path}")

# api = sly.Api.from_env()
# new_project = api.project.create(7, "test", change_name_if_conflict=True)
# state = {}
# state["projectId"] = 409
# state["batch_size"] = 1  # 8
# state["output_project_id"] = new_project.id
# m._inference_project_id(api, state)
