import os
from pathlib import Path
import supervisely as sly
from dotenv import load_dotenv

load_dotenv("train/local.env")

app_root_directory = str(Path(__file__).parents[2])
app_data_dir = os.path.join(app_root_directory, "tempfiles")
project_dir = os.path.join(app_data_dir, "project_dir")
yolov8_project_dir = os.path.join(app_data_dir, "yolov8_project_dir")

# app_session_id = sly.io.env.task_id()
app_session_id = 777  # for debug

det_models_data_path = os.path.join(app_root_directory, "models", "det_models_data.json")
seg_models_data_path = os.path.join(app_root_directory, "models", "seg_models_data.json")
pose_models_data_path = os.path.join(app_root_directory, "models", "pose_models_data.json")
det_models_data = sly.json.load_json_file(det_models_data_path)
seg_models_data = sly.json.load_json_file(seg_models_data_path)
pose_models_data = sly.json.load_json_file(pose_models_data_path)

train_params_filepath = os.path.join(app_root_directory, "train", "training_params.yaml")
