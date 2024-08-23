import os
from pathlib import Path
import supervisely as sly
from dotenv import load_dotenv

load_dotenv("local.env")

debug_session = bool(os.environ.get("DEBUG_SESSION", False))
root_source_path = str(Path(__file__).parents[2])
app_root_directory = str(Path(__file__).parents[1])
app_data_dir = os.path.join(app_root_directory, "tempfiles")
project_dir = os.path.join(app_data_dir, "project_dir")
yolov8_project_dir = os.path.join(app_data_dir, "yolov8_project_dir")

if sly.is_production():
    app_session_id = sly.io.env.task_id()
    root_model_checkpoint_dir = sly.app.get_synced_data_dir()
else:
    app_session_id = 60466  # for debug
    root_model_checkpoint_dir = os.path.join(app_root_directory, "runs")


det_models_data_path = os.path.join(root_source_path, "models", "det_models_data.json")
seg_models_data_path = os.path.join(root_source_path, "models", "seg_models_data.json")
pose_models_data_path = os.path.join(root_source_path, "models", "pose_models_data.json")
det_models_data = sly.json.load_json_file(det_models_data_path)
seg_models_data = sly.json.load_json_file(seg_models_data_path)
pose_models_data = sly.json.load_json_file(pose_models_data_path)

if sly.is_production() or debug_session:
    train_params_filepath = "train/training_params.yml"
else:
    train_params_filepath = "training_params.yml"  # for debug
train_counter, val_counter = 0, 0
center_matches = {}
keypoints_template = None
keypoints_classes = []
