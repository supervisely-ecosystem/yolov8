import os
from pathlib import Path
from time import sleep

import requests
import supervisely as sly
from dotenv import load_dotenv


load_dotenv("supervisely.env")
load_dotenv("local.env")

api = sly.Api()

# get env variables
GLOBAL_TIMEOUT = 1  # seconds
AGENT_ID = 341  # agent id to run training on
APP_NAME = "supervisely-ecosystem/yolov8/train"
PROJECT_ID = sly.env.project_id()
TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()
DATA_DIR = sly.app.get_data_dir()
task_type = (
    "object detection"  # you can choose "instance segmentation" or "pose estimation"
)

BRANCH = True
APP_VERSION = "auto-train-updates"

##################### PART 1: TRAINING #####################

module_id = api.app.get_ecosystem_module_id(APP_NAME)
module_info = api.app.get_ecosystem_module_info(module_id)
project_name = api.project.get_info_by_id(PROJECT_ID).name

sly.logger.info(f"Starting AutoTrain for application {module_info.name}")

params = module_info.get_arguments(images_project=PROJECT_ID)

session = api.app.start(
    agent_id=AGENT_ID,
    module_id=module_id,
    workspace_id=WORKSPACE_ID,
    description=f"AutoTrain session for {module_info.name}",
    task_name="AutoTrain/train",
    params=params,
    app_version=APP_VERSION,
    is_branch=BRANCH,
)

task_id = session.task_id
domain = sly.env.server_address()
token = api.task.get_info_by_id(task_id)["meta"]["sessionToken"]
post_shutdown = f"{domain}/net/{token}/sly/shutdown"

while not api.task.get_status(task_id) is api.task.Status.STARTED:
    sleep(GLOBAL_TIMEOUT)
else:
    sleep(10)  # still need a time after status changed

sly.logger.info(f"Session started: #{task_id}")

# read yaml file with training parameters
train_params_filepath = "scripts/auto_train_params.yaml"
with open(train_params_filepath, "r") as f:
    train_params = f.read()
# :green_book: You can set any parameters you want to customize training in the data field
api.task.send_request(
    task_id,
    "auto_train",
    data={
        "yaml_string": train_params,
    },  # :green_book: train paramaters
    timeout=10e6,
)

team_files_folder = Path("/yolov8_train") / task_type / project_name / str(task_id)
weights = Path(team_files_folder) / "weights"
best = None

while best is None:
    sleep(GLOBAL_TIMEOUT)
    if api.file.dir_exists(TEAM_ID, str(weights)):
        for filename in api.file.listdir(TEAM_ID, str(weights)):
            if os.path.basename(filename).startswith("best"):
                best = str(weights / filename)
                sly.logger.info(f"Checkpoint founded : {best}")

requests.post(post_shutdown)

sly.logger.info("Training completed")
sly.logger.info(
    f"The weights of trained model and other artifacts uploaded in Team Files: {str(team_files_folder)}"
)
