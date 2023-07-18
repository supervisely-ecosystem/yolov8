import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from time import sleep
import supervisely as sly

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")

GLOBAL_TIMEOUT = 1  # seconds
AGENT_ID = 230  # A5000
PROJECT_ID = sly.env.project_id()
TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()
TASK_TYPE = "object detection"


DEBUG_SESSION = True
if DEBUG_SESSION:
    APP_VERSION = "auto-train"
    BRANCH=True
else:
    APP_VERSION = None
    BRANCH=False


def train_model(api: sly.Api) -> Path:
    train_app_name = "supervisely-ecosystem/yolov8/train"

    module_id = api.app.get_ecosystem_module_id(train_app_name)
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
    
    # task_id = 38357
    task_id = session.task_id
    domain = sly.env.server_address()
    token = api.task.get_info_by_id(task_id)['meta']['sessionToken']
    post_shutdown = f"{domain}/net/{token}/sly/shutdown"

    while not api.task.get_status(task_id) is api.task.Status.STARTED:
        sleep(GLOBAL_TIMEOUT)
    else:
        sleep(10) # still need a time after status changed 

    sly.logger.info(f"Session started: #{task_id}")

    api.task.send_request(
        task_id,
        "auto_train",
        data={
            "project_id": PROJECT_ID,
            "dataset_ids": [64611],
            "task_type": TASK_TYPE,
            "model": "YOLOv8n-det",
            "train_mode": "finetune", # finetune / scratch
            "n_epochs": 100,
            "patience": 50,
            "batch_size": 16,
            "input_image_size": 640,
            "optimizer": "AdamW", # AdamW, Adam, SGD, RMSProp
            "n_workers": 8,
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "amp": True,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        },
        timeout=10e6,
    )

    team_files_folder = Path("/yolov8_train") / TASK_TYPE / project_name / str(task_id)
    weights = Path(team_files_folder) / "weights"
    best_founded = False

    while not best_founded:
        sleep(GLOBAL_TIMEOUT)
        if api.file.dir_exists(TEAM_ID, str(weights)):
            for filename in api.file.listdir(TEAM_ID, str(weights)):
                if os.path.basename(filename).startswith("best"):
                    best_founded = True
                    best = weights / filename
                    sly.logger.info(
                        f"Checkpoint founded : {str(best)}"
                    )

    requests.post(post_shutdown)

    return team_files_folder


if __name__ == "__main__":
    api = sly.Api()
    result_folder = train_model(api)
    sly.logger.info("Training completed")
    sly.logger.info("The weights of trained model, predictions visualization and other training artifacts can be found in the following Team Files folder:")
    sly.logger.info(result_folder)