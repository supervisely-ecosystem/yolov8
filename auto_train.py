import os
from pathlib import Path
from dotenv import load_dotenv
from time import sleep

import supervisely as sly

load_dotenv(os.path.expanduser("~/supervisely.env"))

GLOBAL_TIMEOUT = 1  # seconds
PROJECT_ID = 20739
TEAM_ID = 438
WORKSPACE_ID = 657
TASK_TYPE = "object detection"


DEBUG_SESSION = True
if DEBUG_SESSION:
    APP_VERSION = 'auto-train'
    BRANCH=True
else:
    APP_VERSION = None
    BRANCH=False


def train_model(api: sly.Api) -> Path:
    train_app_name = "supervisely-ecosystem/yolov8/train"

    module_id = api.app.get_ecosystem_module_id(train_app_name)
    module_info = api.app.get_ecosystem_module_info(module_id)
    # project_name = api.project.get_info_by_id(PROJECT_ID).name

    sly.logger.info(f"Starting AutoTrain for application {module_info.name}")

    # params = module_info.get_arguments(images_project=PROJECT_ID)

    # session = api.app.start(
    #     agent_id=AGENT_ID,
    #     module_id=module_id,
    #     workspace_id=WORKSPACE_ID,
    #     description=f"AutoTrain session for {module_info.name}",
    #     task_name="AutoTrain/train",
    #     params=params,
    #     app_version=APP_VERSION,
    #     is_branch=BRANCH,
    # )


    task_id = 38252

    # TODO: дождаться запуска
    sleep(10)
    sly.logger.info(f"Session started: #{task_id}")

    api.task.send_request(
        task_id,
        "auto_train",
        data={
            "project_id": PROJECT_ID,
            "task_type": TASK_TYPE,
        }
    )


    team_files_folder = Path("/yolov8_train") / TASK_TYPE / project_name / str(task_id)
    weights = Path(team_files_folder) / "weights"
    best_founded = False

    while not best_founded:
        sleep(GLOBAL_TIMEOUT)
        if api.file.dir_exists(TEAM_ID, str(weights)):
            for filename in api.file.listdir(TEAM_ID, str(best)):
                if filename.startswith("best"):
                    best_founded = True
                    best = weights / filename
                    sly.logger.info(
                        f"Checkpoint founded : {str(best)}"
                    )

    return best, task_id


def serve_model(api: sly.Api, best_weights: Path, session_id: int):
    serve_app_name = "supervisely-ecosystem/yolov8/serve"

    module_id = api.app.get_ecosystem_module_id(serve_app_name)
    module_info = api.app.get_ecosystem_module_info(module_id)
    project_name = api.project.get_info_by_id(PROJECT_ID).name

    sly.logger.info(f"Starting AutoTrain for application {module_info.name}")

    params = module_info.get_arguments()

    session = api.app.start(
        agent_id=AGENT_ID,
        module_id=module_id,
        workspace_id=WORKSPACE_ID,
        description=f"AutoTrain session for {module_info.name}",
        task_name="AutoTrain/serve",
        params=params,
    )

    # model_checkpoint_name = best_weights.name
    # model_checkpoints = Path(".") / "auto_train_weights" / str(session_id)

    # f"./auto_train_weights/{session_id}"
    # api.file.download(
    #     TEAM_ID,
    #     str(best_weights),
    #     str(model_checkpoints),
    # )
    
    # settings = {
    #     "conf": 0.25,
    #     "iou": 0.7,
    #     "half": False,
    #     "max_det": 300,
    #     "agnostic_nms": False,
    # }




if __name__ == "__main__":
    api = sly.Api()
    best_path, task_id = train_model(api)
    print(best_path, task_id)
    # serve_model(api, "", "")