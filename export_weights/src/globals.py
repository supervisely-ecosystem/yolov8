import os

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development():
    # * For convinient development, has no effect in the production.
    load_dotenv("export_weights/local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


class ModalState:
    """Modal state"""

    FORMAT = "modal.state.Format"

    @classmethod
    def format(cls):
        return os.getenv(cls.FORMAT, "onnx")


class State:
    """App state"""

    def __init__(self):
        self.selected_team = sly.env.team_id()
        self.selected_file = sly.env.team_files_file()
        self.format = ModalState.format()


LOCAL_WEIGHTS_DIR = "/sly_task_data/weights" if sly.is_production() else "debug_data/weights"
STATE = State()
Api = sly.Api()
