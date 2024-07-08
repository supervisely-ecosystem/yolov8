# Description: This file contains versioning features and the Workflow class that is used to add input and output to the workflow.

import supervisely as sly
import os


def check_compatibility(func):
    def wrapper(self, *args, **kwargs):
        if self.is_compatible is None:
            self.is_compatible = self.check_instance_ver_compatibility()
        if not self.is_compatible:
            return
        return func(self, *args, **kwargs)

    return wrapper


class Workflow:
    def __init__(self, api: sly.Api, min_instance_version: str = None):
        self.is_compatible = None
        self.api = api
        self._min_instance_version = (
            "6.9.31" if min_instance_version is None else min_instance_version
        )
    
    def check_instance_ver_compatibility(self):
        if self.api.instance_version < self._min_instance_version:
            sly.logger.info(
                f"Supervisely instance version does not support workflow and versioning features. To use them, please update your instance minimum to version {self._min_instance_version}."
            )
            return False
        return True
    
    @check_compatibility
    def add_input(self, project_info: sly.ProjectInfo, file_info = None):
        project_version_id = self.api.project.version.create(
            project_info, "Train YOLO (v8, v9)", f"This backup was created automatically by Supervisely before the Train YOLO task with ID: {self.api.task_id}"
        )
        if project_version_id is None:
            project_version_id = project_info.version.get("id", None) if project_info.version else None
        self.api.app.workflow.add_input_project(project_info.id, version_id=project_version_id)
        if file_info is not None:
             self.api.app.workflow.add_input_file(file_info, model_weight=True)

    @check_compatibility
    def add_output(self, model_filename: str, team_files_dir: str, best_filename: str):
        weights_file_path_in_team_files_dir = os.path.join(team_files_dir, "weights", best_filename)
        best_filename_info = self.api.file.get_info_by_path(sly.env.team_id(), weights_file_path_in_team_files_dir)
        module_id = self.api.task.get_info_by_id(self.api.task_id).get("meta", {}).get("app", {}).get("id")
        if model_filename and "v8" in model_filename:
            model_name = "YOLOv8"
        elif model_filename and "v9" in model_filename:
            model_name = "YOLOv9"
        else:
            model_name = "Custom Model"
        if best_filename_info:
            meta = {
                "customNodeSettings": {
                "title": f"<h4>Train {model_name}</h4>",
                "mainLink": {
                    "url": f"/apps/{module_id}/sessions/{self.api.task_id}" if module_id else f"apps/sessions/{self.api.task_id}",
                    "title": "Show Results"
                }
            },
            "customRelationSettings": {
                "icon": {
                    "icon": "zmdi-folder",
                    "color": "#FFA500",
                    "backgroundColor": "#FFE8BE"
                },
                "title": "<h4>Checkpoints</h4>",
                "mainLink": {"url": f"/files/{best_filename_info.id}/true", "title": "Open Folder"}
            }
        }
            self.api.app.workflow.add_output_file(best_filename_info, model_weight=True, meta=meta)
        else:
            sly.logger.error(f"File {best_filename_info.path} not found in team files. Cannot set workflow output.")

