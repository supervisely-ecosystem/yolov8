# Description: This file contains versioning features and the Workflow class that is used to add input and output to the workflow.

import supervisely as sly
import os
from supervisely.api.file_api import FileInfo


def check_compatibility(func):
    def wrapper(self, *args, **kwargs):
        if self.is_compatible is None:
            try:
                self.is_compatible = self.check_instance_ver_compatibility()
            except Exception as e:
                sly.logger.error(
                    "Can not check compatibility with Supervisely instance. "
                    f"Workflow and versioning features will be disabled. Error: {repr(e)}"
                )
                self.is_compatible = False
        if not self.is_compatible:
            return
        return func(self, *args, **kwargs)

    return wrapper


class Workflow:
    def __init__(self, api: sly.Api, min_instance_version: str = None):
        self.is_compatible = None
        self.api = api
        self._min_instance_version = "6.9.31" if min_instance_version is None else min_instance_version

    def check_instance_ver_compatibility(self):
        if not self.api.is_version_supported(self._min_instance_version):
            sly.logger.info(
                f"Supervisely instance version {self.api.instance_version} does not support workflow and versioning features."
            )
            if not sly.is_community():
                sly.logger.info(
                    f"To use them, please update your instance to version {self._min_instance_version} or higher."
                )
            return False
        return True

    @check_compatibility
    def add_input(self, project_info: sly.ProjectInfo, file_info=None):
        try:
            project_version_id = self.api.project.version.create(
                project_info,
                "Train YOLO (v8, v9)",
                f"This backup was created automatically by Supervisely before the Train YOLO task with ID: {self.api.task_id}",
            )
        except Exception as e:
            sly.logger.warning(f"Failed to create a project version: {repr(e)}")
            project_version_id = None

        try:
            if project_version_id is None:
                project_version_id = project_info.version.get("id", None) if project_info.version else None
            self.api.app.workflow.add_input_project(project_info.id, version_id=project_version_id)
            if file_info is not None:
                self.api.app.workflow.add_input_file(file_info, model_weight=True)
            sly.logger.debug(
                f"Workflow Input: Project ID - {project_info.id}, Project Version ID - {project_version_id}, Input File - {True if file_info else False}"
            )
        except Exception as e:
            sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")

    @check_compatibility
    def add_output(
        self,
        model_filename: str,
        team_files_dir: str,
        best_filename: str,
        template_vis_file: FileInfo,
    ):
        try:
            weights_file_path_in_team_files_dir = os.path.join(team_files_dir, "weights", best_filename)
            best_filename_info = self.api.file.get_info_by_path(sly.env.team_id(), weights_file_path_in_team_files_dir)
            module_id = self.api.task.get_info_by_id(self.api.task_id).get("meta", {}).get("app", {}).get("id")
            sly.logger.debug(
                f"Workflow Output: Model filename - {model_filename}, Team Files dir - {team_files_dir}, Best filename - {best_filename}"
            )
            if model_filename and "v8" in model_filename:
                model_name = "YOLOv8"
            elif model_filename and "v9" in model_filename:
                model_name = "YOLOv9"
            else:
                model_name = "Custom Model"
            train_app_node = {
                "title": f"<h4>Train {model_name}</h4>",
                "mainLink": {
                    "url": (
                        f"/apps/{module_id}/sessions/{self.api.task_id}"
                        if module_id
                        else f"apps/sessions/{self.api.task_id}"
                    ),
                    "title": "Show Results",
                },
            }
            if best_filename_info:
                meta = {
                    "customNodeSettings": train_app_node,
                    "customRelationSettings": {
                        "icon": {
                            "icon": "zmdi-folder",
                            "color": "#FFA500",
                            "backgroundColor": "#FFE8BE",
                        },
                        "title": "<h4>Checkpoints</h4>",
                        "mainLink": {
                            "url": f"/files/{best_filename_info.id}/true",
                            "title": "Open Folder",
                        },
                    },
                }
                sly.logger.debug(f"Workflow Output: meta \n    {meta}")
                self.api.app.workflow.add_output_file(best_filename_info, model_weight=True, meta=meta)

                meta = {
                    "customNodeSettings": train_app_node,
                    "customRelationSettings": {
                        "icon": {
                            "icon": "zmdi-check-circle-u",
                            "color": "#DEL389",
                            "backgroundColor": "#FFC1CC",
                        },
                        "title": "<h4>Model Benchmark</h4>",
                        "mainLink": {
                            "url": f"/model-benchmark?id={template_vis_file.id}",
                            "title": "Open Report",
                        },
                    },
                }
                self.api.app.workflow.add_output_file(template_vis_file, meta=meta)

            else:
                sly.logger.debug(
                    f"File {weights_file_path_in_team_files_dir} not found in Team Files. Cannot set workflow output."
                )
        except Exception as e:
            sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")
