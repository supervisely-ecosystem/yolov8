# This module contains functions that are used to configure the input and output of the workflow for the current app,
# and versioning feature that creates a project version before the task starts.

import os

import supervisely as sly


def workflow_input(
    api: sly.Api,
    project_info: sly.ProjectInfo,
    file_info: sly.api.file_api.FileInfo = None,
):
    try:
        version_in_title = "v8 | v9 | v10"
        if file_info is not None:
            if "v8" in file_info.name:
                version_in_title = "v8"
            elif "v9" in file_info.name:
                version_in_title = "v9"
            elif "v10" in file_info.name:
                version_in_title = "v10"
        project_version_id = api.project.version.create(
            project_info,
            f"Train YOLO{version_in_title}",
            f"This backup was created automatically by Supervisely before the Train YOLO{version_in_title} task with ID: {api.task_id}",
        )
    except Exception as e:
        sly.logger.warning(f"Failed to create a project version: {repr(e)}")
        project_version_id = None

    try:
        if project_version_id is None:
            project_version_id = (
                project_info.version.get("id", None) if project_info.version else None
            )
        api.app.workflow.add_input_project(project_info.id, version_id=project_version_id)
        if file_info is not None:
            api.app.workflow.add_input_file(file_info, model_weight=True)
        sly.logger.debug(
            f"Workflow Input: Project ID - {project_info.id}, Project Version ID - {project_version_id}, Input File - {True if file_info else False}"
        )
    except Exception as e:
        sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")


def workflow_output(
    api: sly.Api,
    model_filename: str,
    team_files_dir: str,
    best_filename: str,
):
    try:
        weights_file_path_in_team_files_dir = os.path.join(team_files_dir, "weights", best_filename)
        best_filename_info = api.file.get_info_by_path(
            sly.env.team_id(), weights_file_path_in_team_files_dir
        )
        module_id = api.task.get_info_by_id(api.task_id).get("meta", {}).get("app", {}).get("id")
        sly.logger.debug(
            f"Workflow Output: Model filename - {model_filename}, Team Files dir - {team_files_dir}, Best filename - {best_filename}"
        )
        model_name = "Custom Model"
        if model_filename and "v8" in model_filename:
            model_name = "YOLOv8"
        elif model_filename and "v9" in model_filename:
            model_name = "YOLOv9"
        elif model_filename and "v10" in model_filename:
            model_name = "YOLOv10"
            
        if best_filename_info:
            node_settings = sly.WorkflowSettings(
                title=f"Train {model_name}",
                url=(
                    f"/apps/{module_id}/sessions/{api.task_id}"
                    if module_id
                    else f"apps/sessions/{api.task_id}"
                ),
                url_title="Show Results",
            )
            relation_settings = sly.WorkflowSettings(
                title="Checkpoints",
                icon="folder",
                icon_color="#FFA500",
                icon_bg_color="#FFE8BE",
                url=f"/files/{best_filename_info.id}/true",
                url_title="Open Folder",
            )
            meta = sly.WorkflowMeta(
                relation_settings=relation_settings, node_settings=node_settings
            )
            sly.logger.debug(f"Workflow Output: meta \n    {meta}")
            api.app.workflow.add_output_file(best_filename_info, model_weight=True, meta=meta)
        else:
            sly.logger.debug(
                f"File with checkpoints not found in Team Files. Cannot set workflow output."
            )
    except Exception as e:
        sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")
