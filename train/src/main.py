import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import numpy as np
import yaml
import random
import supervisely as sly
import supervisely.io.env as env
import src.globals as g
from dotenv import load_dotenv
import yaml
from supervisely.app.widgets import (
    Container,
    Card,
    SelectString,
    InputNumber,
    Input,
    Button,
    Field,
    Progress,
    SelectDataset,
    ClassesTable,
    DoneLabel,
    Editor,
    Checkbox,
    RadioTabs,
    RadioTable,
    RadioGroup,
    NotificationBox,
    FileThumbnail,
    GridPlot,
    FolderThumbnail,
    TrainValSplits,
    Flexbox,
    Image,
    GridGallery,
    TaskLogs,
    Stepper,
    Text,
    Collapse,
    ImageSlider,
    Dialog,
)
from src.utils import verify_train_val_sets
from src.sly_to_yolov8 import check_bbox_exist_on_images, transform
from src.callbacks import on_train_batch_end
from src.dataset_cache import download_project
from ultralytics import YOLO
import torch
from src.metrics_watcher import Watcher
import threading
import pandas as pd
from functools import partial
from urllib.request import urlopen
import math
import ruamel.yaml
from fastapi import Response, Request


# function for updating global variables
def update_globals(new_dataset_ids):
    global dataset_ids, project_id, workspace_id, project_info, project_meta
    dataset_ids = new_dataset_ids
    if dataset_ids and all(ds_id is not None for ds_id in dataset_ids):
        project_id = api.dataset.get_info_by_id(dataset_ids[0]).project_id
        workspace_id = api.project.get_info_by_id(project_id).workspace_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        print(f"Project is {project_info.name}, {dataset_ids}")
    elif project_id:
        workspace_id = api.project.get_info_by_id(project_id).workspace_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    else:
        print("All globals set to None")
        dataset_ids = []
        project_id, workspace_id, project_info, project_meta = [None] * 4


# authentication
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()
team_id = sly.env.team_id()

# if app had started from context menu, one of this has to be set:
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)
dataset_ids = [dataset_id] if dataset_id else []
update_globals(dataset_ids)

sly.logger.info(f"App root directory: {g.app_root_directory}")


### 1. Dataset selection
dataset_selector = SelectDataset(project_id=project_id, multiselect=True, select_all_datasets=True)
use_cache_text = Text("Use cached data stored on the agent to optimize project downlaod")
use_cache_checkbox = Checkbox(use_cache_text, checked=True)
select_data_button = Button("Select data")
select_done = DoneLabel("Successfully selected input data")
select_done.hide()
reselect_data_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>Reselect data',
    button_type="warning",
    button_size="small",
    plain=True,
)
reselect_data_button.hide()
project_settings_content = Container(
    [
        dataset_selector,
        use_cache_checkbox,
        select_data_button,
        select_done,
        reselect_data_button,
    ]
)
card_project_settings = Card(title="Dataset selection", content=project_settings_content)


### 2. Project classes
task_type_items = [
    RadioGroup.Item(value="object detection"),
    RadioGroup.Item(value="instance segmentation"),
    RadioGroup.Item(value="pose estimation"),
]
task_type_select = RadioGroup(items=task_type_items, direction="vertical")
task_type_select_f = Field(
    content=task_type_select,
    title="Task type",
)
classes_table = ClassesTable()
select_classes_button = Button("select classes")
select_classes_button.hide()
select_other_classes_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>select other classes',
    button_type="warning",
    button_size="small",
    plain=True,
)
select_other_classes_button.hide()
classes_done = DoneLabel()
classes_done.hide()
classes_content = Container(
    [
        task_type_select_f,
        classes_table,
        select_classes_button,
        select_other_classes_button,
        classes_done,
    ]
)
card_classes = Card(
    title="Task type & training classes",
    description=(
        "Select task type and classes, that should be used for training. "
        "Supported shapes include rectangle, bitmap, polygon and graph"
    ),
    content=classes_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_classes.collapse()
card_classes.lock()


### 3.1 Train / validation split
train_val_split = TrainValSplits(project_id=project_id)
unlabeled_images_select = SelectString(values=["keep unlabeled images", "ignore unlabeled images"])
unlabeled_images_select_f = Field(
    content=unlabeled_images_select,
    title="What to do with unlabeled images",
    description=(
        "Sometimes unlabeled images can be used to reduce noise in predictions, "
        "sometimes it is a mistake in training data"
    ),
)
split_data_button = Button("Split data")
resplit_data_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>Re-split data',
    button_type="warning",
    button_size="small",
    plain=True,
)
resplit_data_button.hide()
split_done = DoneLabel("Data was successfully splitted")
split_done.hide()

### 3.2 Check if there are images without bounding boxes (for pose estimation)
bbox_miss_gallery = ImageSlider(previews=[], height=100, selectable=False)
bbox_miss_collapse_item = Collapse.Item(
    name="show_images",
    title="Following images have no bounding boxes for graphs:",
    content=bbox_miss_gallery,
)
bbox_miss_collapse = Collapse(items=[bbox_miss_collapse_item])
bbox_miss_collapse.hide()
bbox_miss_text = Text("Select options to handle them:")
bbox_miss_manual_checkbox = Checkbox(
    "Stop processing (I will add bounding boxes manually)", checked=True
)
bbox_miss_manual_checkbox.disable()
bbox_miss_auto_checkbox = Checkbox(
    "Continue processing (bounding boxes will be created automatically)"
)
bbox_miss_btn = Button("OK")
bbox_miss_content = Container(
    widgets=[
        bbox_miss_collapse,
        bbox_miss_text,
        bbox_miss_manual_checkbox,
        bbox_miss_auto_checkbox,
        bbox_miss_btn,
    ]
)
bbox_miss_dialog = Dialog(title="Images with no bounding boxes", content=bbox_miss_content)
bbox_miss_check_progress = Progress()
train_val_content = Container(
    [
        train_val_split,
        unlabeled_images_select_f,
        split_data_button,
        resplit_data_button,
        split_done,
        bbox_miss_dialog,
        bbox_miss_check_progress,
        bbox_miss_collapse
    ]
)
card_train_val_split = Card(
    title="Train / validation split",
    description="Define how to split your data into train / val subsets",
    content=train_val_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_train_val_split.collapse()
card_train_val_split.lock()


### 4. Model selection
models_table_notification = NotificationBox(
    title="List of models in the table below depends on selected task type",
    description=(
        "If you want to see list of available models for another computer vision task, "
        "please, go back to task type & training classes step and change task type"
    ),
)
model_tabs_titles = ["Pretrained models", "Custom models"]
model_tabs_descriptions = [
    "Models trained outside Supervisely",
    "Models trained in Supervsely and located in Team Files",
]
models_table_columns = [key for key in g.det_models_data[0].keys()]
models_table_rows = []
for element in g.det_models_data:
    models_table_rows.append(list(element.values()))
models_table = RadioTable(
    columns=models_table_columns,
    rows=models_table_rows,
)
models_table_content = Container([models_table_notification, models_table])
team_files_url = f"{env.server_address()}/files/"
team_files_button = Button(
    text="Open Team Files",
    button_type="info",
    plain=True,
    icon="zmdi zmdi-folder",
    link=team_files_url,
)
model_path_input = Input(placeholder=f"Path to model file in Team Files")
model_path_input_f = Field(
    model_path_input,
    title=f"Copy path to model file from Team Files and paste to field below",
    description="Copy path in Team Files",
)
model_file_thumbnail = FileThumbnail()
custom_tab_content = Container(
    [
        team_files_button,
        model_path_input_f,
        model_file_thumbnail,
    ]
)
model_tabs_contents = [models_table_content, custom_tab_content]
model_tabs = RadioTabs(
    titles=model_tabs_titles,
    contents=model_tabs_contents,
    descriptions=model_tabs_descriptions,
)
select_model_button = Button("Select model")
reselect_model_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>Reselect model',
    button_type="warning",
    button_size="small",
    plain=True,
)
reselect_model_button.hide()
model_not_found_text = Text("Custom model not found", status="error")
model_not_found_text.hide()
model_select_done = DoneLabel("Model was successfully selected")
model_select_done.hide()
model_selection_content = Container(
    [
        model_tabs,
        select_model_button,
        reselect_model_button,
        model_not_found_text,
        model_select_done,
    ]
)
card_model_selection = Card(
    title="Model settings",
    description="Choose model size or how weights should be initialized",
    content=model_selection_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_model_selection.collapse()
card_model_selection.lock()


### 5. Training hyperparameters
select_train_mode = SelectString(values=["Finetune mode", "Scratch mode"])
select_train_mode_f = Field(
    content=select_train_mode,
    title="Select training mode",
    description=(
        "Finetune mode - .pt file with pretrained model weights will be downloaded, "
        "Scratch mode - model weights will be initialized randomly"
    ),
)
n_epochs_input = InputNumber(value=100, min=1)
n_epochs_input_f = Field(content=n_epochs_input, title="Number of epochs")
patience_input = InputNumber(value=50, min=1)
patience_input_f = Field(
    content=patience_input,
    title="Patience",
    description="Number of epochs to wait for no observable improvement for early stopping of training",
)
batch_size_input = InputNumber(value=16, min=1)
batch_size_input_f = Field(content=batch_size_input, title="Batch size")
image_size_input = InputNumber(value=640, step=10)
image_size_input_f = Field(content=image_size_input, title="Input image size")
select_optimizer = SelectString(values=["AdamW", "Adam", "SGD", "RMSProp"])
select_optimizer_f = Field(content=select_optimizer, title="Optimizer")
save_best = Checkbox(content="save best checkpoint", checked=True)
save_best.disable()
save_last = Checkbox(content="save last checkpoint", checked=True)
save_last.disable()
save_checkpoints_content = Flexbox(
    widgets=[save_best, save_last],
    center_content=False,
)
n_workers_input = InputNumber(value=8, min=1)
n_workers_input_f = Field(
    content=n_workers_input,
    title="Number of workers",
    description="Number of worker threads for data loading",
)
additional_config_items = [
    RadioGroup.Item(value="custom"),
    RadioGroup.Item(value="import template from Team Files"),
]
additional_config_radio = RadioGroup(additional_config_items, direction="horizontal")
additional_config_radio_f = Field(
    content=additional_config_radio,
    title="Define way of passing additional parameters",
    description="Create custom config or import template from Team Files",
)
additional_config_template_select = SelectString(values=["No data"])
additional_config_template_select_f = Field(
    content=additional_config_template_select,
    title="Select template",
)
additional_config_template_select_f.hide()
no_templates_notification = NotificationBox(
    title="No templates found",
    description=(
        "There are no templates for this task type in Team Files. "
        "You can create custom config and save it as a template to "
        "Team Files - you will be able to reuse it in your future experiments"
    ),
)
no_templates_notification.hide()
train_settings_editor = Editor(language_mode="yaml", height_lines=50)
with open(g.train_params_filepath, "r") as f:
    train_params = f.read()
train_settings_editor.set_text(train_params)
train_settings_editor_f = Field(
    content=train_settings_editor,
    title="Additional configuration",
    description="Tune learning rate, augmentations and other parameters",
)
save_train_params_button = Button("Save training hyperparameters")
save_template_button = Button(
    text="Save template to Team Files",
    icon="zmdi zmdi-cloud-upload",
)
save_params_flexbox = Flexbox(
    widgets=[save_train_params_button, save_template_button],
    gap=20,
)
reselect_train_params_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>Change training hyperparameters',
    button_type="warning",
    button_size="small",
    plain=True,
)
reselect_train_params_button.hide()
train_params_done = DoneLabel("Successfully saved training hyperparameters")
train_params_done.hide()
save_template_done = DoneLabel("Successfully uploaded template to Team Files")
save_template_done.hide()
train_params_content = Container(
    [
        select_train_mode_f,
        n_epochs_input_f,
        patience_input_f,
        batch_size_input_f,
        image_size_input_f,
        select_optimizer_f,
        save_checkpoints_content,
        n_workers_input_f,
        additional_config_radio_f,
        additional_config_template_select_f,
        no_templates_notification,
        train_settings_editor_f,
        save_params_flexbox,
        reselect_train_params_button,
        train_params_done,
        save_template_done,
    ]
)
card_train_params = Card(
    title="Training hyperparameters",
    description="Define general settings and advanced configuration",
    content=train_params_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_train_params.collapse()
card_train_params.lock()


### 6. Training progress
start_training_button = Button("start training")
logs_button = Button(
    text="Show logs",
    plain=True,
    button_size="mini",
    icon="zmdi zmdi-caret-down-circle",
)
task_logs = TaskLogs(task_id=g.app_session_id)
task_logs.hide()
progress_bar_download_project = Progress()
progress_bar_convert_to_yolo = Progress()
progress_bar_download_model = Progress()
progress_bar_epochs = Progress()
progress_bar_iters = Progress(hide_on_finish=False)
plot_titles = ["train", "val", "precision & recall"]
grid_plot = GridPlot(data=plot_titles, columns=3, gap=20)
grid_plot_f = Field(grid_plot, "Training and validation metrics")
grid_plot_f.hide()
plot_notification = NotificationBox(
    title="Some metrics can have unserializable values",
    description="During training process model performance metrics can have NaN / Inf values on some epochs and may not be displayed on the plots",
)
plot_notification.hide()
train_batches_gallery = GridGallery(
    columns_number=3,
    show_opacity_slider=False,
)
train_batches_gallery_f = Field(train_batches_gallery, "Train batches visualization")
train_batches_gallery_f.hide()
val_batches_gallery = GridGallery(
    columns_number=2,
    show_opacity_slider=False,
    enable_zoom=True,
    sync_views=True,
)
val_batches_gallery_f = Field(val_batches_gallery, "Model predictions visualization")
val_batches_gallery_f.hide()
additional_gallery = GridGallery(
    columns_number=3,
    show_opacity_slider=False,
    enable_zoom=True,
)
additional_gallery_f = Field(additional_gallery, "Additional training results visualization")
additional_gallery_f.hide()
progress_bar_upload_artifacts = Progress()
train_done = DoneLabel("Training completed. Training artifacts were uploaded to Team Files")
train_done.hide()
train_progress_content = Container(
    [
        start_training_button,
        logs_button,
        task_logs,
        progress_bar_download_project,
        progress_bar_convert_to_yolo,
        progress_bar_download_model,
        progress_bar_epochs,
        progress_bar_iters,
        grid_plot_f,
        plot_notification,
        train_batches_gallery_f,
        val_batches_gallery_f,
        additional_gallery_f,
        progress_bar_upload_artifacts,
        train_done,
    ]
)
card_train_progress = Card(
    title="Training progress",
    description="Track progress, detailed logs, metrics charts and other visualizations",
    content=train_progress_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_train_progress.collapse()
card_train_progress.lock()


### 7. Training artifacts
train_artifacts_folder = FolderThumbnail()
card_train_artifacts = Card(
    title="Training artifacts",
    description="Checkpoints, logs and other visualizations",
    content=train_artifacts_folder,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_train_artifacts.collapse()
card_train_artifacts.lock()


stepper = Stepper(
    widgets=[
        card_project_settings,
        card_classes,
        card_train_val_split,
        card_model_selection,
        card_train_params,
        card_train_progress,
        card_train_artifacts,
    ]
)


app = sly.Application(
    layout=Container(
        widgets=[
            stepper,
        ]
    ),
)
server = app.get_server()


@dataset_selector.value_changed
def on_dataset_selected(new_dataset_ids):
    if new_dataset_ids == []:
        select_data_button.hide()
    elif new_dataset_ids != [] and reselect_data_button.is_hidden():
        select_data_button.show()
    update_globals(new_dataset_ids)
    if sly.project.download.is_cached(project_id):
        use_cache_text.text = "Use cached data stored on the agent to optimize project downlaod"
    else:
        use_cache_text.text = "Cache data on the agent to optimize project download for future trainings"


@select_data_button.click
def select_input_data():
    project_shapes = [cls.geometry_type.geometry_name() for cls in project_meta.obj_classes]
    if "bitmap" in project_shapes or "polygon" in project_shapes:
        task_type_select.set_value("instance segmentation")
        models_table_columns = [key for key in g.seg_models_data[0].keys()]
        models_table_subtitles = [None] * len(models_table_columns)
        models_table_rows = []
        for element in g.seg_models_data:
            models_table_rows.append(list(element.values()))
        models_table.set_data(
            columns=models_table_columns,
            rows=models_table_rows,
            subtitles=models_table_subtitles,
        )
    elif "graph" in project_shapes:
        task_type_select.set_value("pose estimation")
        models_table_columns = [key for key in g.pose_models_data[0].keys()]
        models_table_subtitles = [None] * len(models_table_columns)
        models_table_rows = []
        for element in g.pose_models_data:
            models_table_rows.append(list(element.values()))
        models_table.set_data(
            columns=models_table_columns,
            rows=models_table_rows,
            subtitles=models_table_subtitles,
        )
    select_data_button.loading = True
    dataset_selector.disable()
    use_cache_text.disable()
    classes_table.read_project_from_id(project_id)
    classes_table.select_all()
    selected_classes = classes_table.get_selected_classes()
    _update_select_classes_button(selected_classes)
    select_data_button.loading = False
    select_data_button.hide()
    select_done.show()
    reselect_data_button.show()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_classes.unlock()
    card_classes.uncollapse()


@reselect_data_button.click
def reselect_input_data():
    select_data_button.show()
    reselect_data_button.hide()
    select_done.hide()
    dataset_selector.enable()
    use_cache_text.enable()
    curr_step = stepper.get_active_step()
    curr_step -= 1
    stepper.set_active_step(curr_step)


def _update_select_classes_button(selected_classes):
    n_classes = len(selected_classes)
    if n_classes > 0:
        if n_classes > 1:
            select_classes_button.text = f"Select {n_classes} classes"
        else:
            select_classes_button.text = f"Select {n_classes} class"
        select_classes_button.show()
    else:
        select_classes_button.hide()

@classes_table.value_changed
def on_classes_selected(selected_classes):
    _update_select_classes_button(selected_classes)


@task_type_select.value_changed
def select_task(task_type):
    project_shapes = [cls.geometry_type.geometry_name() for cls in project_meta.obj_classes]
    if task_type == "object detection":
        select_classes_button.enable()
        models_table_columns = [key for key in g.det_models_data[0].keys()]
        models_table_subtitles = [None] * len(models_table_columns)
        models_table_rows = []
        for element in g.det_models_data:
            models_table_rows.append(list(element.values()))
        models_table.set_data(
            columns=models_table_columns,
            rows=models_table_rows,
            subtitles=models_table_subtitles,
        )
    elif task_type == "instance segmentation":
        if "bitmap" not in project_shapes and "polygon" not in project_shapes:
            sly.app.show_dialog(
                title="There are no classes of shape mask (bitmap / polygon) in selected project",
                description="Please, change task type or select another project with classes of shape bitmap / polygon",
                status="warning",
            )
            select_classes_button.disable()
        else:
            select_classes_button.enable()
            models_table_columns = [key for key in g.seg_models_data[0].keys()]
            models_table_subtitles = [None] * len(models_table_columns)
            models_table_rows = []
            for element in g.seg_models_data:
                models_table_rows.append(list(element.values()))
            models_table.set_data(
                columns=models_table_columns,
                rows=models_table_rows,
                subtitles=models_table_subtitles,
            )
    elif task_type == "pose estimation":
        if "graph" not in project_shapes:
            sly.app.show_dialog(
                title="There are no classes of shape keypoints (graph) in selected project",
                description="Please, change task type or select another project with classes of shape graph",
                status="warning",
            )
            select_classes_button.disable()
        elif "rectangle" not in project_shapes:
            sly.app.show_dialog(
                title="There are no classes of shape rectangle in selected project (bounding boxes are required for pose estimation)",
                description="Please, change task type or select another project with classes of shape rectangle",
                status="warning",
            )
            select_classes_button.disable()
        else:
            select_classes_button.enable()
            models_table_columns = [key for key in g.pose_models_data[0].keys()]
            models_table_subtitles = [None] * len(models_table_columns)
            models_table_rows = []
            for element in g.pose_models_data:
                models_table_rows.append(list(element.values()))
            models_table.set_data(
                columns=models_table_columns,
                rows=models_table_rows,
                subtitles=models_table_subtitles,
            )


@select_classes_button.click
def select_classes():
    selected_classes = classes_table.get_selected_classes()
    selected_shapes = [
        cls.geometry_type.geometry_name()
        for cls in project_meta.obj_classes
        if cls.name in selected_classes
    ]
    task_type = task_type_select.get_value()
    if task_type == "pose estimation" and (
        "graph" not in selected_shapes or "rectangle" not in selected_shapes
    ):
        sly.app.show_dialog(
            title="Pose estimation task requires input project to have at least one class of shape graph and one class of shape rectangle",
            description="Please, select both classes of shape rectangle and graph or change task type",
            status="warning",
        )
    else:
        n_classes = len(classes_table.get_selected_classes())
        if n_classes > 1:
            classes_done.text = f"{n_classes} classes were selected successfully"
        else:
            classes_done.text = f"{n_classes} class was selected successfully"
        select_classes_button.hide()
        classes_done.show()
        select_other_classes_button.show()
        classes_table.disable()
        task_type_select.disable()
        curr_step = stepper.get_active_step()
        curr_step += 1
        stepper.set_active_step(curr_step)
        card_train_val_split.unlock()
        card_train_val_split.uncollapse()


@select_other_classes_button.click
def select_other_classes():
    classes_table.enable()
    task_type_select.enable()
    select_other_classes_button.hide()
    classes_done.hide()
    select_classes_button.show()
    curr_step = stepper.get_active_step()
    curr_step -= 1
    stepper.set_active_step(curr_step)


@split_data_button.click
def split_data():
    split_data_button.loading = True
    train_val_split.disable()
    unlabeled_images_select.disable()
    split_done.show()
    task_type = task_type_select.get_value()
    if task_type == "pose estimation":
        selected_classes = classes_table.get_selected_classes()
        image_urls = check_bbox_exist_on_images(
            api,selected_classes, dataset_ids, project_meta, bbox_miss_check_progress
        )
        if len(image_urls) > 0:
            bbox_miss_gallery.set_data(previews=image_urls)
            bbox_miss_dialog.show()
            bbox_miss_collapse.show()
    split_data_button.loading = False
    split_data_button.hide()

    resplit_data_button.show()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_model_selection.unlock()
    card_model_selection.uncollapse()


@resplit_data_button.click
def resplit_data():
    train_val_split.enable()
    unlabeled_images_select.enable()
    split_data_button.show()
    split_done.hide()
    resplit_data_button.hide()
    curr_step = stepper.get_active_step()
    curr_step -= 1
    stepper.set_active_step(curr_step)


@bbox_miss_auto_checkbox.value_changed
def on_auto_change(value):
    if value:
        bbox_miss_manual_checkbox.uncheck()
    else:
        bbox_miss_manual_checkbox.check()


@bbox_miss_btn.click
def close_dialog():
    bbox_miss_dialog.hide()
    if bbox_miss_manual_checkbox.is_checked():
        bbox_miss_collapse.set_active_panel(bbox_miss_collapse_item.name)
        msg = (
            "Application will be stopped (corresponding option is selected). "
            "Add bounding boxes to images with no bounding boxes and restart the app"
        )
        sly.app.show_dialog("Warning", msg, status="warning")
        app.stop()


@model_tabs.value_changed
def model_tab_changed(value):
    if value == "Pretrained models":
        model_not_found_text.hide()
        model_select_done.hide()


@select_model_button.click
def select_model():
    weights_type = model_tabs.get_active_tab()
    file_exists = True
    if weights_type == "Custom models":
        custom_link = model_path_input.get_value()
        if custom_link != "":
            file_exists = api.file.exists(sly.env.team_id(), custom_link)
        else:
            file_exists = False
    if not file_exists and weights_type == "Custom models":
        model_not_found_text.show()
        model_select_done.hide()
    else:
        model_select_done.show()
        model_not_found_text.hide()
        select_model_button.hide()
        model_tabs.disable()
        models_table.disable()
        model_path_input.disable()
        reselect_model_button.show()
        curr_step = stepper.get_active_step()
        curr_step += 1
        stepper.set_active_step(curr_step)
        card_train_params.unlock()
        card_train_params.uncollapse()


@reselect_model_button.click
def reselect_model():
    select_model_button.show()
    model_not_found_text.hide()
    model_select_done.hide()
    model_tabs.enable()
    models_table.enable()
    model_path_input.enable()
    reselect_model_button.hide()
    curr_step = stepper.get_active_step()
    curr_step -= 1
    stepper.set_active_step(curr_step)


@model_path_input.value_changed
def change_file_preview(value):
    file_info = None
    if value != "":
        file_info = api.file.get_info_by_path(sly.env.team_id(), value)
    if file_info is None:
        model_not_found_text.show()
        model_select_done.hide()
        model_file_thumbnail.set(None)
    else:
        model_not_found_text.hide()
        model_file_thumbnail.set(file_info)


@additional_config_radio.value_changed
def change_radio(value):
    if value == "import template from Team Files":
        remote_templates_dir = os.path.join(
            "/yolov8_train", task_type_select.get_value(), "param_templates"
        )
        templates = api.file.list(team_id, remote_templates_dir)
        if len(templates) == 0:
            no_templates_notification.show()
        else:
            template_names = [template["name"] for template in templates]
            additional_config_template_select.set(template_names)
            additional_config_template_select_f.show()
    else:
        additional_config_template_select_f.hide()
        no_templates_notification.hide()


@additional_config_template_select.value_changed
def change_template(template):
    remote_templates_dir = os.path.join(
        "/yolov8_train", task_type_select.get_value(), "param_templates"
    )
    remote_template_path = os.path.join(remote_templates_dir, template)
    local_template_path = os.path.join(g.app_data_dir, template)
    api.file.download(team_id, remote_template_path, local_template_path)
    with open(local_template_path, "r") as f:
        train_params = f.read()
    train_settings_editor.set_text(train_params)


@save_template_button.click
def upload_template():
    save_template_button.loading = True
    remote_templates_dir = os.path.join(
        "/yolov8_train", task_type_select.get_value(), "param_templates"
    )
    additional_params = train_settings_editor.get_text()
    ryaml = ruamel.yaml.YAML()
    additional_params = ryaml.load(additional_params)
    # additional_params = yaml.safe_load(additional_params)
    filename = project_info.name.replace(" ", "_") + "_param_template.yml"
    with open(filename, "w") as outfile:
        # yaml.dump(additional_params, outfile, default_flow_style=False)
        ryaml.dump(additional_params, outfile)
    remote_filepath = os.path.join(remote_templates_dir, filename)
    api.file.upload(team_id, filename, api.file.get_free_name(team_id, remote_filepath))
    sly.fs.silent_remove(filename)
    save_template_button.loading = False
    save_template_button.hide()
    save_template_done.show()


@save_train_params_button.click
def save_train_params():
    save_train_params_button.hide()
    train_params_done.show()
    reselect_train_params_button.show()
    select_train_mode.disable()
    n_epochs_input.disable()
    patience_input.disable()
    batch_size_input.disable()
    image_size_input.disable()
    select_optimizer.disable()
    n_workers_input.disable()
    train_settings_editor.readonly = True
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_train_progress.unlock()
    card_train_progress.uncollapse()


@reselect_train_params_button.click
def change_train_params():
    save_train_params_button.show()
    train_params_done.hide()
    reselect_train_params_button.hide()
    select_train_mode.enable()
    n_epochs_input.enable()
    patience_input.enable()
    batch_size_input.enable()
    image_size_input.enable()
    select_optimizer.enable()
    n_workers_input.enable()
    train_settings_editor.readonly = False
    save_template_button.show()
    save_template_done.hide()
    curr_step = stepper.get_active_step()
    curr_step -= 1
    stepper.set_active_step(curr_step)


@logs_button.click
def change_logs_visibility():
    if task_logs.is_hidden():
        task_logs.show()
        logs_button.text = "Hide logs"
        logs_button.icon = "zmdi zmdi-caret-up-circle"
    else:
        task_logs.hide()
        logs_button.text = "Show logs"
        logs_button.icon = "zmdi zmdi-caret-down-circle"


@start_training_button.click
def start_training():
    task_type = task_type_select.get_value()

    local_dir = g.root_model_checkpoint_dir
    if task_type == "object detection":
        necessary_geometries = ["rectangle"]
        checkpoint_dir = os.path.join(local_dir, "detect")
        local_artifacts_dir = os.path.join(local_dir, "detect", "train")
    elif task_type == "pose estimation":
        necessary_geometries = ["graph", "rectangle"]
        checkpoint_dir = os.path.join(local_dir, "pose")
        local_artifacts_dir = os.path.join(local_dir, "pose", "train")
    elif task_type == "instance segmentation":
        necessary_geometries = ["bitmap", "polygon"]
        checkpoint_dir = os.path.join(local_dir, "segment")
        local_artifacts_dir = os.path.join(local_dir, "segment", "train")

    sly.logger.info(f"Local artifacts dir: {local_artifacts_dir}")

    if os.path.exists(local_artifacts_dir):
        sly.fs.remove_dir(local_artifacts_dir)
    start_training_button.loading = True
    # get number of images in selected datasets
    dataset_infos = [api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids]
    n_images = sum([info.images_count for info in dataset_infos])
    # download dataset
    download_project(
        api=api,
        project_info=project_info,
        dataset_infos=dataset_infos,
        use_cache=use_cache_checkbox.is_checked(),
        progress=progress_bar_download_project
    )
    # remove unselected classes
    selected_classes = classes_table.get_selected_classes()
    sly.Project.remove_classes_except(g.project_dir, classes_to_keep=selected_classes, inplace=True)
    # remove classes with unnecessary shapes
    if task_type != "object detection":
        unnecessary_classes = []
        for cls in project_meta.obj_classes:
            if (
                cls.name in selected_classes
                and cls.geometry_type.geometry_name() not in necessary_geometries
            ):
                unnecessary_classes.append(cls.name)
        if len(unnecessary_classes) > 0:
            sly.Project.remove_classes(
                g.project_dir, classes_to_remove=unnecessary_classes, inplace=True
            )
    # transfer project to detection task if necessary
    if task_type == "object detection":
        sly.Project.to_detection_task(g.project_dir, inplace=True)
    # remove unlabeled images if such option was selected by user
    if unlabeled_images_select.get_value() == "ignore unlabeled images":
        n_images_before = n_images
        sly.Project.remove_items_without_objects(g.project_dir, inplace=True)
        project = sly.Project(g.project_dir, sly.OpenMode.READ)
        n_images_after = project.total_items
        if n_images_before != n_images_after:
            train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
            train_set, val_set = train_val_split.get_splits()
            val_part = len(val_set) / (len(train_set) + len(val_set))
            new_val_count = round(n_images_after * val_part)
            if new_val_count < 1:
                sly.app.show_dialog(
                    title="An error occurted",
                    description="Val split length is 0 after ignoring images. Please check your data",
                    status="error",
                )
                raise ValueError(
                    "Val split length is 0 after ignoring images. Please check your data"
                )
    # split the data
    train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
    train_set, val_set = train_val_split.get_splits()
    verify_train_val_sets(train_set, val_set)
    # convert dataset from supervisely to yolo format
    if os.path.exists(g.yolov8_project_dir):
        sly.fs.clean_dir(g.yolov8_project_dir)
    transform(
        g.project_dir,
        g.yolov8_project_dir,
        train_set,
        val_set,
        progress_bar_convert_to_yolo,
        task_type,
    )
    # download model
    weights_type = model_tabs.get_active_tab()

    def download_monitor(monitor, api: sly.Api, progress: sly.Progress):
        value = monitor
        if progress.total == 0:
            progress.set(value, monitor.len, report=False)
        else:
            progress.set_current_value(value, report=False)
        weights_pbar.update(progress.current)

    if weights_type == "Pretrained models":
        selected_model = models_table.get_selected_row()[0]
        if selected_model.endswith("det"):
            selected_model = selected_model[:-4]
        if select_train_mode.get_value() == "Finetune mode":
            model_filename = selected_model.lower() + ".pt"
            pretrained = True
            weights_dst_path = os.path.join(g.app_data_dir, model_filename)
            weights_url = (
                f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_filename}"
            )
            with urlopen(weights_url) as file:
                weights_size = file.length

            progress = sly.Progress(
                message="",
                total_cnt=weights_size,
                is_size=True,
            )
            progress_cb = partial(download_monitor, api=api, progress=progress)

            with progress_bar_download_model(
                message="Downloading model weights...",
                total=weights_size,
                unit="bytes",
                unit_scale=True,
            ) as weights_pbar:
                sly.fs.download(
                    url=weights_url,
                    save_path=weights_dst_path,
                    progress=progress_cb,
                )
            model = YOLO(weights_dst_path)
        else:
            model_filename = selected_model.lower() + ".yaml"
            pretrained = False
            model = YOLO(model_filename)
    elif weights_type == "Custom models":
        custom_link = model_path_input.get_value()
        model_filename = "custom_model.pt"
        weights_dst_path = os.path.join(g.app_data_dir, model_filename)
        file_info = api.file.get_info_by_path(sly.env.team_id(), custom_link)
        if file_info is None:
            raise FileNotFoundError(f"Custon model file not found: {custom_link}")
        file_size = file_info.sizeb
        progress = sly.Progress(
            message="",
            total_cnt=file_size,
            is_size=True,
        )
        progress_cb = partial(download_monitor, api=api, progress=progress)
        with progress_bar_download_model(
            message="Downloading model weights...",
            total=file_size,
            unit="bytes",
            unit_scale=True,
        ) as weights_pbar:
            api.file.download(
                team_id=sly.env.team_id(),
                remote_path=custom_link,
                local_save_path=weights_dst_path,
                progress_cb=progress_cb,
            )
        pretrained = True
        model = YOLO(weights_dst_path)

    # add callbacks to model
    model.add_callback("on_train_batch_end", on_train_batch_end)

    progress_bar_download_model.hide()
    # get additional training params
    additional_params = train_settings_editor.get_text()
    additional_params = yaml.safe_load(additional_params)
    if task_type == "pose estimation":
        additional_params["fliplr"] = 0.0
    # set up epoch progress bar and grid plot
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    grid_plot_f.show()
    plot_notification.show()
    watch_file = os.path.join(local_artifacts_dir, "results.csv")
    plotted_train_batches = []
    remote_images_path = f"/yolov8_train/{task_type}/{project_info.name}/images/{g.app_session_id}/"

    def check_number(value):
        # if value is not str, NaN, infinity or negative infinity
        if isinstance(value, (int, float)) and math.isfinite(value):
            return True
        else:
            return False

    def on_results_file_changed(filepath, pbar):
        # read results file
        results = pd.read_csv(filepath)
        results.columns = [col.replace(" ", "") for col in results.columns]
        print(results.tail(1))
        # get losses values
        train_box_loss = results["train/box_loss"].iat[-1]
        train_cls_loss = results["train/cls_loss"].iat[-1]
        train_dfl_loss = results["train/dfl_loss"].iat[-1]
        if "train/pose_loss" in results.columns:
            train_pose_loss = results["train/pose_loss"].iat[-1]
        if "train/kobj_loss" in results.columns:
            train_kobj_loss = results["train/kobj_loss"].iat[-1]
        if "train/seg_loss" in results.columns:
            train_seg_loss = results["train/seg_loss"].iat[-1]
        precision = results["metrics/precision(B)"].iat[-1]
        recall = results["metrics/recall(B)"].iat[-1]
        val_box_loss = results["val/box_loss"].iat[-1]
        val_cls_loss = results["val/cls_loss"].iat[-1]
        val_dfl_loss = results["val/dfl_loss"].iat[-1]
        if "val/pose_loss" in results.columns:
            val_pose_loss = results["val/pose_loss"].iat[-1]
        if "val/kobj_loss" in results.columns:
            val_kobj_loss = results["val/kobj_loss"].iat[-1]
        if "val/seg_loss" in results.columns:
            val_seg_loss = results["val/seg_loss"].iat[-1]
        # update progress bar
        x = results["epoch"].iat[-1]
        pbar.update(int(x) + 1 - pbar.n)
        # add new points to plots
        if check_number(float(train_box_loss)):
            grid_plot.add_scalar("train/box loss", float(train_box_loss), int(x))
        if check_number(float(train_cls_loss)):
            grid_plot.add_scalar("train/cls loss", float(train_cls_loss), int(x))
        if check_number(float(train_dfl_loss)):
            grid_plot.add_scalar("train/dfl loss", float(train_dfl_loss), int(x))
        if "train/pose_loss" in results.columns:
            if check_number(float(train_pose_loss)):
                grid_plot.add_scalar("train/pose loss", float(train_pose_loss), int(x))
        if "train/kobj_loss" in results.columns:
            if check_number(float(train_kobj_loss)):
                grid_plot.add_scalar("train/kobj loss", float(train_kobj_loss), int(x))
        if "train/seg_loss" in results.columns:
            if check_number(float(train_seg_loss)):
                grid_plot.add_scalar("train/seg loss", float(train_seg_loss), int(x))
        if check_number(float(precision)):
            grid_plot.add_scalar("precision & recall/precision", float(precision), int(x))
        if check_number(float(recall)):
            grid_plot.add_scalar("precision & recall/recall", float(recall), int(x))
        if check_number(float(val_box_loss)):
            grid_plot.add_scalar("val/box loss", float(val_box_loss), int(x))
        if check_number(float(val_cls_loss)):
            grid_plot.add_scalar("val/cls loss", float(val_cls_loss), int(x))
        if check_number(float(val_dfl_loss)):
            grid_plot.add_scalar("val/dfl loss", float(val_dfl_loss), int(x))
        if "val/pose_loss" in results.columns:
            if check_number(float(val_pose_loss)):
                grid_plot.add_scalar("val/pose loss", float(val_pose_loss), int(x))
        if "val/kobj_loss" in results.columns:
            if check_number(float(val_kobj_loss)):
                grid_plot.add_scalar("val/kobj loss", float(val_kobj_loss), int(x))
        if "val/seg_loss" in results.columns:
            if check_number(float(val_seg_loss)):
                grid_plot.add_scalar("val/seg loss", float(val_seg_loss), int(x))
        # visualize train batch
        batch = f"train_batch{x}.jpg"
        local_train_batches_path = os.path.join(local_artifacts_dir, batch)
        if (
            os.path.exists(local_train_batches_path)
            and batch not in plotted_train_batches
            and x < 10
        ):
            plotted_train_batches.append(batch)
            remote_train_batches_path = os.path.join(remote_images_path, batch)
            tf_train_batches_info = api.file.upload(
                team_id, local_train_batches_path, remote_train_batches_path
            )
            train_batches_gallery.append(tf_train_batches_info.full_storage_url)
            if x == 0:
                train_batches_gallery_f.show()

    watcher = Watcher(
        watch_file,
        on_results_file_changed,
        progress_bar_epochs(message="Epochs:", total=n_epochs_input.get_value()),
    )
    # train model and upload best checkpoints to team files
    device = 0 if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(g.yolov8_project_dir, "data_config.yaml")
    sly.logger.info(f"Using device: {device}")

    def watcher_func():
        watcher.watch()

    def disable_watcher():
        watcher.running = False

    app.call_before_shutdown(disable_watcher)

    threading.Thread(target=watcher_func, daemon=True).start()
    if len(train_set) > 300:
        n_train_batches = math.ceil(len(train_set) / batch_size_input.get_value())
        train_batches_filepath = "train_batches.txt"

        def on_train_batches_file_changed(filepath, pbar):
            g.train_counter += 1
            if g.train_counter % n_train_batches == 0:
                g.train_counter = 0
                pbar.reset()
            else:
                pbar.update(g.train_counter % n_train_batches - pbar.n)

        train_batch_watcher = Watcher(
            train_batches_filepath,
            on_train_batches_file_changed,
            progress_bar_iters(message="Training batches:", total=n_train_batches),
        )

        def train_batch_watcher_func():
            train_batch_watcher.watch()

        def train_batch_watcher_disable():
            train_batch_watcher.running = False

        app.call_before_shutdown(train_batch_watcher_disable)

        threading.Thread(target=train_batch_watcher_func, daemon=True).start()

    def stop_on_batch_end_if_needed(*args, **kwargs):
        if app.is_stopped():
            raise app.StopException("This error is expected.")

    model.add_callback("on_train_batch_end", stop_on_batch_end_if_needed)
    model.add_callback("on_val_batch_end", stop_on_batch_end_if_needed)

    with app.handle_stop():
        model.train(
            data=data_path,
            epochs=n_epochs_input.get_value(),
            patience=patience_input.get_value(),
            batch=batch_size_input.get_value(),
            imgsz=image_size_input.get_value(),
            save_period=1000,
            device=device,
            workers=n_workers_input.get_value(),
            optimizer=select_optimizer.get_value(),
            pretrained=pretrained,
            project=checkpoint_dir,
            **additional_params,
        )

    progress_bar_iters.hide()
    progress_bar_epochs.hide()
    watcher.running = False

    # visualize model predictions
    for i in range(4):
        val_batch_labels_id, val_batch_preds_id = None, None
        labels_path = os.path.join(local_artifacts_dir, f"val_batch{i}_labels.jpg")
        if os.path.exists(labels_path):
            remote_labels_path = os.path.join(remote_images_path, f"val_batch{i}_labels.jpg")
            tf_labels_info = api.file.upload(team_id, labels_path, remote_labels_path)
            val_batch_labels_id = val_batches_gallery.append(
                image_url=tf_labels_info.full_storage_url,
                title="labels",
            )
        preds_path = os.path.join(local_artifacts_dir, f"val_batch{i}_pred.jpg")
        if os.path.exists(preds_path):
            remote_preds_path = os.path.join(remote_images_path, f"val_batch{i}_pred.jpg")
            tf_preds_info = api.file.upload(team_id, preds_path, remote_preds_path)
            val_batch_preds_id = val_batches_gallery.append(
                image_url=tf_preds_info.full_storage_url,
                title="predictions",
            )
        if val_batch_labels_id and val_batch_preds_id:
            val_batches_gallery.sync_images([val_batch_labels_id, val_batch_preds_id])
        if i == 0:
            val_batches_gallery_f.show()

    # visualize additional training results
    confusion_matrix_path = os.path.join(local_artifacts_dir, "confusion_matrix_normalized.png")
    if os.path.exists(confusion_matrix_path):
        remote_confusion_matrix_path = os.path.join(
            remote_images_path, "confusion_matrix_normalized.png"
        )
        tf_confusion_matrix_info = api.file.upload(
            team_id, confusion_matrix_path, remote_confusion_matrix_path
        )
        additional_gallery.append(tf_confusion_matrix_info.full_storage_url)
        additional_gallery_f.show()
    pr_curve_path = os.path.join(local_artifacts_dir, "PR_curve.png")
    if os.path.exists(pr_curve_path):
        remote_pr_curve_path = os.path.join(remote_images_path, "PR_curve.png")
        tf_pr_curve_info = api.file.upload(team_id, pr_curve_path, remote_pr_curve_path)
        additional_gallery.append(tf_pr_curve_info.full_storage_url)
    f1_curve_path = os.path.join(local_artifacts_dir, "F1_curve.png")
    if os.path.exists(f1_curve_path):
        remote_f1_curve_path = os.path.join(remote_images_path, "F1_curve.png")
        tf_f1_curve_info = api.file.upload(team_id, f1_curve_path, remote_f1_curve_path)
        additional_gallery.append(tf_f1_curve_info.full_storage_url)
    box_f1_curve_path = os.path.join(local_artifacts_dir, "BoxF1_curve.png")
    if os.path.exists(box_f1_curve_path):
        remote_box_f1_curve_path = os.path.join(remote_images_path, "BoxF1_curve.png")
        tf_box_f1_curve_info = api.file.upload(team_id, box_f1_curve_path, remote_box_f1_curve_path)
        additional_gallery.append(tf_box_f1_curve_info.full_storage_url)
    pose_f1_curve_path = os.path.join(local_artifacts_dir, "PoseF1_curve.png")
    if os.path.exists(pose_f1_curve_path):
        remote_pose_f1_curve_path = os.path.join(remote_images_path, "PoseF1_curve.png")
        tf_pose_f1_curve_info = api.file.upload(
            team_id, pose_f1_curve_path, remote_pose_f1_curve_path
        )
        additional_gallery.append(tf_pose_f1_curve_info.full_storage_url)
    mask_f1_curve_path = os.path.join(local_artifacts_dir, "MaskF1_curve.png")
    if os.path.exists(mask_f1_curve_path):
        remote_mask_f1_curve_path = os.path.join(remote_images_path, "MaskF1_curve.png")
        tf_mask_f1_curve_info = api.file.upload(
            team_id, mask_f1_curve_path, remote_mask_f1_curve_path
        )
        additional_gallery.append(tf_mask_f1_curve_info.full_storage_url)

    # rename best checkpoint file
    results = pd.read_csv(watch_file)
    results.columns = [col.replace(" ", "") for col in results.columns]
    results["fitness"] = (0.1 * results["metrics/mAP50(B)"]) + (
        0.9 * results["metrics/mAP50-95(B)"]
    )
    print("Final results:")
    print(results)
    best_epoch = results["fitness"].idxmax()
    best_filename = f"best_{best_epoch}.pt"
    current_best_filepath = os.path.join(local_artifacts_dir, "weights", "best.pt")
    new_best_filepath = os.path.join(local_artifacts_dir, "weights", best_filename)
    os.rename(current_best_filepath, new_best_filepath)

    # add geometry config to saved weights for pose estimation task
    if task_type == "pose estimation":
        for obj_class in project_meta.obj_classes:
            if (
                obj_class.geometry_type.geometry_name() == "graph"
                and obj_class.name in selected_classes
            ):
                geometry_config = obj_class.geometry_config
                break
        weights_filepath = os.path.join(local_artifacts_dir, "weights", best_filename)
        weights_dict = torch.load(weights_filepath)
        weights_dict["geometry_config"] = geometry_config
        torch.save(weights_dict, weights_filepath)

    # save link to app ui
    app_url = f"/apps/sessions/{g.app_session_id}"
    app_link_path = os.path.join(local_artifacts_dir, "open_app.lnk")
    with open(app_link_path, "w") as text_file:
        print(app_url, file=text_file)

    # upload training artifacts to team files
    remote_artifacts_dir = os.path.join(
        "/yolov8_train", task_type_select.get_value(), project_info.name, str(g.app_session_id)
    )

    def upload_monitor(monitor, api: sly.Api, progress: sly.Progress):
        value = monitor.bytes_read
        if progress.total == 0:
            progress.set(value, monitor.len, report=False)
        else:
            progress.set_current_value(value, report=False)
        artifacts_pbar.update(progress.current - artifacts_pbar.n)

    local_files = sly.fs.list_files_recursively(local_artifacts_dir)
    total_size = sum([sly.fs.get_file_size(file_path) for file_path in local_files])
    progress = sly.Progress(
        message="",
        total_cnt=total_size,
        is_size=True,
    )
    progress_cb = partial(upload_monitor, api=api, progress=progress)
    with progress_bar_upload_artifacts(
        message="Uploading train artifacts to Team Files...",
        total=total_size,
        unit="bytes",
        unit_scale=True,
    ) as artifacts_pbar:
        team_files_dir = api.file.upload_directory(
            team_id=sly.env.team_id(),
            local_dir=local_artifacts_dir,
            remote_dir=remote_artifacts_dir,
            progress_size_cb=progress_cb,
        )
    file_info = api.file.get_info_by_path(sly.env.team_id(), team_files_dir + "/results.csv")
    train_artifacts_folder.set(file_info)
    # finish training
    start_training_button.loading = False
    start_training_button.disable()
    logs_button.disable()
    train_done.show()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_train_artifacts.unlock()
    card_train_artifacts.uncollapse()
    # delete app data since it is no longer needed
    sly.fs.remove_dir(g.app_data_dir)
    sly.fs.silent_remove("train_batches.txt")
    # set task output
    sly.output.set_directory(remote_artifacts_dir)
    # stop app
    app.stop()


@server.post("/auto_train")
def auto_train(request: Request):
    sly.logger.info("Starting automatic training session...")
    state = request.state.state
    project_id = state["project_id"]
    task_type = state["task_type"]
    use_cache = state.get("use_cache", True)

    if task_type == "instance segmentation":
        models_table_columns = [key for key in g.seg_models_data[0].keys()]
        models_table_subtitles = [None] * len(models_table_columns)
        models_table_rows = []
        for element in g.seg_models_data:
            models_table_rows.append(list(element.values()))
        models_table.set_data(
            columns=models_table_columns,
            rows=models_table_rows,
            subtitles=models_table_subtitles,
        )
    elif task_type == "pose estimation":
        models_table_columns = [key for key in g.pose_models_data[0].keys()]
        models_table_subtitles = [None] * len(models_table_columns)
        models_table_rows = []
        for element in g.pose_models_data:
            models_table_rows.append(list(element.values()))
        models_table.set_data(
            columns=models_table_columns,
            rows=models_table_rows,
            subtitles=models_table_subtitles,
        )
    dataset_selector.disable()
    if use_cache:
        use_cache_checkbox.check()
    else: 
        use_cache_checkbox.uncheck()
    classes_table.read_project_from_id(project_id)
    classes_table.select_all()
    selected_classes = classes_table.get_selected_classes()
    _update_select_classes_button(selected_classes)
    select_data_button.hide()
    select_done.show()
    reselect_data_button.show()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_classes.unlock()
    card_classes.uncollapse()

    select_classes_button.hide()
    classes_done.show()
    select_other_classes_button.show()
    classes_table.disable()
    task_type_select.disable()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_train_val_split.unlock()
    card_train_val_split.uncollapse()

    train_val_split.disable()
    unlabeled_images_select.disable()
    split_data_button.hide()
    split_done.show()
    resplit_data_button.show()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_model_selection.unlock()
    card_model_selection.uncollapse()

    select_model_button.hide()
    model_select_done.show()
    model_tabs.disable()
    models_table.disable()
    model_path_input.disable()
    reselect_model_button.show()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_train_params.unlock()
    card_train_params.uncollapse()

    save_train_params_button.hide()
    train_params_done.show()
    reselect_train_params_button.show()
    select_train_mode.disable()
    n_epochs_input.disable()
    patience_input.disable()
    batch_size_input.disable()
    image_size_input.disable()
    select_optimizer.disable()
    n_workers_input.disable()
    train_settings_editor.readonly = True
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_train_progress.unlock()
    card_train_progress.uncollapse()

    local_dir = g.root_model_checkpoint_dir
    if task_type == "object detection":
        necessary_geometries = ["rectangle"]
        checkpoint_dir = os.path.join(local_dir, "detect")
        local_artifacts_dir = os.path.join(local_dir, "detect", "train")
    elif task_type == "pose estimation":
        necessary_geometries = ["graph", "rectangle"]
        checkpoint_dir = os.path.join(local_dir, "pose")
        local_artifacts_dir = os.path.join(local_dir, "pose", "train")
    elif task_type == "instance segmentation":
        necessary_geometries = ["bitmap", "polygon"]
        checkpoint_dir = os.path.join(local_dir, "segment")
        local_artifacts_dir = os.path.join(local_dir, "segment", "train")

    sly.logger.info(f"Local artifacts dir: {local_artifacts_dir}")

    if os.path.exists(local_artifacts_dir):
        sly.fs.remove_dir(local_artifacts_dir)
    # start_training_button.loading = True
    # get number of images in selected datasets
    if "dataset_ids" not in state:
        dataset_infos = api.dataset.get_list(project_id)
        dataset_ids = [dataset_info.id for dataset_info in dataset_infos]
    else:
        dataset_ids = state["dataset_ids"]
        dataset_infos = [api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids]
    download_project(
        api=api,
        project_info=project_info,
        dataset_infos=dataset_infos,
        use_cache=use_cache,
        progress=progress_bar_download_project
    )
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    selected_classes = [cls.name for cls in project_meta.obj_classes]
    # remove classes with unnecessary shapes
    if task_type != "object detection":
        unnecessary_classes = []
        for cls in project_meta.obj_classes:
            if (
                cls.name in selected_classes
                and cls.geometry_type.geometry_name() not in necessary_geometries
            ):
                unnecessary_classes.append(cls.name)
        if len(unnecessary_classes) > 0:
            sly.Project.remove_classes(
                g.project_dir, classes_to_remove=unnecessary_classes, inplace=True
            )
    # transfer project to detection task if necessary
    if task_type == "object detection":
        sly.Project.to_detection_task(g.project_dir, inplace=True)
    # split the data
    train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
    train_set, val_set = train_val_split.get_splits()
    verify_train_val_sets(train_set, val_set)
    # convert dataset from supervisely to yolo format
    if os.path.exists(g.yolov8_project_dir):
        sly.fs.clean_dir(g.yolov8_project_dir)
    transform(
        g.project_dir,
        g.yolov8_project_dir,
        train_set,
        val_set,
        progress_bar_convert_to_yolo,
        task_type,
    )

    # download model
    def download_monitor(monitor, api: sly.Api, progress: sly.Progress):
        value = monitor
        if progress.total == 0:
            progress.set(value, monitor.len, report=False)
        else:
            progress.set_current_value(value, report=False)
        weights_pbar.update(progress.current)

    if "model" not in state:
        selected_model = models_table.get_selected_row()[0]
    else:
        selected_model = state["model"]
    if selected_model.endswith("det"):
        selected_model = selected_model[:-4]
    if "train_mode" in state and state["train_mode"] == "scratch":
        model_filename = selected_model.lower() + ".yaml"
        pretrained = False
        model = YOLO(model_filename)
    else:
        model_filename = selected_model.lower() + ".pt"
        pretrained = True
        weights_dst_path = os.path.join(g.app_data_dir, model_filename)
        weights_url = (
            f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_filename}"
        )
        with urlopen(weights_url) as file:
            weights_size = file.length

        progress = sly.Progress(
            message="",
            total_cnt=weights_size,
            is_size=True,
        )
        progress_cb = partial(download_monitor, api=api, progress=progress)

        with progress_bar_download_model(
            message="Downloading model weights...",
            total=weights_size,
            unit="bytes",
            unit_scale=True,
        ) as weights_pbar:
            sly.fs.download(
                url=weights_url,
                save_path=weights_dst_path,
                progress=progress_cb,
            )
        model = YOLO(weights_dst_path)

    # add callbacks to model
    model.add_callback("on_train_batch_end", on_train_batch_end)

    progress_bar_download_model.hide()
    # get additional training params
    additional_params = train_settings_editor.get_text()
    additional_params = yaml.safe_load(additional_params)
    if task_type == "pose estimation":
        additional_params["fliplr"] = 0.0
        if "fliplr" in state:
            state["fliplr"] = 0.0
    # set up epoch progress bar and grid plot
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    grid_plot_f.show()
    plot_notification.show()
    watch_file = os.path.join(local_artifacts_dir, "results.csv")
    plotted_train_batches = []
    remote_images_path = f"/yolov8_train/{task_type}/{project_info.name}/images/{g.app_session_id}/"

    def check_number(value):
        # if value is not str, NaN, infinity or negative infinity
        if isinstance(value, (int, float)) and math.isfinite(value):
            return True
        else:
            return False

    def on_results_file_changed(filepath, pbar):
        # read results file
        results = pd.read_csv(filepath)
        results.columns = [col.replace(" ", "") for col in results.columns]
        print(results.tail(1))
        # get losses values
        train_box_loss = results["train/box_loss"].iat[-1]
        train_cls_loss = results["train/cls_loss"].iat[-1]
        train_dfl_loss = results["train/dfl_loss"].iat[-1]
        if "train/pose_loss" in results.columns:
            train_pose_loss = results["train/pose_loss"].iat[-1]
        if "train/kobj_loss" in results.columns:
            train_kobj_loss = results["train/kobj_loss"].iat[-1]
        if "train/seg_loss" in results.columns:
            train_seg_loss = results["train/seg_loss"].iat[-1]
        precision = results["metrics/precision(B)"].iat[-1]
        recall = results["metrics/recall(B)"].iat[-1]
        val_box_loss = results["val/box_loss"].iat[-1]
        val_cls_loss = results["val/cls_loss"].iat[-1]
        val_dfl_loss = results["val/dfl_loss"].iat[-1]
        if "val/pose_loss" in results.columns:
            val_pose_loss = results["val/pose_loss"].iat[-1]
        if "val/kobj_loss" in results.columns:
            val_kobj_loss = results["val/kobj_loss"].iat[-1]
        if "val/seg_loss" in results.columns:
            val_seg_loss = results["val/seg_loss"].iat[-1]
        # update progress bar
        x = results["epoch"].iat[-1]
        pbar.update(int(x) + 1 - pbar.n)
        # add new points to plots
        if check_number(float(train_box_loss)):
            grid_plot.add_scalar("train/box loss", float(train_box_loss), int(x))
        if check_number(float(train_cls_loss)):
            grid_plot.add_scalar("train/cls loss", float(train_cls_loss), int(x))
        if check_number(float(train_dfl_loss)):
            grid_plot.add_scalar("train/dfl loss", float(train_dfl_loss), int(x))
        if "train/pose_loss" in results.columns:
            if check_number(float(train_pose_loss)):
                grid_plot.add_scalar("train/pose loss", float(train_pose_loss), int(x))
        if "train/kobj_loss" in results.columns:
            if check_number(float(train_kobj_loss)):
                grid_plot.add_scalar("train/kobj loss", float(train_kobj_loss), int(x))
        if "train/seg_loss" in results.columns:
            if check_number(float(train_seg_loss)):
                grid_plot.add_scalar("train/seg loss", float(train_seg_loss), int(x))
        if check_number(float(precision)):
            grid_plot.add_scalar("precision & recall/precision", float(precision), int(x))
        if check_number(float(recall)):
            grid_plot.add_scalar("precision & recall/recall", float(recall), int(x))
        if check_number(float(val_box_loss)):
            grid_plot.add_scalar("val/box loss", float(val_box_loss), int(x))
        if check_number(float(val_cls_loss)):
            grid_plot.add_scalar("val/cls loss", float(val_cls_loss), int(x))
        if check_number(float(val_dfl_loss)):
            grid_plot.add_scalar("val/dfl loss", float(val_dfl_loss), int(x))
        if "val/pose_loss" in results.columns:
            if check_number(float(val_pose_loss)):
                grid_plot.add_scalar("val/pose loss", float(val_pose_loss), int(x))
        if "val/kobj_loss" in results.columns:
            if check_number(float(val_kobj_loss)):
                grid_plot.add_scalar("val/kobj loss", float(val_kobj_loss), int(x))
        if "val/seg_loss" in results.columns:
            if check_number(float(val_seg_loss)):
                grid_plot.add_scalar("val/seg loss", float(val_seg_loss), int(x))
        # visualize train batch
        batch = f"train_batch{x}.jpg"
        local_train_batches_path = os.path.join(local_artifacts_dir, batch)
        if (
            os.path.exists(local_train_batches_path)
            and batch not in plotted_train_batches
            and x < 10
        ):
            plotted_train_batches.append(batch)
            remote_train_batches_path = os.path.join(remote_images_path, batch)
            tf_train_batches_info = api.file.upload(
                team_id, local_train_batches_path, remote_train_batches_path
            )
            train_batches_gallery.append(tf_train_batches_info.full_storage_url)
            if x == 0:
                train_batches_gallery_f.show()

    watcher = Watcher(
        watch_file,
        on_results_file_changed,
        progress_bar_epochs(message="Epochs:", total=n_epochs_input.get_value()),
    )
    # train model and upload best checkpoints to team files
    device = 0 if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(g.yolov8_project_dir, "data_config.yaml")
    sly.logger.info(f"Using device: {device}")

    def watcher_func():
        watcher.watch()

    threading.Thread(target=watcher_func, daemon=True).start()
    if len(train_set) > 300:
        n_train_batches = math.ceil(len(train_set) / batch_size_input.get_value())
        train_batches_filepath = "train_batches.txt"

        def on_train_batches_file_changed(filepath, pbar):
            g.train_counter += 1
            if g.train_counter % n_train_batches == 0:
                g.train_counter = 0
                pbar.reset()
            else:
                pbar.update(g.train_counter % n_train_batches - pbar.n)

        train_batch_watcher = Watcher(
            train_batches_filepath,
            on_train_batches_file_changed,
            progress_bar_iters(message="Training batches:", total=n_train_batches),
        )

        def train_batch_watcher_func():
            train_batch_watcher.watch()

        threading.Thread(target=train_batch_watcher_func, daemon=True).start()

    model.train(
        data=data_path,
        project=checkpoint_dir,
        epochs=state.get("n_epochs", n_epochs_input.get_value()),
        patience=state.get("patience", patience_input.get_value()),
        batch=state.get("batch_size", batch_size_input.get_value()),
        imgsz=state.get("input_image_size", image_size_input.get_value()),
        save_period=1000,
        device=device,
        workers=state.get("n_workers", n_workers_input.get_value()),
        optimizer=state.get("optimizer", select_optimizer.get_value()),
        pretrained=pretrained,
        lr0=state.get("lr0", additional_params["lr0"]),
        lrf=state.get("lrf", additional_params["lr0"]),
        momentum=state.get("momentum", additional_params["momentum"]),
        weight_decay=state.get("weight_decay", additional_params["weight_decay"]),
        warmup_epochs=state.get("warmup_epochs", additional_params["warmup_epochs"]),
        warmup_momentum=state.get("warmup_momentum", additional_params["warmup_momentum"]),
        warmup_bias_lr=state.get("warmup_bias_lr", additional_params["warmup_bias_lr"]),
        amp=state.get("amp", additional_params["amp"]),
        hsv_h=state.get("hsv_h", additional_params["hsv_h"]),
        hsv_s=state.get("hsv_s", additional_params["hsv_s"]),
        hsv_v=state.get("hsv_v", additional_params["hsv_v"]),
        degrees=state.get("degrees", additional_params["degrees"]),
        translate=state.get("translate", additional_params["translate"]),
        scale=state.get("scale", additional_params["scale"]),
        shear=state.get("shear", additional_params["shear"]),
        perspective=state.get("perspective", additional_params["perspective"]),
        flipud=state.get("flipud", additional_params["flipud"]),
        fliplr=state.get("fliplr", additional_params["fliplr"]),
        mosaic=state.get("mosaic", additional_params["mosaic"]),
        mixup=state.get("mixup", additional_params["mixup"]),
        copy_paste=state.get("copy_paste", additional_params["copy_paste"]),
    )
    progress_bar_iters.hide()
    progress_bar_epochs.hide()
    watcher.running = False

    # visualize model predictions
    for i in range(4):
        val_batch_labels_id, val_batch_preds_id = None, None
        labels_path = os.path.join(local_artifacts_dir, f"val_batch{i}_labels.jpg")
        if os.path.exists(labels_path):
            remote_labels_path = os.path.join(remote_images_path, f"val_batch{i}_labels.jpg")
            tf_labels_info = api.file.upload(team_id, labels_path, remote_labels_path)
            val_batch_labels_id = val_batches_gallery.append(
                image_url=tf_labels_info.full_storage_url,
                title="labels",
            )
        preds_path = os.path.join(local_artifacts_dir, f"val_batch{i}_pred.jpg")
        if os.path.exists(preds_path):
            remote_preds_path = os.path.join(remote_images_path, f"val_batch{i}_pred.jpg")
            tf_preds_info = api.file.upload(team_id, preds_path, remote_preds_path)
            val_batch_preds_id = val_batches_gallery.append(
                image_url=tf_preds_info.full_storage_url,
                title="predictions",
            )
        if val_batch_labels_id and val_batch_preds_id:
            val_batches_gallery.sync_images([val_batch_labels_id, val_batch_preds_id])
        if i == 0:
            val_batches_gallery_f.show()

    # visualize additional training results
    confusion_matrix_path = os.path.join(local_artifacts_dir, "confusion_matrix_normalized.png")
    if os.path.exists(confusion_matrix_path):
        remote_confusion_matrix_path = os.path.join(
            remote_images_path, "confusion_matrix_normalized.png"
        )
        tf_confusion_matrix_info = api.file.upload(
            team_id, confusion_matrix_path, remote_confusion_matrix_path
        )
        additional_gallery.append(tf_confusion_matrix_info.full_storage_url)
        additional_gallery_f.show()
    pr_curve_path = os.path.join(local_artifacts_dir, "PR_curve.png")
    if os.path.exists(pr_curve_path):
        remote_pr_curve_path = os.path.join(remote_images_path, "PR_curve.png")
        tf_pr_curve_info = api.file.upload(team_id, pr_curve_path, remote_pr_curve_path)
        additional_gallery.append(tf_pr_curve_info.full_storage_url)
    f1_curve_path = os.path.join(local_artifacts_dir, "F1_curve.png")
    if os.path.exists(f1_curve_path):
        remote_f1_curve_path = os.path.join(remote_images_path, "F1_curve.png")
        tf_f1_curve_info = api.file.upload(team_id, f1_curve_path, remote_f1_curve_path)
        additional_gallery.append(tf_f1_curve_info.full_storage_url)
    box_f1_curve_path = os.path.join(local_artifacts_dir, "BoxF1_curve.png")
    if os.path.exists(box_f1_curve_path):
        remote_box_f1_curve_path = os.path.join(remote_images_path, "BoxF1_curve.png")
        tf_box_f1_curve_info = api.file.upload(team_id, box_f1_curve_path, remote_box_f1_curve_path)
        additional_gallery.append(tf_box_f1_curve_info.full_storage_url)
    pose_f1_curve_path = os.path.join(local_artifacts_dir, "PoseF1_curve.png")
    if os.path.exists(pose_f1_curve_path):
        remote_pose_f1_curve_path = os.path.join(remote_images_path, "PoseF1_curve.png")
        tf_pose_f1_curve_info = api.file.upload(
            team_id, pose_f1_curve_path, remote_pose_f1_curve_path
        )
        additional_gallery.append(tf_pose_f1_curve_info.full_storage_url)
    mask_f1_curve_path = os.path.join(local_artifacts_dir, "MaskF1_curve.png")
    if os.path.exists(mask_f1_curve_path):
        remote_mask_f1_curve_path = os.path.join(remote_images_path, "MaskF1_curve.png")
        tf_mask_f1_curve_info = api.file.upload(
            team_id, mask_f1_curve_path, remote_mask_f1_curve_path
        )
        additional_gallery.append(tf_mask_f1_curve_info.full_storage_url)

    # rename best checkpoint file
    results = pd.read_csv(watch_file)
    results.columns = [col.replace(" ", "") for col in results.columns]
    results["fitness"] = (0.1 * results["metrics/mAP50(B)"]) + (
        0.9 * results["metrics/mAP50-95(B)"]
    )
    print("Final results:")
    print(results)
    best_epoch = results["fitness"].idxmax()
    best_filename = f"best_{best_epoch}.pt"
    current_best_filepath = os.path.join(local_artifacts_dir, "weights", "best.pt")
    new_best_filepath = os.path.join(local_artifacts_dir, "weights", best_filename)
    os.rename(current_best_filepath, new_best_filepath)

    # add geometry config to saved weights for pose estimation task
    if task_type == "pose estimation":
        for obj_class in project_meta.obj_classes:
            if (
                obj_class.geometry_type.geometry_name() == "graph"
                and obj_class.name in selected_classes
            ):
                geometry_config = obj_class.geometry_config
                break
        weights_filepath = os.path.join(local_artifacts_dir, "weights", best_filename)
        weights_dict = torch.load(weights_filepath)
        weights_dict["geometry_config"] = geometry_config
        torch.save(weights_dict, weights_filepath)

    # save link to app ui
    app_url = f"/apps/sessions/{g.app_session_id}"
    app_link_path = os.path.join(local_artifacts_dir, "open_app.lnk")
    with open(app_link_path, "w") as text_file:
        print(app_url, file=text_file)

    # upload training artifacts to team files
    remote_artifacts_dir = os.path.join(
        "/yolov8_train", task_type, project_info.name, str(g.app_session_id)
    )

    def upload_monitor(monitor, api: sly.Api, progress: sly.Progress):
        value = monitor.bytes_read
        if progress.total == 0:
            progress.set(value, monitor.len, report=False)
        else:
            progress.set_current_value(value, report=False)
        artifacts_pbar.update(progress.current - artifacts_pbar.n)

    local_files = sly.fs.list_files_recursively(local_artifacts_dir)
    total_size = sum([sly.fs.get_file_size(file_path) for file_path in local_files])
    progress = sly.Progress(
        message="",
        total_cnt=total_size,
        is_size=True,
    )
    progress_cb = partial(upload_monitor, api=api, progress=progress)
    with progress_bar_upload_artifacts(
        message="Uploading train artifacts to Team Files...",
        total=total_size,
        unit="bytes",
        unit_scale=True,
    ) as artifacts_pbar:
        team_files_dir = api.file.upload_directory(
            team_id=sly.env.team_id(),
            local_dir=local_artifacts_dir,
            remote_dir=remote_artifacts_dir,
            progress_size_cb=progress_cb,
        )
    file_info = api.file.get_info_by_path(sly.env.team_id(), team_files_dir + "/results.csv")
    train_artifacts_folder.set(file_info)
    # finish training
    start_training_button.loading = False
    start_training_button.disable()
    logs_button.disable()
    train_done.show()
    curr_step = stepper.get_active_step()
    curr_step += 1
    stepper.set_active_step(curr_step)
    card_train_artifacts.unlock()
    card_train_artifacts.uncollapse()
    # delete app data since it is no longer needed
    sly.fs.remove_dir(g.app_data_dir)
    sly.fs.silent_remove("train_batches.txt")
    # set task output
    sly.output.set_directory(remote_artifacts_dir)
    return {"result": "successfully finished automatic training session"}
