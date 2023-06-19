import os
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
)
from src.utils import get_train_val_sets, verify_train_val_sets
from src.sly_to_yolov8 import transform
from ultralytics import YOLO
import torch
from src.metrics_watcher import Watcher
import threading
import pandas as pd
from functools import partial
from urllib.request import urlopen
import math
import shutil


# function for updating global variables
def update_globals(new_dataset_ids):
    global dataset_ids, project_id, workspace_id, project_info, project_meta
    dataset_ids = new_dataset_ids
    if dataset_ids:
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
sly.fs.mkdir(g.static_dir)

# if app had started from context menu, one of this has to be set:
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)
dataset_ids = [dataset_id] if dataset_id else []
update_globals(dataset_ids)

sly.logger.info(f"App root directory: {g.app_root_directory}")


### 1. Dataset selection
dataset_selector = SelectDataset(project_id=project_id, multiselect=True, select_all_datasets=True)
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
    description="Select task type and classes, that should be used for training. Supported shapes include rectangle, bitmap, polygon and graph",
    content=classes_content,
    collapsable=True,
    lock_message="Complete the previous step to unlock",
)
card_classes.collapse()
card_classes.lock()


### 3. Train / validation split
train_val_split = TrainValSplits(project_id=project_id)
unlabeled_images_select = SelectString(values=["keep unlabeled images", "ignore unlabeled images"])
unlabeled_images_select_f = Field(
    content=unlabeled_images_select,
    title="What to do with unlabeled images",
    description="Sometimes unlabeled images can be used to reduce noise in predictions, sometimes it is a mistake in training data",
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
train_val_content = Container(
    [
        train_val_split,
        unlabeled_images_select_f,
        split_data_button,
        resplit_data_button,
        split_done,
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
    description="If you want to see list of available models for another computer vision task, please, go back to task type & training classes step and change task type",
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
model_select_done = DoneLabel("Model was successfully selected")
model_select_done.hide()
model_selection_content = Container(
    [
        model_tabs,
        select_model_button,
        reselect_model_button,
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
    description="Choose whether to finetune pretrained model or train model from scratch",
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
reselect_train_params_button = Button(
    '<i style="margin-right: 5px" class="zmdi zmdi-rotate-left"></i>Change training hyperparameters',
    button_type="warning",
    button_size="small",
    plain=True,
)
reselect_train_params_button.hide()
train_params_done = DoneLabel("Successfully saved training hyperparameters")
train_params_done.hide()
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
        train_settings_editor_f,
        save_train_params_button,
        reselect_train_params_button,
        train_params_done,
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
progress_bar_download_project = Progress()
progress_bar_convert_to_yolo = Progress()
progress_bar_download_model = Progress()
progress_bar_epochs = Progress()
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
progress_bar_upload_artifacts = Progress()
train_done = DoneLabel("Training completed. Training artifacts were uploaded to Team Files")
train_done.hide()
train_progress_content = Container(
    [
        start_training_button,
        progress_bar_download_project,
        progress_bar_convert_to_yolo,
        progress_bar_download_model,
        progress_bar_epochs,
        grid_plot_f,
        plot_notification,
        train_batches_gallery_f,
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


app = sly.Application(
    layout=Container(
        widgets=[
            card_project_settings,
            card_classes,
            card_train_val_split,
            card_model_selection,
            card_train_params,
            card_train_progress,
            card_train_artifacts,
        ]
    ),
    static_dir=g.static_dir,
)


@dataset_selector.value_changed
def on_dataset_selected(new_dataset_ids):
    if new_dataset_ids == []:
        select_data_button.hide()
    elif new_dataset_ids != [] and reselect_data_button.is_hidden():
        select_data_button.show()
    update_globals(new_dataset_ids)


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
    classes_table.read_meta(project_meta)
    select_data_button.loading = False
    select_data_button.hide()
    select_done.show()
    reselect_data_button.show()
    card_classes.unlock()
    card_classes.uncollapse()


@reselect_data_button.click
def reselect_input_data():
    select_data_button.show()
    reselect_data_button.hide()
    select_done.hide()
    dataset_selector.enable()


@classes_table.value_changed
def on_classes_selected(selected_classes):
    n_classes = len(selected_classes)
    if n_classes > 0:
        if n_classes > 1:
            select_classes_button.text = f"Select {n_classes} classes"
        else:
            select_classes_button.text = f"Select {n_classes} class"
        select_classes_button.show()
    else:
        select_classes_button.hide()


@task_type_select.value_changed
def select_task(task_type):
    project_shapes = [cls.geometry_type.geometry_name() for cls in project_meta.obj_classes]
    if task_type == "object detection":
        if "rectangle" not in project_shapes:
            sly.app.show_dialog(
                title="There are no classes of shape rectangle in selected project",
                description="Please, change task type or select another project with classes of shape rectangle",
                status="warning",
            )
            select_classes_button.disable()
        else:
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
    card_train_val_split.unlock()
    card_train_val_split.uncollapse()


@select_other_classes_button.click
def select_other_classes():
    classes_table.enable()
    task_type_select.enable()
    select_other_classes_button.hide()
    classes_done.hide()
    select_classes_button.show()


@split_data_button.click
def split_data():
    train_val_split.disable()
    unlabeled_images_select.disable()
    split_data_button.hide()
    split_done.show()
    resplit_data_button.show()
    card_model_selection.unlock()
    card_model_selection.uncollapse()


@resplit_data_button.click
def resplit_data():
    train_val_split.enable()
    unlabeled_images_select.enable()
    split_data_button.show()
    split_done.hide()
    resplit_data_button.hide()


@select_model_button.click
def select_model():
    select_model_button.hide()
    model_select_done.show()
    model_tabs.disable()
    models_table.disable()
    model_path_input.disable()
    reselect_model_button.show()
    card_train_params.unlock()
    card_train_params.uncollapse()


@reselect_model_button.click
def reselect_model():
    select_model_button.show()
    model_select_done.hide()
    model_tabs.enable()
    models_table.enable()
    model_path_input.enable()
    reselect_model_button.hide()


@model_path_input.value_changed
def change_file_preview(value):
    file_info = None
    if value != "":
        file_info = api.file.get_info_by_path(sly.env.team_id(), value)
    model_file_thumbnail.set(file_info)


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


@start_training_button.click
def start_training():
    task_type = task_type_select.get_value()
    if sly.is_production():
        local_dir = g.root_source_path
    else:
        local_dir = g.app_root_directory
    if task_type == "object detection":
        necessary_geometries = ["rectangle"]
        local_artifacts_dir = os.path.join(local_dir, "runs", "detect", "train")
    elif task_type == "pose estimation":
        necessary_geometries = ["graph"]
        local_artifacts_dir = os.path.join(local_dir, "runs", "pose", "train")
    elif task_type == "instance segmentation":
        necessary_geometries = ["bitmap", "polygon"]
        local_artifacts_dir = os.path.join(local_dir, "runs", "segment", "train")

    sly.logger.info(f"Local artifacts dir: {local_artifacts_dir}")

    if os.path.exists(local_artifacts_dir):
        sly.fs.remove_dir(local_artifacts_dir)
    start_training_button.loading = True
    # get number of images in selected datasets
    n_images = 0
    for dataset_id in dataset_ids:
        dataset_info = api.dataset.get_info_by_id(dataset_id)
        n_images += dataset_info.images_count
    # download dataset
    if os.path.exists(g.project_dir):
        sly.fs.clean_dir(g.project_dir)
    with progress_bar_download_project(message="Downloading input data...", total=n_images) as pbar:
        sly.download(
            api=api,
            project_id=project_id,
            dest_dir=g.project_dir,
            dataset_ids=dataset_ids,
            log_progress=True,
            progress_cb=pbar.update,
        )
    # remove unselected classes
    selected_classes = classes_table.get_selected_classes()
    sly.Project.remove_classes_except(g.project_dir, classes_to_keep=selected_classes, inplace=True)
    # remove classes with unnecessary shapes
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
    # remove unlabeled images if such option was selected by user
    if unlabeled_images_select.get_value() == "ignore unlabeled images":
        n_images_before = n_images
        sly.Project.remove_items_without_objects(g.project_dir, inplace=True)
        project = sly.Project(g.project_dir, sly.OpenMode.READ)
        n_images_after = project.total_items
        if n_images_before != n_images_after:
            random_content = train_val_split._get_random_content()
            random_split_table = random_content._widgets[0]
            split_counts = random_split_table.get_splits_counts()
            val_part = split_counts["val"] / split_counts["total"]
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
    train_set, val_set = get_train_val_sets(g.project_dir, train_val_split, api, project_id)
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
            shutil.copy(local_train_batches_path, g.static_dir)
            # show images
            static_train_batches_path = f"/static/train_batch{x}.jpg"
            train_batches_gallery.append(static_train_batches_path)
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
        **additional_params,
    )
    progress_bar_epochs.hide()
    watcher.running = False

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
    train_done.show()
    card_train_artifacts.unlock()
    card_train_artifacts.uncollapse()
    # delete app data since it is no longer needed
    sly.fs.remove_dir(g.app_data_dir)
    app.stop()
