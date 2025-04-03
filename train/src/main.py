import os
import re
from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import random
import threading
import uuid
from functools import partial
from pathlib import Path
from urllib.request import urlopen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruamel.yaml
import supervisely as sly
import supervisely.io.env as env
import torch
import src.globals as g
from src.utils import custom_plot, get_eval_results_dir_name, verify_train_val_sets
from src.sly_to_yolov8 import check_bbox_exist_on_images, transform
from src.dataset_cache import download_project
import src.workflow as w
from src.metrics_watcher import Watcher
from src.serve import YOLOv8ModelMB
import yaml
from dotenv import load_dotenv
from fastapi import Request, Response
from supervisely._utils import abs_url, is_development
from supervisely.app.widgets import (  # SelectDataset,
    Button,
    Card,
    Checkbox,
    ClassesTable,
    Collapse,
    Container,
    Dialog,
    DoneLabel,
    Editor,
    Empty,
    Field,
    FileThumbnail,
    Flexbox,
    FolderThumbnail,
    GridGallery,
    GridPlot,
    ImageSlider,
    Input,
    InputNumber,
    NotificationBox,
    Progress,
    RadioGroup,
    RadioTable,
    RadioTabs,
    RandomSplitsTable,
    ReloadableArea,
    ReportThumbnail,
    SelectDatasetTree,
    SelectString,
    SlyTqdm,
    Stepper,
    Switch,
    TaskLogs,
    Text,
    TrainValSplits,
    Tooltip,
)
from supervisely.nn.artifacts.yolov8 import YOLOv8
from supervisely.nn.benchmark import (
    ObjectDetectionBenchmark,
    InstanceSegmentationBenchmark,
)
from supervisely.nn.inference import SessionJSON
from supervisely.nn import TaskType
from ultralytics.utils.metrics import ConfusionMatrix
from src.early_stopping.custom_yolo import YOLO as CustomYOLO
import ruamel.yaml
import io

from src.profiler import MemoryProfiler


ConfusionMatrix.plot = custom_plot
plt.switch_backend("Agg")
root_source_path = str(Path(__file__).parents[2])

# authentication
if sly.is_development():
    load_dotenv("local.env")
    load_dotenv("supervisely.env")
api = sly.Api(retry_count=7)
team_id = sly.env.team_id()
server_address = sly.env.server_address()


def update_split_tabs_for_nested_datasets(selected_dataset_ids):
    global dataset_ids, train_val_split, ds_name_to_id
    sum_items_count = 0
    temp_dataset_names = set()
    temp_dataset_infos = []
    datasets_tree = api.dataset.get_tree(project_id)

    dataset_id_to_info = {}
    ds_name_to_id = {}

    def _get_dataset_ids_infos_map(ds_tree):
        for ds_info in ds_tree.keys():
            dataset_id_to_info[ds_info.id] = ds_info
            if ds_tree[ds_info]:
                _get_dataset_ids_infos_map(ds_tree[ds_info])

    _get_dataset_ids_infos_map(datasets_tree)

    def _get_full_name(ds_id):
        ds_info = dataset_id_to_info[ds_id]
        full_name = ds_info.name
        while ds_info.parent_id is not None:
            ds_info = dataset_id_to_info[ds_info.parent_id]
            full_name = ds_info.name + "/" + full_name
        return full_name

    for ds_id in selected_dataset_ids:

        def _get_dataset_infos(ds_tree, nested=False):

            for ds_info in ds_tree.keys():
                need_add = ds_info.id == ds_id or nested
                if need_add:
                    temp_dataset_infos.append(ds_info)
                    name = _get_full_name(ds_info.id)
                    temp_dataset_names.add(name)
                    ds_name_to_id[name] = ds_info.id
                if ds_tree[ds_info]:
                    _get_dataset_infos(ds_tree[ds_info], nested=need_add)

        _get_dataset_infos(datasets_tree)

    dataset_ids = list(set([ds_info.id for ds_info in temp_dataset_infos]))
    unique_ds = set([ds_info for ds_info in temp_dataset_infos])
    sum_items_count = sum([ds_info.items_count for ds_info in unique_ds])

    contents = []
    split_methods = []
    tabs_descriptions = []

    split_methods.append("Random")
    tabs_descriptions.append("Shuffle data and split with defined probability")
    contents.append(
        Container([RandomSplitsTable(sum_items_count)], direction="vertical", gap=5)
    )

    split_methods.append("Based on item tags")
    tabs_descriptions.append("Images should have assigned train or val tag")
    contents.append(train_val_split._get_tags_content())

    split_methods.append("Based on datasets")
    tabs_descriptions.append("Select one or several datasets for every split")

    notification_box = NotificationBox(
        title="Notice: How to make equal splits",
        description="Choose the same dataset(s) for train/validation to make splits equal. Can be used for debug and for tiny projects",
        box_type="info",
    )
    train_ds_select = SelectString(temp_dataset_names, multiple=True)
    val_ds_select = SelectString(temp_dataset_names, multiple=True)
    train_val_split._train_ds_select = train_ds_select
    train_val_split._val_ds_select = val_ds_select
    train_field = Field(
        train_ds_select,
        title="Train dataset(s)",
        description="all images in selected dataset(s) are considered as training set",
    )
    val_field = Field(
        val_ds_select,
        title="Validation dataset(s)",
        description="all images in selected dataset(s) are considered as validation set",
    )

    contents.append(
        Container(
            widgets=[notification_box, train_field, val_field],
            direction="vertical",
            gap=5,
        )
    )
    content = RadioTabs(
        titles=split_methods,
        descriptions=tabs_descriptions,
        contents=contents,
    )
    train_val_split._content = content
    train_val_split.update_data()
    train_val_split_area.reload()


# function for updating global variables
def update_globals(new_dataset_ids):
    sly.logger.debug(f"Updating globals with new dataset_ids: {new_dataset_ids}")
    global dataset_ids, project_id, workspace_id, project_info, project_meta
    dataset_ids = new_dataset_ids
    if dataset_ids and all(ds_id is not None for ds_id in dataset_ids):
        project_id = api.dataset.get_info_by_id(dataset_ids[0]).project_id
        workspace_id = api.project.get_info_by_id(project_id).workspace_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        print(f"Project is {project_info.name}, {dataset_ids}")
    elif project_id:
        workspace_id = api.project.get_info_by_id(
            project_id, raise_error=True
        ).workspace_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    else:
        print("All globals set to None")
        dataset_ids = []
        project_id, workspace_id, project_info, project_meta = [None] * 4


yolov8_artifacts = YOLOv8(team_id)
framework_folder = yolov8_artifacts.framework_folder

# if app had started from context menu, one of this has to be set:
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)
dataset_ids = [dataset_id] if dataset_id else []
update_globals(dataset_ids)

sly.logger.info(f"App root directory: {g.app_root_directory}")


### 1. Dataset selection
# dataset_selector = SelectDataset(project_id=project_id, multiselect=True, select_all_datasets=True)
dataset_selector = SelectDatasetTree(
    project_id=project_id,
    multiselect=True,
    select_all_datasets=True,
    allowed_project_types=[sly.ProjectType.IMAGES],
)
use_cache_text = Text(
    "Use cached data stored on the agent to optimize project download"
)
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
card_project_settings = Card(
    title="Dataset selection", content=project_settings_content
)


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
train_val_split_area = ReloadableArea(train_val_split)
unlabeled_images_select = SelectString(
    values=["keep unlabeled images", "ignore unlabeled images"]
)
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
bbox_miss_dialog = Dialog(
    title="Images with no bounding boxes", content=bbox_miss_content
)
bbox_miss_check_progress = Progress()
train_val_content = Container(
    [
        train_val_split_area,
        unlabeled_images_select_f,
        split_data_button,
        resplit_data_button,
        split_done,
        bbox_miss_dialog,
        bbox_miss_check_progress,
        bbox_miss_collapse,
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
models_table_columns = [
    key
    for key in g.det_models_data[0].keys()
    if key not in ["weights_url", "yaml_config"]
]
models_table_rows = []
for element in g.det_models_data:
    models_table_rows.append(list(element.values())[:-2])
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
freeze_layers = Switch(switched=False)
freeze_layers_f = Field(
    content=freeze_layers,
    title="Freeze layers",
    description=(
        "Layer freezing is a technique used in transfer learning to keep the "
        "weights of specific layers unchanged during fine-tuning. It allows "
        "to preserve learned features in frozen layers and focus on fine-tuning "
        "only necessary parts of neural network to fit new dataset. This technique"
        " can be used to reduce computational load and prevent overfitting when "
        "fine-tuning model on small datasets"
    ),
)
n_frozen_layers_input = InputNumber(value=1, min=1, max=90)
n_frozen_layers_input_f = Field(
    content=n_frozen_layers_input, title="Number of layers to freeze"
)
n_frozen_layers_input_f.hide()

# Model Benchmark evaluation
run_model_benchmark_checkbox = Checkbox(
    content="Run Model Benchmark evaluation", checked=True
)
run_speedtest_checkbox = Checkbox(content="Run speed test", checked=True)
model_benchmark_f = Field(
    Container(
        widgets=[
            run_model_benchmark_checkbox,
            run_speedtest_checkbox,
        ]
    ),
    title="Model Evaluation Benchmark",
    description=f"Generate evaluation dashboard with visualizations and detailed analysis of the model performance after training. The best checkpoint will be used for evaluation. You can also run speed test to evaluate model inference speed.",
)
docs_link = '<a href="https://docs.supervisely.com/neural-networks/model-evaluation-benchmark/" target="_blank">documentation</a>'
model_benchmark_learn_more = Text(
    f"Learn more about Model Benchmark in the {docs_link}.", status="info"
)

# ONNX / TensorRT export
export_model_switch = Switch(switched=False)
export_model_switch_f = Field(
    content=export_model_switch,
    title="Export weights to ONNX / TensorRT format",
    description="After training the 'best.pt' checkpoint will be exported to ONNX or TensorRT format and saved to Team Files. "
    "Exported model can be deployed in various frameworks and used for efficient inference on edge devices.",
)
export_onnx_checkbox = Checkbox(content="Export to ONNX", checked=False)
export_tensorrt_checkbox = Checkbox(
    content="Export to TensorRT (may take some time)", checked=False
)
export_fp16_switch = Switch(switched=False)
export_fp16_switch_f = Field(
    content=export_fp16_switch,
    title="FP16 mode",
    description="Export model in FP16 precision to reduce model size and increase inference speed.",
)
export_model_container = Container(
    [
        export_onnx_checkbox,
        export_tensorrt_checkbox,
        export_fp16_switch_f,
    ]
)
export_model_container.hide()

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
        freeze_layers_f,
        n_frozen_layers_input_f,
        model_benchmark_f,
        model_benchmark_learn_more,
        Empty(),  # add gap
        export_model_switch_f,
        export_model_container,
        Empty(),  # add gap
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
stop_training_button = Button(text="stop training", button_type="danger")
stop_training_tooltip = Tooltip(
    text="all training artefacts will be saved",
    content=stop_training_button,
    placement="right",
)
stop_training_tooltip.hide()
start_stop_container = Container(
    widgets=[
        start_training_button,
        stop_training_tooltip,
        Empty(),
    ],
    direction="horizontal",
    overflow="wrap",
    fractions=[1, 1, 4],
)
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
making_training_vis_f = Field(Empty(), "", "Making training visualizations...")
making_training_vis_f.hide()
uploading_artefacts_f = Field(Empty(), "", "Uploading Artefacts...")
uploading_artefacts_f.hide()
creating_report_f = Field(Empty(), "", "Creating report on model...")
creating_report_f.hide()
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
additional_gallery_f = Field(
    additional_gallery, "Additional training results visualization"
)
additional_gallery_f.hide()
progress_bar_upload_artifacts = Progress()
model_benchmark_pbar = SlyTqdm()
model_benchmark_pbar_secondary = Progress(hide_on_finish=False)
train_done = DoneLabel(
    "Training completed. Training artifacts were uploaded to Team Files"
)
train_done.hide()
train_progress_content = Container(
    [
        start_stop_container,
        logs_button,
        task_logs,
        creating_report_f,
        progress_bar_download_project,
        progress_bar_convert_to_yolo,
        progress_bar_download_model,
        progress_bar_epochs,
        progress_bar_iters,
        progress_bar_upload_artifacts,
        model_benchmark_pbar,
        model_benchmark_pbar_secondary,
        train_done,
        grid_plot_f,
        plot_notification,
        train_batches_gallery_f,
        val_batches_gallery_f,
        additional_gallery_f,
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

model_benchmark_report = ReportThumbnail()
model_benchmark_report.hide()
card_train_artifacts = Card(
    title="Training artifacts",
    description="Checkpoints, logs and other visualizations",
    content=Container([train_artifacts_folder, model_benchmark_report]),
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
    sly.logger.debug(f"Selected datasets widget value changed to: {new_dataset_ids}")
    if new_dataset_ids == []:
        select_data_button.hide()
    elif new_dataset_ids != [] and reselect_data_button.is_hidden():
        select_data_button.show()
    update_globals(new_dataset_ids)
    if sly.project.download.is_cached(project_id):
        use_cache_text.text = (
            "Use cached data stored on the agent to optimize project download"
        )
    else:
        use_cache_text.text = (
            "Cache data on the agent to optimize project download for future trainings"
        )


@select_data_button.click
def select_input_data():
    selected_datasets = set()
    for dataset_id in dataset_selector.get_selected_ids():
        selected_datasets.add(dataset_id)
        for ds in api.dataset.get_nested(project_id=project_id, dataset_id=dataset_id):
            selected_datasets.add(ds.id)
    update_globals(list(selected_datasets))
    update_split_tabs_for_nested_datasets(dataset_ids)
    sly.logger.debug(f"Select data button clicked, selected datasets: {dataset_ids}")
    project_shapes = [
        cls.geometry_type.geometry_name() for cls in project_meta.obj_classes
    ]
    if "bitmap" in project_shapes or "polygon" in project_shapes:
        task_type_select.set_value("instance segmentation")
        models_table_columns = [
            key
            for key in g.seg_models_data[0].keys()
            if key not in ["weights_url", "yaml_config"]
        ]
        models_table_subtitles = [None] * len(models_table_columns)
        models_table_rows = []
        for element in g.seg_models_data:
            models_table_rows.append(list(element.values())[:-2])
        models_table.set_data(
            columns=models_table_columns,
            rows=models_table_rows,
            subtitles=models_table_subtitles,
        )
    elif "graph" in project_shapes:
        task_type_select.set_value("pose estimation")
        models_table_columns = [
            key
            for key in g.pose_models_data[0].keys()
            if key not in ["weights_url", "yaml_config"]
        ]
        models_table_subtitles = [None] * len(models_table_columns)
        models_table_rows = []
        for element in g.pose_models_data:
            models_table_rows.append(list(element.values())[:-2])
        models_table.set_data(
            columns=models_table_columns,
            rows=models_table_rows,
            subtitles=models_table_subtitles,
        )
    select_data_button.loading = True
    dataset_selector.disable()
    use_cache_text.disable()
    classes_table.read_project_from_id(project_id, dataset_ids=dataset_ids)
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
    project_shapes = [
        cls.geometry_type.geometry_name() for cls in project_meta.obj_classes
    ]
    if task_type == "object detection":
        select_classes_button.enable()
        models_table_columns = [
            key
            for key in g.det_models_data[0].keys()
            if key not in ["weights_url", "yaml_config"]
        ]
        models_table_subtitles = [None] * len(models_table_columns)
        models_table_rows = []
        for element in g.det_models_data:
            models_table_rows.append(list(element.values())[:-2])
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
            models_table_columns = [
                key
                for key in g.seg_models_data[0].keys()
                if key not in ["weights_url", "yaml_config"]
            ]
            models_table_subtitles = [None] * len(models_table_columns)
            models_table_rows = []
            for element in g.seg_models_data:
                models_table_rows.append(list(element.values())[:-2])
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
            models_table_columns = [
                key
                for key in g.pose_models_data[0].keys()
                if key not in ["weights_url", "yaml_config"]
            ]
            models_table_subtitles = [None] * len(models_table_columns)
            models_table_rows = []
            for element in g.pose_models_data:
                models_table_rows.append(list(element.values())[:-2])
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
            api, selected_classes, dataset_ids, project_meta, bbox_miss_check_progress
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


@freeze_layers.value_changed
def change_freezing(value):
    if value:
        n_frozen_layers_input_f.show()
    else:
        n_frozen_layers_input_f.hide()


@run_model_benchmark_checkbox.value_changed
def change_model_benchmark(value):
    if value:
        run_speedtest_checkbox.show()
    else:
        run_speedtest_checkbox.hide()


@export_model_switch.value_changed
def change_export_model(value):
    if value:
        export_model_container.show()
    else:
        export_model_container.hide()


@additional_config_radio.value_changed
def change_radio(value):
    if value == "import template from Team Files":
        remote_templates_dir = os.path.join(
            framework_folder, task_type_select.get_value(), "param_templates"
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
        framework_folder, task_type_select.get_value(), "param_templates"
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
        framework_folder, task_type_select.get_value(), "param_templates"
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
    run_model_benchmark_checkbox.disable()
    run_speedtest_checkbox.disable()
    export_model_switch.disable()
    export_onnx_checkbox.disable()
    export_tensorrt_checkbox.disable()
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
    run_model_benchmark_checkbox.enable()
    run_speedtest_checkbox.enable()
    export_model_switch.enable()
    export_onnx_checkbox.enable()
    export_tensorrt_checkbox.enable()
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


@stop_training_button.click
def stop_training_process():
    stop_training_tooltip.loading = True
    sly.logger.info("Stopping training process...")
    g.stop_event.set()


@start_training_button.click
def start_training():
    profiler = MemoryProfiler()

    start_training_button.loading = True

    if g.IN_PROGRESS is True:
        start_training_button.disable()
        return
    g.IN_PROGRESS = True

    task_type = task_type_select.get_value()
    use_cache = use_cache_checkbox.is_checked()

    local_dir = g.root_model_checkpoint_dir
    if task_type == "object detection":
        necessary_geometries = ["rectangle"]
        checkpoint_dir = os.path.join(local_dir, "detect")
        local_artifacts_dir = os.path.join(local_dir, "detect", "train")
        models_data = g.det_models_data
    elif task_type == "pose estimation":
        necessary_geometries = ["graph", "rectangle"]
        checkpoint_dir = os.path.join(local_dir, "pose")
        local_artifacts_dir = os.path.join(local_dir, "pose", "train")
        models_data = g.pose_models_data
    elif task_type == "instance segmentation":
        necessary_geometries = ["bitmap", "polygon"]
        checkpoint_dir = os.path.join(local_dir, "segment")
        local_artifacts_dir = os.path.join(local_dir, "segment", "train")
        models_data = g.seg_models_data

    sly.logger.info(f"Local artifacts dir: {local_artifacts_dir}")

    if os.path.exists(local_artifacts_dir):
        sly.fs.remove_dir(local_artifacts_dir)
    # get number of images in selected datasets
    dataset_infos = [
        api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids
    ]
    n_images = sum([info.images_count for info in dataset_infos])
    download_project(
        api=api,
        project_info=project_info,
        dataset_infos=dataset_infos,
        use_cache=use_cache,
        progress=progress_bar_download_project,
    )

    # remove unselected classes
    selected_classes = classes_table.get_selected_classes()
    try:
        sly.Project.remove_classes_except(
            g.project_dir, classes_to_keep=selected_classes, inplace=True
        )
    except Exception:
        if not use_cache:
            raise
        sly.logger.warn(
            f"Error during classes removing. Will try to re-download project without cache",
            exc_info=True,
        )
        download_project(
            api=api,
            project_info=project_info,
            dataset_infos=dataset_infos,
            use_cache=False,
            progress=progress_bar_download_project,
        )
        sly.Project.remove_classes_except(
            g.project_dir, classes_to_keep=selected_classes, inplace=True
        )

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
    # extract geometry configs
    if task_type == "pose estimation":
        nodes_order = []
        cls2config = {}
        total_config = {"nodes": {}, "edges": []}
        for cls in project_meta.obj_classes:
            if (
                cls.name in selected_classes
                and cls.geometry_type.geometry_name() == "graph"
            ):
                g.keypoints_classes.append(cls.name)
                geometry_config = cls.geometry_config
                cls2config[cls.name] = geometry_config
                for key, value in geometry_config["nodes"].items():
                    label = value["label"]
                    g.node_id2label[key] = label
                    if label not in total_config["nodes"]:
                        total_config["nodes"][label] = value
                        nodes_order.append(label)
        if len(total_config["nodes"]) == 17:
            total_config["nodes"][uuid.uuid4().hex[:6]] = {
                "label": "fictive",
                "color": [0, 0, 255],
                "loc": [0, 0],
            }
        g.keypoints_template = total_config

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
            train_val_split._project_id = None
            train_val_split.update_data()
            train_set, val_set = train_val_split.get_splits()
            val_part = len(val_set) / (len(train_set) + len(val_set))
            new_val_count = round(n_images_after * val_part)
            if new_val_count < 1:
                sly.app.show_dialog(
                    title="An error occured",
                    description="Val split length is 0 after ignoring images. Please check your data",
                    status="error",
                )
                raise ValueError(
                    "Val split length is 0 after ignoring images. Please check your data"
                )
    # split the data
    try:
        train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        train_val_split._project_id = None
        train_val_split.update_data()
        train_set, val_set = train_val_split.get_splits()
        train_val_split._project_id = project_id
    except Exception:
        if not use_cache:
            raise
        sly.logger.warning(
            "Error during data splitting. Will try to re-download project without cache",
            exc_info=True,
        )
        download_project(
            api=api,
            project_info=project_info,
            dataset_infos=dataset_infos,
            use_cache=False,
            progress=progress_bar_download_project,
        )
        train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        train_val_split._project_id = None
        train_val_split.update_data()
        train_set, val_set = train_val_split.get_splits()
        train_val_split._project_id = project_id
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

    profiler.measure("Preprocessing finished")
    
    # download model
    weights_type = model_tabs.get_active_tab()

    def download_monitor(monitor, api: sly.Api, progress: sly.Progress):
        value = monitor
        if progress.total == 0:
            progress.set(value, monitor.len, report=False)
        else:
            progress.set_current_value(value, report=False)
        weights_pbar.update(progress.current)

    file_info = None

    g.stop_event = threading.Event()

    if weights_type == "Pretrained models":
        selected_index = models_table.get_selected_row_index()
        selected_dict = models_data[selected_index]
        weights_url = selected_dict["weights_url"]
        model_filename = weights_url.split("/")[-1]
        selected_model_name = selected_dict["Model"].split(" ")[0]  # "YOLOv8n-det"
        if select_train_mode.get_value() == "Finetune mode":
            pretrained = True
            weights_dst_path = os.path.join(g.app_data_dir, model_filename)
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
            model = CustomYOLO(weights_dst_path, stop_event=g.stop_event)
        else:
            yaml_config = selected_dict["yaml_config"]
            pretrained = False
            model = CustomYOLO(yaml_config, stop_event=g.stop_event)
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
        model = CustomYOLO(weights_dst_path, stop_event=g.stop_event)
        try:
            # get model_name from previous training
            selected_model_name = model.ckpt["sly_metadata"]["model_name"]
        except Exception:
            selected_model_name = "custom_model.pt"

    profiler.measure("Model downloaded")

    # ---------------------------------- Init And Set Workflow Input --------------------------------- #
    w.workflow_input(api, project_info, file_info)
    # ----------------------------------------------- - ---------------------------------------------- #

    # add callbacks to model
    def on_train_batch_end(trainer):
        with open("train_batches.txt", "w") as file:
            file.write("train batch end")

    def freeze_callback(trainer):
        model = trainer.model
        num_freeze = n_frozen_layers_input.get_value()
        print(f"Freezing {num_freeze} layers...")
        freeze = [f"model.{x}." for x in range(num_freeze)]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f"freezing {k}")
                v.requires_grad = False
        print(f"{num_freeze} layers were frozen")

    model.add_callback("on_train_batch_end", on_train_batch_end)
    if freeze_layers.is_switched():
        model.add_callback("on_train_start", freeze_callback)

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
    remote_images_path = (
        f"{framework_folder}/{task_type}/{project_info.name}/images/{g.app_session_id}/"
    )

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
            grid_plot.add_scalar(
                "precision & recall/precision", float(precision), int(x)
            )
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
        batch = f"train_batch{x-1}.jpg"
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
            if sly.is_production():
                train_batches_gallery.append(tf_train_batches_info.storage_path)
            else:
                train_batches_gallery.append(tf_train_batches_info.full_storage_url)
            if x == 1:
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

    profiler.measure("Prepared training")

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

    profiler.measure("Watchers started")

    def stop_on_batch_end_if_needed(trainer_validator, *args, **kwargs):
        app_is_stopped = app.is_stopped()
        not_ready_for_api_calls = False
        if not app_is_stopped:
            not_ready_for_api_calls = (
                api.app.is_ready_for_api_calls(g.app_session_id) is False
            )
        if (
            (app_is_stopped or not_ready_for_api_calls)
            and sly.is_production()
            and server_address != "https://demo.supervisely.com"
        ):
            print(f"Stopping the train process...")
            trainer_validator.stop = True
            raise app.StopException("This error is expected.")

    # model.add_callback("on_train_batch_end", stop_on_batch_end_if_needed)
    # model.add_callback("on_val_batch_end", stop_on_batch_end_if_needed)

    def train_model():
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
            profiler=profiler,
            **additional_params,
        )

    stop_training_tooltip.show()

    train_thread = threading.Thread(target=train_model, args=())
    profiler.measure("Starting training")
    train_thread.start()
    train_thread.join()
    profiler.measure("Training finished")

    # if app.is_stopped():
    #     print("Stopping the app...")
    #     sly.fs.remove_dir(g.app_data_dir)
    #     watcher.running = False
    #     app.stop()
    #     return
    if not app.is_stopped():
        progress_bar_iters.hide()
        progress_bar_epochs.hide()
    watcher.running = False

    # visualize model predictions
    making_training_vis_f.show()
    for i in range(4):
        val_batch_labels_id, val_batch_preds_id = None, None
        labels_path = os.path.join(local_artifacts_dir, f"val_batch{i}_labels.jpg")
        if os.path.exists(labels_path):
            remote_labels_path = os.path.join(
                remote_images_path, f"val_batch{i}_labels.jpg"
            )
            tf_labels_info = api.file.upload(team_id, labels_path, remote_labels_path)
            if sly.is_production():
                val_batch_labels_id = val_batches_gallery.append(
                    image_url=tf_labels_info.storage_path,
                    title="labels",
                )
            else:
                val_batch_labels_id = val_batches_gallery.append(
                    image_url=tf_labels_info.full_storage_url,
                    title="labels",
                )
        preds_path = os.path.join(local_artifacts_dir, f"val_batch{i}_pred.jpg")
        if os.path.exists(preds_path):
            remote_preds_path = os.path.join(
                remote_images_path, f"val_batch{i}_pred.jpg"
            )
            tf_preds_info = api.file.upload(team_id, preds_path, remote_preds_path)
            if sly.is_production():
                val_batch_preds_id = val_batches_gallery.append(
                    image_url=tf_preds_info.storage_path,
                    title="predictions",
                )
            else:
                val_batch_preds_id = val_batches_gallery.append(
                    image_url=tf_preds_info.full_storage_url,
                    title="predictions",
                )
        if val_batch_labels_id and val_batch_preds_id:
            val_batches_gallery.sync_images([val_batch_labels_id, val_batch_preds_id])
        if i == 0:
            val_batches_gallery_f.show()

    stop_training_tooltip.loading = False
    stop_training_tooltip.hide()

    # visualize additional training results
    confusion_matrix_path = os.path.join(
        local_artifacts_dir, "confusion_matrix_normalized.png"
    )
    if os.path.exists(confusion_matrix_path):
        remote_confusion_matrix_path = os.path.join(
            remote_images_path, "confusion_matrix_normalized.png"
        )
        tf_confusion_matrix_info = api.file.upload(
            team_id, confusion_matrix_path, remote_confusion_matrix_path
        )
        if not app.is_stopped():
            if sly.is_production():
                additional_gallery.append(tf_confusion_matrix_info.storage_path)
            else:
                additional_gallery.append(tf_confusion_matrix_info.full_storage_url)
            additional_gallery_f.show()
    pr_curve_path = os.path.join(local_artifacts_dir, "PR_curve.png")
    if os.path.exists(pr_curve_path):
        remote_pr_curve_path = os.path.join(remote_images_path, "PR_curve.png")
        tf_pr_curve_info = api.file.upload(team_id, pr_curve_path, remote_pr_curve_path)
        if not app.is_stopped():
            if sly.is_production():
                additional_gallery.append(tf_pr_curve_info.storage_path)
            else:
                additional_gallery.append(tf_pr_curve_info.full_storage_url)
    f1_curve_path = os.path.join(local_artifacts_dir, "F1_curve.png")
    if os.path.exists(f1_curve_path):
        remote_f1_curve_path = os.path.join(remote_images_path, "F1_curve.png")
        tf_f1_curve_info = api.file.upload(team_id, f1_curve_path, remote_f1_curve_path)
        if not app.is_stopped():
            if sly.is_production():
                additional_gallery.append(tf_f1_curve_info.storage_path)
            else:
                additional_gallery.append(tf_f1_curve_info.full_storage_url)
    box_f1_curve_path = os.path.join(local_artifacts_dir, "BoxF1_curve.png")
    if os.path.exists(box_f1_curve_path):
        remote_box_f1_curve_path = os.path.join(remote_images_path, "BoxF1_curve.png")
        tf_box_f1_curve_info = api.file.upload(
            team_id, box_f1_curve_path, remote_box_f1_curve_path
        )
        if not app.is_stopped():
            if sly.is_production():
                additional_gallery.append(tf_box_f1_curve_info.storage_path)
            else:
                additional_gallery.append(tf_box_f1_curve_info.full_storage_url)
    pose_f1_curve_path = os.path.join(local_artifacts_dir, "PoseF1_curve.png")
    if os.path.exists(pose_f1_curve_path):
        remote_pose_f1_curve_path = os.path.join(remote_images_path, "PoseF1_curve.png")
        tf_pose_f1_curve_info = api.file.upload(
            team_id, pose_f1_curve_path, remote_pose_f1_curve_path
        )
        if not app.is_stopped():
            if sly.is_production():
                additional_gallery.append(tf_pose_f1_curve_info.storage_path)
            else:
                additional_gallery.append(tf_pose_f1_curve_info.full_storage_url)
    mask_f1_curve_path = os.path.join(local_artifacts_dir, "MaskF1_curve.png")
    if os.path.exists(mask_f1_curve_path):
        remote_mask_f1_curve_path = os.path.join(remote_images_path, "MaskF1_curve.png")
        tf_mask_f1_curve_info = api.file.upload(
            team_id, mask_f1_curve_path, remote_mask_f1_curve_path
        )
        if not app.is_stopped():
            if sly.is_production():
                additional_gallery.append(tf_mask_f1_curve_info.storage_path)
            else:
                additional_gallery.append(tf_mask_f1_curve_info.full_storage_url)

    profiler.measure("Visualization finished")

    making_training_vis_f.hide()
    # rename best checkpoint file
    uploading_artefacts_f.show()
    if not os.path.isfile(watch_file):
        sly.logger.warning(
            "The file with results does not exist, training was not completed successfully."
        )
        app.stop()
        return
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
        weights_filepath = os.path.join(local_artifacts_dir, "weights", best_filename)
        weights_dict = torch.load(weights_filepath)
        if len(cls2config.keys()) == 1:
            geometry_config = list(cls2config.values())[0]
            weights_dict["geometry_config"] = geometry_config
        elif len(cls2config.keys()) > 1:
            weights_dict["geometry_config"] = {
                "configs": cls2config,
                "nodes_order": nodes_order,
            }
        torch.save(weights_dict, weights_filepath)

    # add model name to saved weights
    def add_sly_metadata_to_ckpt(ckpt_path):
        loaded = torch.load(ckpt_path, map_location="cpu")
        loaded["sly_metadata"] = {"model_name": selected_model_name}
        torch.save(loaded, ckpt_path)

    best_path = os.path.join(local_artifacts_dir, "weights", best_filename)
    last_path = os.path.join(local_artifacts_dir, "weights", "last.pt")
    if os.path.exists(best_path):
        add_sly_metadata_to_ckpt(best_path)
    if os.path.exists(last_path):
        add_sly_metadata_to_ckpt(last_path)

    # save link to app ui
    app_url = f"/apps/sessions/{g.app_session_id}"
    app_link_path = os.path.join(local_artifacts_dir, "open_app.lnk")
    with open(app_link_path, "w") as text_file:
        print(app_url, file=text_file)

    profiler.measure("Results saved")

    # Exporting to ONNX / TensorRT
    if export_model_switch.is_switched() and os.path.exists(best_path):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            export_weights(best_path, selected_model_name, model_benchmark_pbar)
        except Exception as e:
            sly.logger.error(f"Error during model export: {e}")
        finally:
            model_benchmark_pbar.hide()
            profiler.measure("Exporting weigths finished")

    # upload training artifacts to team files
    upload_artifacts_dir = os.path.join(
        framework_folder,
        task_type_select.get_value(),
        project_info.name,
        str(g.app_session_id),
    )

    if not app.is_stopped():

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
            remote_artifacts_dir = api.file.upload_directory(
                team_id=sly.env.team_id(),
                local_dir=local_artifacts_dir,
                remote_dir=upload_artifacts_dir,
                progress_size_cb=progress_cb,
            )
        progress_bar_upload_artifacts.hide()
    else:
        sly.logger.info(
            "Uploading training artifacts before stopping the app... (progress bar is disabled)"
        )
        remote_artifacts_dir = api.file.upload_directory(
            team_id=sly.env.team_id(),
            local_dir=local_artifacts_dir,
            remote_dir=upload_artifacts_dir,
        )
        sly.logger.info("Training artifacts uploaded successfully")
    
    profiler.measure("Artifacts uploaded")

    uploading_artefacts_f.hide()
    remote_weights_dir = yolov8_artifacts.get_weights_path(remote_artifacts_dir)

    # ------------------------------------- Model Benchmark ------------------------------------- #
    model_benchmark_done = False
    if run_model_benchmark_checkbox.is_checked():
        try:
            profiler.measure("Starting benchmark")
            if task_type in [TaskType.INSTANCE_SEGMENTATION, TaskType.OBJECT_DETECTION]:
                sly.logger.info(
                    f"Creating the report for the best model: {best_filename!r}"
                )
                creating_report_f.show()
                model_benchmark_pbar.show()
                model_benchmark_pbar(
                    message="Starting Model Benchmark evaluation...", total=1
                )

                # 0. Serve trained model
                m = YOLOv8ModelMB(
                    model_dir=local_artifacts_dir + "/weights",
                    use_gui=False,
                    custom_inference_settings=os.path.join(
                        root_source_path, "serve", "custom_settings.yaml"
                    ),
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                sly.logger.info(f"Using device: {device}")

                checkpoint_path = os.path.join(remote_weights_dir, best_filename)
                deploy_params = dict(
                    device=device,
                    runtime=sly.nn.inference.RuntimeType.PYTORCH,
                    model_source="Custom models",
                    task_type=task_type,
                    checkpoint_name=best_filename,
                    checkpoint_url=checkpoint_path,
                )
                m._load_model(deploy_params)
                m.serve()
                m.model.overrides["verbose"] = False
                session = SessionJSON(api, session_url="http://localhost:8000")
                sly.fs.remove_dir(g.app_data_dir + "/benchmark")

                # 1. Init benchmark (todo: auto-detect task type)
                benchmark_dataset_ids = None
                benchmark_images_ids = None
                train_dataset_ids = None
                train_images_ids = None

                split_method = train_val_split._content.get_active_tab()

                if split_method == "Based on datasets":
                    if hasattr(train_val_split._val_ds_select, "get_selected_ids"):
                        benchmark_dataset_ids = (
                            train_val_split._val_ds_select.get_selected_ids()
                        )
                        train_dataset_ids = (
                            train_val_split._train_ds_select.get_selected_ids()
                        )
                    else:
                        benchmark_dataset_ids = [
                            ds_name_to_id[d]
                            for d in train_val_split._val_ds_select.get_value()
                        ]
                        train_dataset_ids = [
                            ds_name_to_id[d]
                            for d in train_val_split._train_ds_select.get_value()
                        ]
                else:

                    def get_image_infos_by_split(split: list):
                        ds_infos_dict = {
                            ds_info.name: ds_info for ds_info in dataset_infos
                        }
                        image_names_per_dataset = {}
                        for item in split:
                            image_names_per_dataset.setdefault(
                                item.dataset_name, []
                            ).append(item.name)
                        image_infos = []
                        for (
                            dataset_name,
                            image_names,
                        ) in image_names_per_dataset.items():
                            if "/" in dataset_name:
                                dataset_name = dataset_name.split("/")[-1]
                            ds_info = ds_infos_dict[dataset_name]
                            for batched_names in sly.batched(image_names, 200):
                                batch = api.image.get_list(
                                        ds_info.id,
                                        filters=[
                                            {
                                                "field": "name",
                                                "operator": "in",
                                                "value": batched_names,
                                            }
                                        ],
                                        force_metadata_for_links=False,
                                    )
                                image_infos.extend(batch)
                        return image_infos

                    val_image_infos = get_image_infos_by_split(val_set)
                    train_image_infos = get_image_infos_by_split(train_set)
                    benchmark_images_ids = [img_info.id for img_info in val_image_infos]
                    train_images_ids = [img_info.id for img_info in train_image_infos]

                if task_type == TaskType.OBJECT_DETECTION:
                    bm = ObjectDetectionBenchmark(
                        api,
                        project_info.id,
                        output_dir=g.app_data_dir + "/benchmark",
                        gt_dataset_ids=benchmark_dataset_ids,
                        gt_images_ids=benchmark_images_ids,
                        progress=model_benchmark_pbar,
                        progress_secondary=model_benchmark_pbar_secondary,
                        classes_whitelist=selected_classes,
                    )
                elif task_type == TaskType.INSTANCE_SEGMENTATION:
                    bm = InstanceSegmentationBenchmark(
                        api,
                        project_info.id,
                        output_dir=g.app_data_dir + "/benchmark",
                        gt_dataset_ids=benchmark_dataset_ids,
                        gt_images_ids=benchmark_images_ids,
                        progress=model_benchmark_pbar,
                        progress_secondary=model_benchmark_pbar_secondary,
                        classes_whitelist=selected_classes,
                    )
                else:
                    raise ValueError(
                        f"Model benchmark for task type {task_type} is not implemented (coming soon)"
                    )

                train_info = {
                    "app_session_id": g.app_session_id,
                    "train_dataset_ids": train_dataset_ids,
                    "train_images_ids": train_images_ids,
                    "images_count": len(train_set),
                }
                bm.train_info = train_info

                # 2. Run inference
                bm.run_inference(session)

                # 3. Pull results from the server
                gt_project_path, dt_project_path = bm.download_projects(
                    save_images=False
                )

                # 4. Evaluate
                bm._evaluate(gt_project_path, dt_project_path)
                bm._dump_eval_inference_info(bm._eval_inference_info)

                # 5. Upload evaluation results
                eval_res_dir = get_eval_results_dir_name(
                    api, g.app_session_id, project_info
                )
                bm.upload_eval_results(eval_res_dir + "/evaluation/")

                # 6. Speed test
                if run_speedtest_checkbox.is_checked():
                    bm.run_speedtest(session, project_info.id)
                    model_benchmark_pbar_secondary.hide()
                    bm.upload_speedtest_results(eval_res_dir + "/speedtest/")

                # 7. Prepare visualizations, report and upload
                bm.visualize()
                remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")
                report = bm.upload_report_link(remote_dir)

                # 8. UI updates
                benchmark_report_template = api.file.get_info_by_path(
                    sly.env.team_id(), remote_dir + "template.vue"
                )
                model_benchmark_done = True
                creating_report_f.hide()
                model_benchmark_report.set(benchmark_report_template)
                model_benchmark_report.show()
                model_benchmark_pbar.hide()
                sly.logger.info(
                    f"Predictions project name: {bm.dt_project_info.name}. Workspace_id: {bm.dt_project_info.workspace_id}"
                )
        except Exception as e:
            sly.logger.error(f"Model benchmark failed. {repr(e)}", exc_info=True)
            creating_report_f.hide()
            model_benchmark_pbar.hide()
            model_benchmark_pbar_secondary.hide()
            try:
                if bm.dt_project_info:
                    api.project.remove(bm.dt_project_info.id)
            except Exception as e2:
                pass

        profiler.measure("Benchmark finished")

    # ----------------------------------------------- - ---------------------------------------------- #

    # ------------------------------------- Set Workflow Outputs ------------------------------------- #
    if not model_benchmark_done:
        benchmark_report_template = None
    w.workflow_output(
        api,
        model_filename,
        remote_artifacts_dir,
        best_filename,
        benchmark_report_template,
    )
    # ----------------------------------------------- - ---------------------------------------------- #

    if not app.is_stopped():
        file_info = api.file.get_info_by_path(
            sly.env.team_id(), remote_artifacts_dir + "/results.csv"
        )
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
        # card_train_progress.collapse()

    # upload sly_metadata.json
    yolov8_artifacts.generate_metadata(
        app_name=yolov8_artifacts.app_name,
        task_id=g.app_session_id,
        artifacts_folder=remote_artifacts_dir,
        weights_folder=remote_weights_dir,
        weights_ext=yolov8_artifacts.weights_ext,
        project_name=project_info.name,
        task_type=task_type,
        config_path=None,
    )

    # delete app data since it is no longer needed
    sly.fs.remove_dir(g.app_data_dir)
    sly.fs.silent_remove("train_batches.txt")
    # set task output
    sly.output.set_directory(remote_artifacts_dir)
    # stop app
    app.stop()
    profiler.measure("App stopped")
    profiler.upload(api, sly.env.team_id(), upload_artifacts_dir)


@server.post("/auto_train")
def auto_train(request: Request):
    sly.logger.info("Starting automatic training session...")
    profiler = MemoryProfiler()
    state = request.state.state

    if "yaml_string" in state:
        state = yaml.safe_load(state["yaml_string"])

    project_id = state["project_id"]
    task_type = state["task_type"]
    use_cache = state.get("use_cache", True)

    select_data_button.hide()

    local_dir = g.root_model_checkpoint_dir
    if task_type == "object detection":
        necessary_geometries = ["rectangle"]
        checkpoint_dir = os.path.join(local_dir, "detect")
        local_artifacts_dir = os.path.join(local_dir, "detect", "train")
        models_data = g.det_models_data
    elif task_type == "pose estimation":
        necessary_geometries = ["graph", "rectangle"]
        checkpoint_dir = os.path.join(local_dir, "pose")
        local_artifacts_dir = os.path.join(local_dir, "pose", "train")
        models_data = g.pose_models_data
    elif task_type == "instance segmentation":
        necessary_geometries = ["bitmap", "polygon"]
        checkpoint_dir = os.path.join(local_dir, "segment")
        local_artifacts_dir = os.path.join(local_dir, "segment", "train")
        models_data = g.seg_models_data

    task_type_select.set_value(task_type)
    models_table_columns = [
        key
        for key in models_data[0].keys()
        if key not in ["weights_url", "yaml_config"]
    ]
    models_table_subtitles = [None] * len(models_table_columns)
    models_table_rows = []
    for element in models_data:
        models_table_rows.append(list(element.values())[:-2])
    models_table.set_data(
        columns=models_table_columns,
        rows=models_table_rows,
        subtitles=models_table_subtitles,
    )

    sly.logger.info(f"Local artifacts dir: {local_artifacts_dir}")

    if os.path.exists(local_artifacts_dir):
        sly.fs.remove_dir(local_artifacts_dir)

    # get number of images in selected datasets
    if "dataset_ids" not in state:
        dataset_infos = api.dataset.get_list(project_id)
        dataset_ids = [dataset_info.id for dataset_info in dataset_infos]
    else:
        dataset_ids = state["dataset_ids"]
        dataset_infos = [
            api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids
        ]

    dataset_selector.disable()
    classes_table.read_project_from_id(project_id, dataset_ids=dataset_ids)
    classes_table.select_all()
    selected_classes = classes_table.get_selected_classes()
    _update_select_classes_button(selected_classes)

    stepper.set_active_step(1)
    card_classes.unlock()
    card_classes.uncollapse()

    n_images = sum([info.images_count for info in dataset_infos])
    download_project(
        api=api,
        project_info=project_info,
        dataset_infos=dataset_infos,
        use_cache=use_cache,
        progress=progress_bar_download_project,
    )

    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    selected_classes = [cls.name for cls in project_meta.obj_classes]

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
    stepper.set_active_step(2)
    card_train_val_split.unlock()
    card_train_val_split.uncollapse()

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
    # extract geometry configs
    if task_type == "pose estimation":
        nodes_order = []
        cls2config = {}
        total_config = {"nodes": {}, "edges": []}
        for cls in project_meta.obj_classes:
            if (
                cls.name in selected_classes
                and cls.geometry_type.geometry_name() == "graph"
            ):
                g.keypoints_classes.append(cls.name)
                geometry_config = cls.geometry_config
                cls2config[cls.name] = geometry_config
                for key, value in geometry_config["nodes"].items():
                    label = value["label"]
                    g.node_id2label[key] = label
                    if label not in total_config["nodes"]:
                        total_config["nodes"][label] = value
                        nodes_order.append(label)
        if len(total_config["nodes"]) == 17:
            total_config["nodes"][uuid.uuid4().hex[:6]] = {
                "label": "fictive",
                "color": [0, 0, 255],
                "loc": [0, 0],
            }
        g.keypoints_template = total_config

    # transfer project to detection task if necessary
    if task_type == "object detection":
        sly.Project.to_detection_task(g.project_dir, inplace=True)
    # split the data
    try:
        train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        train_set, val_set = train_val_split.get_splits()
    except Exception:
        if not use_cache:
            raise
        sly.logger.warning(
            "Error during data splitting. Will try to re-download project without cache",
            exc_info=True,
        )
        download_project(
            api=api,
            project_info=project_info,
            dataset_infos=dataset_infos,
            use_cache=False,
            progress=progress_bar_download_project,
        )
        train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        train_set, val_set = train_val_split.get_splits()
    verify_train_val_sets(train_set, val_set)

    train_val_split.disable()
    unlabeled_images_select.disable()
    split_done.show()
    split_data_button.hide()
    resplit_data_button.show()
    stepper.set_active_step(3)
    card_model_selection.unlock()
    card_model_selection.uncollapse()

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

    profiler.measure("Preprocessing finished")

    # download model
    weights_type = "Pretrained models"

    def download_monitor(monitor, api: sly.Api, progress: sly.Progress):
        value = monitor
        if progress.total == 0:
            progress.set(value, monitor.len, report=False)
        else:
            progress.set_current_value(value, report=False)
        weights_pbar.update(progress.current)

    file_info = None

    g.stop_event = threading.Event()

    if weights_type == "Pretrained models":
        if "model" not in state:
            selected_index = 0
        else:
            selected_model = state["model"]
            found_index = False
            for i, element in enumerate(models_data):
                if selected_model in element.values():
                    selected_index = i
                    found_index = True
                    break
            if not found_index:
                sly.logger.info(
                    f"Unable to find requested model: {selected_model}, switching to default"
                )
                selected_index = 0
        selected_dict = models_data[selected_index]
        weights_url = selected_dict["weights_url"]
        model_filename = weights_url.split("/")[-1]
        selected_model_name = selected_dict["Model"].split(" ")[0]  # "YOLOv8n-det"
        if "train_mode" in state and state["train_mode"] == "finetune":
            pretrained = True
            weights_dst_path = os.path.join(g.app_data_dir, model_filename)
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
            model = CustomYOLO(weights_dst_path, stop_event=g.stop_event)
        else:
            yaml_config = selected_dict["yaml_config"]
            pretrained = False
            model = CustomYOLO(yaml_config, stop_event=g.stop_event)
    # elif weights_type == "Custom models":
    #     custom_link = model_path_input.get_value()
    #     model_filename = "custom_model.pt"
    #     weights_dst_path = os.path.join(g.app_data_dir, model_filename)
    #     file_info = api.file.get_info_by_path(sly.env.team_id(), custom_link)
    #     if file_info is None:
    #         raise FileNotFoundError(f"Custon model file not found: {custom_link}")
    #     file_size = file_info.sizeb
    #     progress = sly.Progress(
    #         message="",
    #         total_cnt=file_size,
    #         is_size=True,
    #     )
    #     progress_cb = partial(download_monitor, api=api, progress=progress)
    #     with progress_bar_download_model(
    #         message="Downloading model weights...",
    #         total=file_size,
    #         unit="bytes",
    #         unit_scale=True,
    #     ) as weights_pbar:
    #         api.file.download(
    #             team_id=sly.env.team_id(),
    #             remote_path=custom_link,
    #             local_save_path=weights_dst_path,
    #             progress_cb=progress_cb,
    #         )
    #     pretrained = True
    #     model = CustomYOLO(weights_dst_path, stop_event=g.stop_event)
    #     try:
    #         # get model_name from previous training
    #         selected_model_name = model.ckpt["sly_metadata"]["model_name"]
    #     except Exception:
    #         selected_model_name = "custom_model.pt"

    model_select_done.show()
    model_not_found_text.hide()
    select_model_button.hide()
    model_tabs.disable()
    models_table.disable()
    model_path_input.disable()
    reselect_model_button.show()
    stepper.set_active_step(4)
    card_train_params.unlock()
    card_train_params.uncollapse()

    profiler.measure("Model downloaded")

    # ---------------------------------- Init And Set Workflow Input --------------------------------- #
    w.workflow_input(api, project_info, file_info)
    # ----------------------------------------------- - ---------------------------------------------- #

    # add callbacks to model
    def on_train_batch_end(trainer):
        with open("train_batches.txt", "w") as file:
            file.write("train batch end")

    def freeze_callback(trainer):
        model = trainer.model
        num_freeze = n_frozen_layers_input.get_value()
        print(f"Freezing {num_freeze} layers...")
        freeze = [f"model.{x}." for x in range(num_freeze)]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f"freezing {k}")
                v.requires_grad = False
        print(f"{num_freeze} layers were frozen")

    model.add_callback("on_train_batch_end", on_train_batch_end)
    if freeze_layers.is_switched():
        model.add_callback("on_train_start", freeze_callback)

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
    remote_images_path = (
        f"{framework_folder}/{task_type}/{project_info.name}/images/{g.app_session_id}/"
    )

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
            grid_plot.add_scalar(
                "precision & recall/precision", float(precision), int(x)
            )
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
        batch = f"train_batch{x-1}.jpg"
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
            train_batches_gallery.append(tf_train_batches_info.storage_path)
            if x == 1:
                train_batches_gallery_f.show()

    watcher = Watcher(
        watch_file,
        on_results_file_changed,
        progress_bar_epochs(
            message="Epochs:", total=state.get("n_epochs", n_epochs_input.get_value())
        ),
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

    profiler.measure("Prepared training")

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
    
    profiler.measure("Watchers started")

    # extract training hyperparameters
    n_epochs = state.get("n_epochs", n_epochs_input.get_value())
    patience = state.get("patience", patience_input.get_value())
    batch_size = state.get("batch_size", batch_size_input.get_value())
    image_size = state.get("input_image_size", image_size_input.get_value())
    n_workers = state.get("n_workers", n_workers_input.get_value())
    optimizer = state.get("optimizer", select_optimizer.get_value())
    lr0 = state.get("lr0", additional_params["lr0"])
    lrf = state.get("lrf", additional_params["lr0"])
    momentum = state.get("momentum", additional_params["momentum"])
    weight_decay = state.get("weight_decay", additional_params["weight_decay"])
    warmup_epochs = state.get("warmup_epochs", additional_params["warmup_epochs"])
    warmup_momentum = state.get("warmup_momentum", additional_params["warmup_momentum"])
    warmup_bias_lr = state.get("warmup_bias_lr", additional_params["warmup_bias_lr"])
    amp = state.get("amp", additional_params["amp"])
    hsv_h = state.get("hsv_h", additional_params["hsv_h"])
    hsv_s = state.get("hsv_s", additional_params["hsv_s"])
    hsv_v = state.get("hsv_v", additional_params["hsv_v"])
    degrees = state.get("degrees", additional_params["degrees"])
    translate = state.get("translate", additional_params["translate"])
    scale = state.get("scale", additional_params["scale"])
    shear = state.get("shear", additional_params["shear"])
    perspective = state.get("perspective", additional_params["perspective"])
    flipud = state.get("flipud", additional_params["flipud"])
    fliplr = state.get("fliplr", additional_params["fliplr"])
    mosaic = state.get("mosaic", additional_params["mosaic"])
    mixup = state.get("mixup", additional_params["mixup"])
    copy_paste = state.get("copy_paste", additional_params["copy_paste"])

    if pretrained:
        select_train_mode.set_value(value="Finetune mode")
    else:
        select_train_mode.set_value(value="Scratch mode")

    n_epochs_input.value = n_epochs
    patience_input.value = patience
    batch_size_input.value = batch_size
    image_size_input.value = image_size
    select_optimizer.set_value(value=optimizer)
    n_workers_input.value = n_workers

    additional_params_text = train_settings_editor.get_text()
    ryaml = ruamel.yaml.YAML()
    additional_params_dict = ryaml.load(additional_params_text)
    additional_params_dict["lr0"] = lr0
    additional_params_dict["lrf"] = lrf
    additional_params_dict["momentum"] = momentum
    additional_params_dict["weight_decay"] = weight_decay
    additional_params_dict["warmup_epochs"] = warmup_epochs
    additional_params_dict["warmup_momentum"] = warmup_momentum
    additional_params_dict["warmup_bias_lr"] = warmup_bias_lr
    additional_params_dict["amp"] = amp
    additional_params_dict["hsv_h"] = hsv_h
    additional_params_dict["hsv_s"] = hsv_s
    additional_params_dict["hsv_v"] = hsv_v
    additional_params_dict["degrees"] = degrees
    additional_params_dict["translate"] = translate
    additional_params_dict["scale"] = scale
    additional_params_dict["shear"] = shear
    additional_params_dict["perspective"] = perspective
    additional_params_dict["flipud"] = flipud
    additional_params_dict["fliplr"] = fliplr
    if task_type == "pose estimation":
        additional_params_dict["fliplr"] = 0.0
    additional_params_dict["mixup"] = mixup
    additional_params_dict["copy_paste"] = copy_paste
    stream = io.BytesIO()
    ryaml.dump(additional_params_dict, stream)
    additional_params_str = stream.getvalue()
    additional_params_str = additional_params_str.decode("utf-8")
    train_settings_editor.set_text(additional_params_str)

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
    run_model_benchmark_checkbox.disable()
    run_speedtest_checkbox.disable()
    export_model_switch.disable()
    export_onnx_checkbox.disable()
    export_tensorrt_checkbox.disable()
    train_settings_editor.readonly = True
    stepper.set_active_step(5)
    card_train_progress.unlock()
    card_train_progress.uncollapse()

    def train_model():
        model.train(
            data=data_path,
            project=checkpoint_dir,
            epochs=n_epochs,
            patience=patience,
            batch=batch_size,
            imgsz=image_size,
            save_period=1000,
            device=device,
            workers=n_workers,
            optimizer=optimizer,
            pretrained=pretrained,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            amp=amp,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            mosaic=mosaic,
            mixup=mixup,
            copy_paste=copy_paste,
        )

    stop_training_tooltip.show()

    train_thread = threading.Thread(target=train_model, args=())
    profiler.measure("Starting training")
    train_thread.start()
    train_thread.join()
    profiler.measure("Training finished")
    watcher.running = False
    progress_bar_iters.hide()
    progress_bar_epochs.hide()

    # visualize model predictions
    making_training_vis_f.show()
    # visualize model predictions
    for i in range(4):
        val_batch_labels_id, val_batch_preds_id = None, None
        labels_path = os.path.join(local_artifacts_dir, f"val_batch{i}_labels.jpg")
        if os.path.exists(labels_path):
            remote_labels_path = os.path.join(
                remote_images_path, f"val_batch{i}_labels.jpg"
            )
            tf_labels_info = api.file.upload(team_id, labels_path, remote_labels_path)
            val_batch_labels_id = val_batches_gallery.append(
                image_url=tf_labels_info.storage_path,
                title="labels",
            )
        preds_path = os.path.join(local_artifacts_dir, f"val_batch{i}_pred.jpg")
        if os.path.exists(preds_path):
            remote_preds_path = os.path.join(
                remote_images_path, f"val_batch{i}_pred.jpg"
            )
            tf_preds_info = api.file.upload(team_id, preds_path, remote_preds_path)
            val_batch_preds_id = val_batches_gallery.append(
                image_url=tf_preds_info.storage_path,
                title="predictions",
            )
        if val_batch_labels_id and val_batch_preds_id:
            val_batches_gallery.sync_images([val_batch_labels_id, val_batch_preds_id])
        if i == 0:
            val_batches_gallery_f.show()

    stop_training_tooltip.loading = False
    stop_training_tooltip.hide()

    # visualize additional training results
    confusion_matrix_path = os.path.join(
        local_artifacts_dir, "confusion_matrix_normalized.png"
    )
    if os.path.exists(confusion_matrix_path):
        remote_confusion_matrix_path = os.path.join(
            remote_images_path, "confusion_matrix_normalized.png"
        )
        tf_confusion_matrix_info = api.file.upload(
            team_id, confusion_matrix_path, remote_confusion_matrix_path
        )
        if not app.is_stopped():
            additional_gallery.append(tf_confusion_matrix_info.storage_path)
            additional_gallery_f.show()
    pr_curve_path = os.path.join(local_artifacts_dir, "PR_curve.png")
    if os.path.exists(pr_curve_path):
        remote_pr_curve_path = os.path.join(remote_images_path, "PR_curve.png")
        tf_pr_curve_info = api.file.upload(team_id, pr_curve_path, remote_pr_curve_path)
        if not app.is_stopped():
            additional_gallery.append(tf_pr_curve_info.storage_path)
    f1_curve_path = os.path.join(local_artifacts_dir, "F1_curve.png")
    if os.path.exists(f1_curve_path):
        remote_f1_curve_path = os.path.join(remote_images_path, "F1_curve.png")
        tf_f1_curve_info = api.file.upload(team_id, f1_curve_path, remote_f1_curve_path)
        if not app.is_stopped():
            additional_gallery.append(tf_f1_curve_info.storage_path)
    box_f1_curve_path = os.path.join(local_artifacts_dir, "BoxF1_curve.png")
    if os.path.exists(box_f1_curve_path):
        remote_box_f1_curve_path = os.path.join(remote_images_path, "BoxF1_curve.png")
        tf_box_f1_curve_info = api.file.upload(
            team_id, box_f1_curve_path, remote_box_f1_curve_path
        )
        if not app.is_stopped():
            additional_gallery.append(tf_box_f1_curve_info.storage_path)
    pose_f1_curve_path = os.path.join(local_artifacts_dir, "PoseF1_curve.png")
    if os.path.exists(pose_f1_curve_path):
        remote_pose_f1_curve_path = os.path.join(remote_images_path, "PoseF1_curve.png")
        tf_pose_f1_curve_info = api.file.upload(
            team_id, pose_f1_curve_path, remote_pose_f1_curve_path
        )
        if not app.is_stopped():
            additional_gallery.append(tf_pose_f1_curve_info.storage_path)
    mask_f1_curve_path = os.path.join(local_artifacts_dir, "MaskF1_curve.png")
    if os.path.exists(mask_f1_curve_path):
        remote_mask_f1_curve_path = os.path.join(remote_images_path, "MaskF1_curve.png")
        tf_mask_f1_curve_info = api.file.upload(
            team_id, mask_f1_curve_path, remote_mask_f1_curve_path
        )
        if not app.is_stopped():
            additional_gallery.append(tf_mask_f1_curve_info.storage_path)

    making_training_vis_f.hide()

    profiler.measure("Visualization finished")

    # rename best checkpoint file
    if not os.path.isfile(watch_file):
        sly.logger.warning(
            "The file with results does not exist, training was not completed successfully."
        )
        app.stop()
        return
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
        weights_filepath = os.path.join(local_artifacts_dir, "weights", best_filename)
        weights_dict = torch.load(weights_filepath)
        if len(cls2config.keys()) == 1:
            geometry_config = list(cls2config.values())[0]
            weights_dict["geometry_config"] = geometry_config
        elif len(cls2config.keys()) > 1:
            weights_dict["geometry_config"] = {
                "configs": cls2config,
                "nodes_order": nodes_order,
            }
        torch.save(weights_dict, weights_filepath)

    # add model name to saved weights
    def add_sly_metadata_to_ckpt(ckpt_path):
        loaded = torch.load(ckpt_path, map_location="cpu")
        loaded["sly_metadata"] = {"model_name": selected_model_name}
        torch.save(loaded, ckpt_path)

    best_path = os.path.join(local_artifacts_dir, "weights", best_filename)
    last_path = os.path.join(local_artifacts_dir, "weights", "last.pt")
    if os.path.exists(best_path):
        add_sly_metadata_to_ckpt(best_path)
    if os.path.exists(last_path):
        add_sly_metadata_to_ckpt(last_path)

    # save link to app ui
    app_url = f"/apps/sessions/{g.app_session_id}"
    app_link_path = os.path.join(local_artifacts_dir, "open_app.lnk")
    with open(app_link_path, "w") as text_file:
        print(app_url, file=text_file)

    profiler.measure("Results saved")

    # Exporting to ONNX / TensorRT
    if export_model_switch.is_switched() and os.path.exists(best_path):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            export_weights(best_path, selected_model_name, model_benchmark_pbar)
        except Exception as e:
            sly.logger.error(f"Error during model export: {e}")
        finally:
            model_benchmark_pbar.hide()
            profiler.measure("Exporting weigths finished")

    # upload training artifacts to team files
    upload_artifacts_dir = os.path.join(
        framework_folder,
        task_type_select.get_value(),
        project_info.name,
        str(g.app_session_id),
    )

    if not app.is_stopped():

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
            remote_artifacts_dir = api.file.upload_directory(
                team_id=sly.env.team_id(),
                local_dir=local_artifacts_dir,
                remote_dir=upload_artifacts_dir,
                progress_size_cb=progress_cb,
            )
        progress_bar_upload_artifacts.hide()
    else:
        sly.logger.info(
            "Uploading training artifacts before stopping the app... (progress bar is disabled)"
        )
        remote_artifacts_dir = api.file.upload_directory(
            team_id=sly.env.team_id(),
            local_dir=local_artifacts_dir,
            remote_dir=upload_artifacts_dir,
        )
        sly.logger.info("Training artifacts uploaded successfully")
    
    profiler.measure("Artifacts uploaded")

    remote_weights_dir = yolov8_artifacts.get_weights_path(remote_artifacts_dir)

    # ------------------------------------- Model Benchmark ------------------------------------- #
    model_benchmark_done = False
    if run_model_benchmark_checkbox.is_checked():
        profiler.measure("Starting benchmark")
        try:
            if task_type in [TaskType.INSTANCE_SEGMENTATION, TaskType.OBJECT_DETECTION]:
                sly.logger.info(
                    f"Creating the report for the best model: {best_filename!r}"
                )
                creating_report_f.show()
                model_benchmark_pbar.show()
                model_benchmark_pbar(
                    message="Starting Model Benchmark evaluation...", total=1
                )

                # 0. Serve trained model
                m = YOLOv8ModelMB(
                    model_dir=local_artifacts_dir + "/weights",
                    use_gui=False,
                    custom_inference_settings=os.path.join(
                        root_source_path, "serve", "custom_settings.yaml"
                    ),
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                sly.logger.info(f"Using device: {device}")

                checkpoint_path = os.path.join(remote_weights_dir, best_filename)
                deploy_params = dict(
                    device=device,
                    runtime=sly.nn.inference.RuntimeType.PYTORCH,
                    model_source="Custom models",
                    task_type=task_type,
                    checkpoint_name=best_filename,
                    checkpoint_url=checkpoint_path,
                )
                m._load_model(deploy_params)
                m.serve()
                m.model.overrides["verbose"] = False
                session = SessionJSON(api, session_url="http://localhost:8000")
                sly.fs.remove_dir(g.app_data_dir + "/benchmark")

                # 1. Init benchmark (todo: auto-detect task type)
                benchmark_dataset_ids = None
                benchmark_images_ids = None
                train_dataset_ids = None
                train_images_ids = None

                split_method = train_val_split._content.get_active_tab()

                if split_method == "Based on datasets":
                    benchmark_dataset_ids = (
                        train_val_split._val_ds_select.get_selected_ids()
                    )
                    train_dataset_ids = (
                        train_val_split._train_ds_select.get_selected_ids()
                    )
                else:

                    def get_image_infos_by_split(split: list):
                        ds_infos_dict = {
                            ds_info.name: ds_info for ds_info in dataset_infos
                        }
                        image_names_per_dataset = {}
                        for item in split:
                            image_names_per_dataset.setdefault(
                                item.dataset_name, []
                            ).append(item.name)
                        image_infos = []
                        for (
                            dataset_name,
                            image_names,
                        ) in image_names_per_dataset.items():
                            if "/" in dataset_name:
                                dataset_name = dataset_name.split("/")[-1]
                            ds_info = ds_infos_dict[dataset_name]
                            for batched_names in sly.batched(image_names, 200):
                                batch = api.image.get_list(
                                        ds_info.id,
                                        filters=[
                                            {
                                                "field": "name",
                                                "operator": "in",
                                                "value": batched_names,
                                            }
                                        ],
                                        force_metadata_for_links=False,
                                    )
                                image_infos.extend(batch)
                        return image_infos

                    val_image_infos = get_image_infos_by_split(val_set)
                    train_image_infos = get_image_infos_by_split(train_set)
                    benchmark_images_ids = [img_info.id for img_info in val_image_infos]
                    train_images_ids = [img_info.id for img_info in train_image_infos]

                if task_type == TaskType.OBJECT_DETECTION:
                    bm = ObjectDetectionBenchmark(
                        api,
                        project_info.id,
                        output_dir=g.app_data_dir + "/benchmark",
                        gt_dataset_ids=benchmark_dataset_ids,
                        gt_images_ids=benchmark_images_ids,
                        progress=model_benchmark_pbar,
                        progress_secondary=model_benchmark_pbar_secondary,
                        classes_whitelist=selected_classes,
                    )
                elif task_type == TaskType.INSTANCE_SEGMENTATION:
                    bm = InstanceSegmentationBenchmark(
                        api,
                        project_info.id,
                        output_dir=g.app_data_dir + "/benchmark",
                        gt_dataset_ids=benchmark_dataset_ids,
                        gt_images_ids=benchmark_images_ids,
                        progress=model_benchmark_pbar,
                        progress_secondary=model_benchmark_pbar_secondary,
                        classes_whitelist=selected_classes,
                    )
                else:
                    raise ValueError(
                        f"Model benchmark for task type {task_type} is not implemented (coming soon)"
                    )

                train_info = {
                    "app_session_id": g.app_session_id,
                    "train_dataset_ids": train_dataset_ids,
                    "train_images_ids": train_images_ids,
                    "images_count": len(train_set),
                }
                bm.train_info = train_info

                # 2. Run inference
                bm.run_inference(session)

                # 3. Pull results from the server
                gt_project_path, dt_project_path = bm.download_projects(
                    save_images=False
                )

                # 4. Evaluate
                bm._evaluate(gt_project_path, dt_project_path)
                bm._dump_eval_inference_info(bm._eval_inference_info)

                # 5. Upload evaluation results
                eval_res_dir = get_eval_results_dir_name(
                    api, g.app_session_id, project_info
                )
                bm.upload_eval_results(eval_res_dir + "/evaluation/")

                # 6. Speed test
                if run_speedtest_checkbox.is_checked():
                    bm.run_speedtest(session, project_info.id)
                    model_benchmark_pbar_secondary.hide()
                    bm.upload_speedtest_results(eval_res_dir + "/speedtest/")

                # 7. Prepare visualizations, report and upload
                bm.visualize()
                remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")
                report = bm.upload_report_link(remote_dir)

                # 8. UI updates
                benchmark_report_template = api.file.get_info_by_path(
                    sly.env.team_id(), remote_dir + "template.vue"
                )
                model_benchmark_done = True
                creating_report_f.hide()
                model_benchmark_report.set(benchmark_report_template)
                model_benchmark_report.show()
                model_benchmark_pbar.hide()
                sly.logger.info(
                    f"Predictions project name: {bm.dt_project_info.name}. Workspace_id: {bm.dt_project_info.workspace_id}"
                )
        except Exception as e:
            sly.logger.error(f"Model benchmark failed. {repr(e)}", exc_info=True)
            creating_report_f.hide()
            model_benchmark_pbar.hide()
            model_benchmark_pbar_secondary.hide()
            try:
                if bm.dt_project_info:
                    api.project.remove(bm.dt_project_info.id)
            except Exception as e2:
                pass

        profiler.measure("Benchmark finished")

    # ----------------------------------------------- - ---------------------------------------------- #

    # ------------------------------------- Set Workflow Outputs ------------------------------------- #
    if not model_benchmark_done:
        benchmark_report_template = None
    w.workflow_output(
        api,
        model_filename,
        remote_artifacts_dir,
        best_filename,
        benchmark_report_template,
    )
    # ----------------------------------------------- - ---------------------------------------------- #

    if not app.is_stopped():
        file_info = api.file.get_info_by_path(
            sly.env.team_id(), remote_artifacts_dir + "/results.csv"
        )
        train_artifacts_folder.set(file_info)
        # finish training
        card_train_artifacts.unlock()
        card_train_artifacts.uncollapse()

    # upload sly_metadata.json
    yolov8_artifacts.generate_metadata(
        app_name=yolov8_artifacts.app_name,
        task_id=g.app_session_id,
        artifacts_folder=remote_artifacts_dir,
        weights_folder=remote_weights_dir,
        weights_ext=yolov8_artifacts.weights_ext,
        project_name=project_info.name,
        task_type=task_type,
        config_path=None,
    )

    # delete app data since it is no longer needed
    sly.fs.remove_dir(g.app_data_dir)
    sly.fs.silent_remove("train_batches.txt")
    # set task output
    sly.output.set_directory(remote_artifacts_dir)
    # stop app
    app.stop()
    profiler.measure("App stopped")
    profiler.upload(api, sly.env.team_id(), upload_artifacts_dir)
    return {"result": "successfully finished automatic training session"}


def export_weights(weights_path, selected_model_name, progress: SlyTqdm):
    from src.model_export import export_checkpoint

    checkpoint_info_path = dump_yaml_checkpoint_info(weights_path, selected_model_name)
    pbar = None
    fp16 = export_fp16_switch.is_switched()
    if export_tensorrt_checkbox.is_checked():
        pbar = progress(
            message="Exporting model to TensorRT, this may take some time...", total=1
        )
        export_checkpoint(weights_path, format="engine", fp16=fp16, dynamic=False)
        pbar.update(1)
    if export_onnx_checkbox.is_checked():
        pbar = progress(message="Exporting model to ONNX...", total=1)
        dynamic = not fp16  # dynamic mode is not supported for fp16
        export_checkpoint(weights_path, format="onnx", fp16=fp16, dynamic=dynamic)
        pbar.update(1)


def dump_yaml_checkpoint_info(weights_path, selected_model_name):
    p = r"yolov(\d+)"
    match = re.match(p, selected_model_name.lower())
    architecture = match.group(0) if match else None
    checkpoint_info = {
        "model_name": selected_model_name,
        "architecture": architecture,
    }
    checkpoint_info_path = os.path.join(
        os.path.dirname(weights_path), "checkpoint_info.yaml"
    )
    with open(checkpoint_info_path, "w") as f:
        yaml.dump(checkpoint_info, f)
    return checkpoint_info_path
