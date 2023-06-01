import os
from pathlib import Path
import numpy as np
import yaml
import random
import supervisely as sly
import supervisely.io.env as env
import train.src.globals as g
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
    Image,
    ModelInfo,
    ClassesTable,
    DoneLabel,
    ProjectThumbnail,
    Editor,
    Select,
    Checkbox,
    RadioTabs,
    RadioTable,
    RadioGroup,
    RandomSplitsTable,
    Text,
    FileThumbnail,
)
from train.src.utils import get_train_val_sets, verify_train_val_sets
from train.src.sly_to_yolov8 import transform
from ultralytics import YOLO
import torch


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
random_split_table = RandomSplitsTable(
    items_count=100,
    start_train_percent=80,
    disabled=False,
)
train_val_tabs = RadioTabs(
    titles=["Random", "Based on image tags", "Based on datasets"],
    descriptions=[
        "Shuffle data and split with defined probability",
        "Images must have assigned train and val tag",
        "Select on or several datasets for every split",
    ],
    contents=[
        random_split_table,
        Text("Currently not supported", status="info"),
        Text("Currently not supported", status="info"),
    ],
)
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
        train_val_tabs,
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
model_tabs_contents = [models_table, custom_tab_content]
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
save_period_input = InputNumber(value=1, min=1)
save_period_input_f = Field(
    content=save_period_input, title="Save period", description="Save checkpoint every n epochs"
)
n_workers_input = InputNumber(value=8, min=1)
n_workers_input_f = Field(
    content=n_workers_input,
    title="Number of workers",
    description="Number of worker threads for data loading",
)
train_settings_editor = Editor(language_mode="yaml", height_lines=30)
with open(g.train_params_filepath, "r") as f:
    train_params = f.read()
train_settings_editor.set_text(train_params)
train_settings_editor_f = Field(
    content=train_settings_editor,
    title="Additional configuration",
    description="Tune learning rate, weight decay and other parameters",
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
        save_period_input_f,
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
train_progress_content = Container(
    [start_training_button, progress_bar_download_project, progress_bar_convert_to_yolo]
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
train_artifacts_url = f"files/"
train_artifacts_button = Button(
    text="Training artifacts",
    button_type="success",
    plain=True,
    icon="zmdi zmdi-folder",
    link=train_artifacts_url,
)
card_train_artifacts = Card(
    title="Training artifacts",
    description="Checkpoints, logs and other visualizations",
    content=train_artifacts_button,
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
    )
)


@dataset_selector.value_changed
def on_dataset_selected(new_dataset_ids):
    if new_dataset_ids == []:
        select_data_button.hide()
    elif new_dataset_ids != [] and select_data_button.is_hidden():
        select_data_button.show()
    update_globals(new_dataset_ids)


@select_data_button.click
def select_input_data():
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
        models_table_columns = [key for key in g.det_models_data[0].keys()]
        models_table_subtitles = [None] * len(models_table_columns)
        models_table_rows = []
        for element in g.det_models_data:
            models_table_rows.append(list(element.values()))
        models_table.set_data(
            columns=models_table_columns, rows=models_table_rows, subtitles=models_table_subtitles
        )
    elif task_type == "instance segmentation":
        if "bitmap" not in project_shapes and "polygon" not in project_shapes:
            sly.app.show_dialog(
                title="There are no classes of shape bitmap / polygon in selected project",
                description="Please, change task type or select another project with classes of shape bitmap / polygon",
                status="warning",
            )
        models_table_columns = [key for key in g.seg_models_data[0].keys()]
        models_table_subtitles = [None] * len(models_table_columns)
        models_table_rows = []
        for element in g.seg_models_data:
            models_table_rows.append(list(element.values()))
        models_table.set_data(
            columns=models_table_columns, rows=models_table_rows, subtitles=models_table_subtitles
        )
    elif task_type == "pose estimation":
        if "graph" not in project_shapes:
            sly.app.show_dialog(
                title="There are no classes of shape graph in selected project",
                description="Please, change task type or select another project with classes of shape graph",
                status="warning",
            )
        models_table_columns = [key for key in g.pose_models_data[0].keys()]
        models_table_subtitles = [None] * len(models_table_columns)
        models_table_rows = []
        for element in g.pose_models_data:
            models_table_rows.append(list(element.values()))
        models_table.set_data(
            columns=models_table_columns, rows=models_table_rows, subtitles=models_table_subtitles
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
    train_val_tabs.disable()
    random_split_table.disable()
    unlabeled_images_select.disable()
    split_data_button.hide()
    split_done.show()
    resplit_data_button.show()
    card_model_selection.unlock()
    card_model_selection.uncollapse()


@resplit_data_button.click
def resplit_data():
    train_val_tabs.enable()
    random_split_table.enable()
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
    save_period_input.disable()
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
    save_period_input.enable()
    n_workers_input.enable()
    train_settings_editor.readonly = False


@start_training_button.click
def start_training():
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
    # remove unlabeled images if such option was selected by user
    if unlabeled_images_select.get_value() == "ignore unlabeled images":
        n_images_before = n_images
        sly.Project.remove_items_without_objects(g.project_dir, inplace=True)
        project = sly.Project(g.project_dir, sly.OpenMode.READ)
        n_images_after = project.total_items
        if n_images_after != n_images_after:
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
    split_method = train_val_tabs.get_active_tab()
    split_counts = random_split_table.get_splits_counts()
    train_set, val_set = get_train_val_sets(g.project_dir, split_method, split_counts)
    verify_train_val_sets(train_set, val_set)
    # convert dataset from supervisely to yolo format
    if os.path.exists(g.yolov8_project_dir):
        sly.fs.clean_dir(g.yolov8_project_dir)
    transform(g.project_dir, g.yolov8_project_dir, train_set, val_set, progress_bar_convert_to_yolo)
    # download model
    weights_type = model_tabs.get_active_tab()
    if weights_type == "Pretrained models":
        selected_model = models_table.get_selected_row()[0]
        if selected_model.endswith("det"):
            selected_model = selected_model[:-4]
        if select_train_mode.get_value() == "Finetune mode":
            model_filename = selected_model.lower() + ".pt"
            pretrained = True
        else:
            model_filename = selected_model.lower() + ".yaml"
            pretrained = False
        model = YOLO(model_filename)
    elif weights_type == "Custom models":
        pass
    # get training params
    n_epochs = n_epochs_input.get_value()
    patience = patience_input.get_value()
    batch_size = batch_size_input.get_value()
    image_size = image_size_input.get_value()
    optimizer = select_optimizer.get_value()
    n_workers = n_workers_input.get_value()
    save_period = save_period_input.get_value()
    additional_params = train_settings_editor.get_text()
    additional_params = yaml.safe_load(additional_params)
    lr0 = additional_params["lr0"]
    lrf = additional_params["lrf"]
    momentum = additional_params["momentum"]
    weight_decay = additional_params["weight_decay"]
    warmup_epochs = additional_params["warmup_epochs"]
    warmup_momentum = additional_params["warmup_momentum"]
    warmup_bias_lr = additional_params["warmup_bias_lr"]
    # train model and upload best checkpoints to team files
    device = 0 if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(g.yolov8_project_dir, "data_config.yaml")
    model.train(
        data=data_path,
        epochs=n_epochs,
        patience=patience,
        batch=batch_size,
        imgsz=image_size,
        save_period=save_period,
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
    )
    # upload training artifacts to team files
    # finish training
    start_training_button.loading = False
    start_training_button.disable()
    card_train_artifacts.unlock()
    card_train_artifacts.uncollapse()
    sly.app.show_dialog(
        title="Training completed",
        description="You can find training artifacts in Team Files",
        status="success",
    )
    # delete app data since it is no longer needed
