import os
import yaml
import supervisely as sly


def _transform_label(class_names, img_size, label: sly.Label):
    class_number = class_names.index(label.obj_class.name)
    rect_geometry = label.geometry.to_bbox()
    center = rect_geometry.center
    x_center = round(center.col / img_size[1], 6)
    y_center = round(center.row / img_size[0], 6)
    width = round(rect_geometry.width / img_size[1], 6)
    height = round(rect_geometry.height / img_size[0], 6)
    result = "{} {} {} {} {}".format(class_number, x_center, y_center, width, height)
    return result


def _create_data_config(output_dir, meta: sly.ProjectMeta):
    class_names = []
    class_colors = []
    for obj_class in meta.obj_classes:
        class_names.append(obj_class.name)
        class_colors.append(obj_class.color)

    data_yaml = {
        "train": os.path.join(output_dir, "images/train"),
        "val": os.path.join(output_dir, "images/val"),
        "labels_train": os.path.join(output_dir, "labels/train"),
        "labels_val": os.path.join(output_dir, "labels/val"),
        "nc": len(class_names),
        "names": class_names,
        "colors": class_colors,
    }
    sly.fs.mkdir(data_yaml["train"])
    sly.fs.mkdir(data_yaml["val"])
    sly.fs.mkdir(data_yaml["labels_train"])
    sly.fs.mkdir(data_yaml["labels_val"])

    config_path = os.path.join(output_dir, "data_config.yaml")
    with open(config_path, "w") as f:
        _ = yaml.dump(data_yaml, f, default_flow_style=None)

    return data_yaml


def _transform_annotation(ann, class_names, save_path):
    yolov8_ann = []
    for label in ann.labels:
        if label.obj_class.name in class_names:
            yolov8_ann.append(_transform_label(class_names, ann.img_size, label))

    with open(save_path, "w") as file:
        file.write("\n".join(yolov8_ann))

    if len(yolov8_ann) == 0:
        return True
    return False


def _transform_set(set_name, data_yaml, project_meta, items, progress_cb):
    res_images_dir = data_yaml[set_name]
    res_labels_dir = data_yaml[f"labels_{set_name}"]
    classes_names = data_yaml["names"]

    used_names = set()
    with progress_cb(
        message=f"Converting {set_name} set to YOLOv8 format...", total=len(items)
    ) as pbar:
        for batch in sly.batched(items, batch_size=max(int(len(items) / 50), 10)):
            for item in batch:
                ann = sly.Annotation.load_json_file(item.ann_path, project_meta)
                _item_name = sly._utils.generate_free_name(
                    used_names, sly.fs.get_file_name(item.name)
                )
                used_names.add(_item_name)

                _ann_name = f"{_item_name}.txt"
                _img_name = f"{_item_name}{sly.fs.get_file_ext(item.img_path)}"

                save_ann_path = os.path.join(res_labels_dir, _ann_name)
                _transform_annotation(ann, classes_names, save_ann_path)
                save_img_path = os.path.join(res_images_dir, _img_name)
                sly.fs.copy_file(item.img_path, save_img_path)
                pbar.update()


def transform(sly_project_dir, yolov8_output_dir, train_set, val_set, progress_cb):
    project = sly.Project(sly_project_dir, sly.OpenMode.READ)
    data_yaml = _create_data_config(yolov8_output_dir, project.meta)

    _transform_set("train", data_yaml, project.meta, train_set, progress_cb)
    _transform_set("val", data_yaml, project.meta, val_set, progress_cb)
