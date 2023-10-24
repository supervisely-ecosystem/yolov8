import os
import yaml
import supervisely as sly
import src.globals as g
import numpy as np
import math


def _transform_label(class_names, img_size, label: sly.Label, task_type, labels_list):
    if task_type == "object detection":
        class_number = class_names.index(label.obj_class.name)
        rect_geometry = label.geometry.to_bbox()
        center = rect_geometry.center
        x_center = round(center.col / img_size[1], 6)
        y_center = round(center.row / img_size[0], 6)
        width = round(rect_geometry.width / img_size[1], 6)
        height = round(rect_geometry.height / img_size[0], 6)
        result = "{} {} {} {} {}".format(class_number, x_center, y_center, width, height)
    elif task_type == "pose estimation":
        # find corresponding bbox for graph
        graph_center = label.geometry.to_bbox().center
        graph_center = [graph_center.col, graph_center.row]
        boxes_list = [label.geometry for label in labels_list if isinstance(label.geometry, sly.Rectangle)]
        center2box = {}
        for box in boxes_list:
            center2box[f"{box.center.col} {box.center.row}"] = box
        distance2center = {}
        for center in center2box.keys():
            cx, cy = center.split()
            distance = math.dist(graph_center, [int(cx), int(cy)])
            distance2center[distance] = center
        min_distance = min(distance2center.keys())
        box_center = distance2center[min_distance]
        # corresponding bbox for graph is the one with the smallest distance to graph center
        matched_box = center2box[box_center]
        box_x, box_y = box_center.split()
        box_x, box_y = int(box_x), int(box_y)
        box_center = [box_x, box_y]
        if box_center not in g.center_matches.values():
            g.center_matches[f"{graph_center[0]} {graph_center[1]}"] = box_center
            class_number = class_names.index(label.obj_class.name)
            x_center = round(box_center[0] / img_size[1], 6)
            y_center = round(box_center[1] / img_size[0], 6)
            width = round(matched_box.width / img_size[1], 6)
            height = round(matched_box.height / img_size[0], 6)
        # if failed to match graphs and boxes, get box by transforming graph to box
        else:
            sly.logger.warn("Failed to match graphs and boxes, boxes will be created by transforming graphs to boxes")
            class_number = class_names.index(label.obj_class.name)
            rect_geometry = label.geometry.to_bbox()
            center = rect_geometry.center
            x_center = round(center.col / img_size[1], 6)
            y_center = round(center.row / img_size[0], 6)
            width = round(rect_geometry.width / img_size[1], 6)
            height = round(rect_geometry.height / img_size[0], 6)
        graph_nodes = label.geometry.nodes
        keypoints = []
        for node_id in g.keypoints_template["nodes"].keys():
            if node_id in graph_nodes.keys():
                visibility = 2
                graph_node = graph_nodes[node_id]
                point_x = round(graph_node.location.col / img_size[1], 6)
                point_y = round(graph_node.location.row / img_size[0], 6)
            else:
                visibility = 0
                point_x, point_y = 0, 0
            keypoints.extend([point_x, point_y, visibility])
        keypoints_str = " ".join(str(point) for point in keypoints)
        result = f"{class_number} {x_center} {y_center} {width} {height} {keypoints_str}"
    elif task_type == "instance segmentation":
        class_number = class_names.index(label.obj_class.name)
        if type(label.geometry) is sly.Bitmap:
            new_obj_class = sly.ObjClass(label.obj_class.name, sly.Polygon)
            labels = label.convert(new_obj_class)
            if len(labels) == 0:
                return None
            for i, label in enumerate(labels):
                if i == 0:
                    points = label.geometry.exterior_np
                else:
                    points = np.concatenate((points, label.geometry.exterior_np), axis=0)
        else:
            points = label.geometry.exterior_np
        points = np.flip(points, axis=1)
        scaled_points = []
        for point in points:
            scaled_points.append(round(point[0] / img_size[1], 6))
            scaled_points.append(round(point[1] / img_size[0], 6))
        scaled_points_str = " ".join([str(point) for point in scaled_points])
        result = f"{class_number} {scaled_points_str}"
    return result

def _create_data_config(output_dir, meta: sly.ProjectMeta, task_type):
    class_names = []
    class_colors = []
    for obj_class in meta.obj_classes:
        if task_type == "pose estimation" and obj_class.geometry_type.geometry_name() != "graph":
            continue
        class_names.append(obj_class.name)
        class_colors.append(obj_class.color)
    if task_type in ["object detection", "instance segmentation"]:
        data_yaml = {
            "train": os.path.join(output_dir, "images/train"),
            "val": os.path.join(output_dir, "images/val"),
            "labels_train": os.path.join(output_dir, "labels/train"),
            "labels_val": os.path.join(output_dir, "labels/val"),
            "nc": len(class_names),
            "names": class_names,
            "colors": class_colors,
        }
    elif task_type == "pose estimation":
        for obj_class in meta.obj_classes:
            if obj_class.geometry_type.geometry_name() == "graph":
                geometry_config = obj_class.geometry_config
                g.keypoints_template = geometry_config
                n_keypoints = len(geometry_config["nodes"])
                flip_idx = [i for i in range(n_keypoints)]
                break
        data_yaml = {
            "train": os.path.join(output_dir, "images/train"),
            "val": os.path.join(output_dir, "images/val"),
            "labels_train": os.path.join(output_dir, "labels/train"),
            "labels_val": os.path.join(output_dir, "labels/val"),
            "kpt_shape": [n_keypoints, 3],
            "flip_idx": flip_idx,
            "names": class_names,
        }
    sly.fs.mkdir(data_yaml["train"])
    sly.fs.mkdir(data_yaml["val"])
    sly.fs.mkdir(data_yaml["labels_train"])
    sly.fs.mkdir(data_yaml["labels_val"])

    config_path = os.path.join(output_dir, "data_config.yaml")
    with open(config_path, "w") as f:
        _ = yaml.dump(data_yaml, f, default_flow_style=None)

    return data_yaml


def _transform_annotation(ann, class_names, save_path, task_type):
    yolov8_ann = []
    for label in ann.labels:
        if label.obj_class.name in class_names:
            if task_type == "pose estimation" and isinstance(label.geometry, sly.Rectangle):
                continue
            transformed_label = _transform_label(class_names, ann.img_size, label, task_type, ann.labels)
            if transformed_label:
                yolov8_ann.append(transformed_label)

    with open(save_path, "w") as file:
        file.write("\n".join(yolov8_ann))

    if len(yolov8_ann) == 0:
        return True
    return False


def _transform_set(set_name, data_yaml, project_meta, items, progress_cb, task_type):
    res_images_dir = data_yaml[set_name]
    res_labels_dir = data_yaml[f"labels_{set_name}"]
    classes_names = data_yaml["names"]

    used_names = set()
    with progress_cb(message=f"Converting {set_name} set to YOLOv8 format...", total=len(items)) as pbar:
        for batch in sly.batched(items, batch_size=max(int(len(items) / 50), 10)):
            for item in batch:
                sly.logger.debug(f"Converting image located at {item.img_path} to supervisely format...")
                ann = sly.Annotation.load_json_file(item.ann_path, project_meta)
                _item_name = sly._utils.generate_free_name(used_names, sly.fs.get_file_name(item.name))
                used_names.add(_item_name)

                _ann_name = f"{_item_name}.txt"
                _img_name = f"{_item_name}{sly.fs.get_file_ext(item.img_path)}"

                save_ann_path = os.path.join(res_labels_dir, _ann_name)
                _transform_annotation(ann, classes_names, save_ann_path, task_type)
                save_img_path = os.path.join(res_images_dir, _img_name)
                sly.fs.copy_file(item.img_path, save_img_path)
                pbar.update()


def transform(
    sly_project_dir,
    yolov8_output_dir,
    train_set,
    val_set,
    progress_cb,
    task_type="object detection",
):
    project = sly.Project(sly_project_dir, sly.OpenMode.READ)
    data_yaml = _create_data_config(yolov8_output_dir, project.meta, task_type)

    _transform_set("train", data_yaml, project.meta, train_set, progress_cb, task_type)
    _transform_set("val", data_yaml, project.meta, val_set, progress_cb, task_type)
