_TASK_TYPE_GEOMETRIES = {
    "object detection": None,
    "instance segmentation": {"bitmap", "polygon"},
    "pose estimation": {"graph"},
}

_TASK_TYPE_CLASS_HINTS = {
    "object detection": "object class",
    "instance segmentation": "bitmap or polygon class",
    "pose estimation": "graph class",
}


def resolve_training_class_names(obj_classes, task_type):
    """Return supported class names or explain how to fix an empty selection."""
    if task_type not in _TASK_TYPE_GEOMETRIES:
        raise ValueError(f"Unsupported task type: {task_type!r}.")

    supported_geometries = _TASK_TYPE_GEOMETRIES[task_type]
    class_names = [
        obj_class.name
        for obj_class in obj_classes
        if supported_geometries is None
        or obj_class.geometry_type.geometry_name() in supported_geometries
    ]
    if not class_names:
        class_hint = _TASK_TYPE_CLASS_HINTS[task_type]
        raise ValueError(
            f"Project has no classes compatible with {task_type!r}. "
            f"Add at least one {class_hint} or choose another task type."
        )
    return class_names
