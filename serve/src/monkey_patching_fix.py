# Monkey patching to avoid a problem with installing wrong onnxruntime requirements
# issue: https://github.com/ultralytics/ultralytics/issues/5093
def monkey_patching_fix():
    import importlib
    import cv2
    import numpy as np
    import ultralytics.nn.autobackend
    import ultralytics.utils.checks as checks
    check_requirements = checks.check_requirements  # save original function
    def check_requirements_dont_install(*args, **kwargs):
        kwargs["install"] = False
        return check_requirements(*args, **kwargs)
    checks.check_requirements = check_requirements_dont_install
    importlib.reload(ultralytics.nn.autobackend)

    # BoT-SORT's OSNet re-identification code calls cv2.resize directly on
    # NumPy slices. A malformed/out-of-frame annotation can make that slice
    # empty and abort the whole video request. Keep one crop per detection so
    # tracker indices remain aligned, but use a neutral crop for invalid boxes.
    try:
        from supervisely.nn.tracker.botsort.osnet_reid.osnet_reid_interface import (
            OsnetReIDModel,
        )

        def safe_get_crops(self, xyxys, img):
            import torch

            h, w = img.shape[:2]
            crops = []
            for box in xyxys:
                x1, y1, x2, y2 = box.round().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    crop = np.zeros(
                        (self.input_shape[0], self.input_shape[1], 3),
                        dtype=img.dtype if img.size else np.uint8,
                    )
                else:
                    crop = cv2.resize(
                        img[y1:y2, x1:x2], self.input_shape[::-1]
                    )
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = crop.astype(np.float32) / 255.0
                crop = (crop - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                crops.append(torch.from_numpy(crop).permute(2, 0, 1))

            return torch.stack(crops).to(
                self.device,
                dtype=torch.float16 if self.half else torch.float32,
            )

        OsnetReIDModel._get_crops = safe_get_crops
    except ImportError:
        # Tracking is optional for image-only inference.
        pass

    try:
        from supervisely.nn.tracker.botsort_tracker import BotSortTracker
        from supervisely.geometry.graph import GraphNodes

        original_stracks_to_tracks = BotSortTracker._stracks_to_tracks
        original_update = BotSortTracker.update

        def safe_update(self, frame, annotation):
            graph_labels = [
                label
                for label in annotation.labels
                if isinstance(label.geometry, GraphNodes)
            ]
            has_mixed_geometry = graph_labels and len(graph_labels) != len(annotation.labels)
            if has_mixed_geometry:
                # Pose annotations contain a redundant Rectangle plus a
                # GraphNodes label. BoT-SORT requires one geometry type per
                # track ID, so track and return only the graph labels.
                annotation = annotation.clone(labels=graph_labels)
            return original_update(self, frame, annotation)

        def safe_stracks_to_tracks(self, output_stracks, detection_track_map, labels):
            tracks = original_stracks_to_tracks(
                self, output_stracks, detection_track_map, labels
            )
            # The SDK can return a tracker class ID that belongs to a
            # different pose label (for example, GraphNodes for a bbox).
            # VideoFigure validates the geometry against the video object's
            # class, so the original label is the authoritative source.
            for track in tracks:
                label = getattr(track, "original_label", None)
                if label is not None:
                    track.class_name = label.obj_class.name
                    track.class_sly_id = label.obj_class.sly_id
            return tracks

        BotSortTracker._stracks_to_tracks = safe_stracks_to_tracks
        BotSortTracker.update = safe_update
    except ImportError:
        pass
