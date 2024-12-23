yolov8_models = [
    {
        "Model": "YOLOv8n-det (COCO)",
        "Size (pixels)": "640",
        "mAP": "37.3",
        "params (M)": "3.2",
        "FLOPs (B)": "8.7",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8n.pt",
        },
    },
    {
        "Model": "YOLOv8n-det (Open Images V7)",
        "Size (pixels)": "640",
        "mAP": "18.4",
        "params (M)": "3.5",
        "FLOPs (B)": "10.5",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-oiv7.pt",
        },
    },
    {
        "Model": "YOLOv8s-det (COCO)",
        "Size (pixels)": "640",
        "mAP": "44.9",
        "params (M)": "11.2",
        "FLOPs (B)": "28.6",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8s.pt",
        },
    },
    {
        "Model": "YOLOv8s-det (Open Images V7)",
        "Size (pixels)": "640",
        "mAP": "27.7",
        "params (M)": "11.4",
        "FLOPs (B)": "29.7",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-oiv7.pt",
        },
    },
    {
        "Model": "YOLOv8m-det (COCO)",
        "Size (pixels)": "640",
        "mAP": "50.2",
        "params (M)": "25.9",
        "FLOPs (B)": "78.9",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8m.pt",
        },
    },
    {
        "Model": "YOLOv8m-det (Open Images V7)",
        "Size (pixels)": "640",
        "mAP": "33.6",
        "params (M)": "26.2",
        "FLOPs (B)": "80.6",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-oiv7.pt",
        },
    },
    {
        "Model": "YOLOv8l-det (COCO)",
        "Size (pixels)": "640",
        "mAP": "52.9",
        "params (M)": "43.7",
        "FLOPs (B)": "165.2",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8l.pt",
        },
    },
    {
        "Model": "YOLOv8l-det (Open Images V7)",
        "Size (pixels)": "640",
        "mAP": "34.9",
        "params (M)": "44.1",
        "FLOPs (B)": "167.4",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-oiv7.pt",
        },
    },
    {
        "Model": "YOLOv8x-det (COCO)",
        "Size (pixels)": "640",
        "mAP": "53.9",
        "params (M)": "68.2",
        "FLOPs (B)": "257.8",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8x.pt",
        },
    },
    {
        "Model": "YOLOv8x-det (Open Images V7)",
        "Size (pixels)": "640",
        "mAP": "36.3",
        "params (M)": "68.7",
        "FLOPs (B)": "260.6",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-oiv7.pt",
        },
    },
    {
        "Model": "YOLOv8n-seg",
        "Size (pixels)": "640",
        "mAP (box)": "36.7",
        "mAP (mask)": "30.5",
        "params (M)": "3.4",
        "FLOPs (B)": "12.6",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8n-seg.pt",
        },
    },
    {
        "Model": "YOLOv8s-seg",
        "Size (pixels)": "640",
        "mAP (box)": "44.6",
        "mAP (mask)": "36.8",
        "params (M)": "11.8",
        "FLOPs (B)": "42.6",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8s-seg.pt",
        },
    },
    {
        "Model": "YOLOv8m-seg",
        "Size (pixels)": "640",
        "mAP (box)": "49.9",
        "mAP (mask)": "40.8",
        "params (M)": "27.3",
        "FLOPs (B)": "110.2",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8m-seg.pt",
        },
    },
    {
        "Model": "YOLOv8l-seg",
        "Size (pixels)": "640",
        "mAP (box)": "52.3",
        "mAP (mask)": "42.6",
        "params (M)": "46.0",
        "FLOPs (B)": "220.5",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8l-seg.pt",
        },
    },
    {
        "Model": "YOLOv8x-seg",
        "Size (pixels)": "640",
        "mAP (box)": "53.4",
        "mAP (mask)": "43.4",
        "params (M)": "71.8",
        "FLOPs (B)": "344.1",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8x-seg.pt",
        },
    },
    {
        "Model": "YOLOv8n-pose",
        "Size (pixels)": "640",
        "mAP": "50.4",
        "params (M)": "3.3",
        "FLOPs (B)": "9.2",
        "meta": {
            "task_type": "pose estimation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8n-pose.pt",
        },
    },
    {
        "Model": "YOLOv8s-pose",
        "Size (pixels)": "640",
        "mAP": "60.0",
        "params (M)": "11.6",
        "FLOPs (B)": "30.2",
        "meta": {
            "task_type": "pose estimation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8s-pose.pt",
        },
    },
    {
        "Model": "YOLOv8m-pose",
        "Size (pixels)": "640",
        "mAP": "65.0",
        "params (M)": "26.4",
        "FLOPs (B)": "81.0",
        "meta": {
            "task_type": "pose estimation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8m-pose.pt",
        },
    },
    {
        "Model": "YOLOv8l-pose",
        "Size (pixels)": "640",
        "mAP": "67.6",
        "params (M)": "44.4",
        "FLOPs (B)": "168.6",
        "meta": {
            "task_type": "pose estimation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8l-pose.pt",
        },
    },
    {
        "Model": "YOLOv8x-pose",
        "Size (pixels)": "640",
        "mAP": "69.2",
        "params (M)": "69.4",
        "FLOPs (B)": "263.2",
        "meta": {
            "task_type": "pose estimation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8x-pose.pt",
        },
    },
    {
        "Model": "YOLOv8x-pose-p6",
        "Size (pixels)": "640",
        "mAP": "71.6",
        "params (M)": "99.1",
        "FLOPs (B)": "1066.4",
        "meta": {
            "task_type": "pose estimation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/YOLOv8x-pose-p6.pt",
        },
    },
    {
        "Model": "YOLOv9c-det",
        "Size (pixels)": "640",
        "mAP": "53.0",
        "params (M)": "25.5",
        "FLOPs (B)": "102.8",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt",
        },
    },
    {
        "Model": "YOLOv9e-det",
        "Size (pixels)": "640",
        "mAP": "55.6",
        "params (M)": "58.1",
        "FLOPs (B)": "192.5",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt",
        },
    },
    {
        "Model": "YOLOv9c-seg",
        "Size (pixels)": "640",
        "mAP (box)": "52.4",
        "mAP (mask)": "42.2",
        "params (M)": "27.9",
        "FLOPs (B)": "159.4",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c-seg.pt",
        },
    },
    {
        "Model": "YOLOv9e-seg",
        "Size (pixels)": "640",
        "mAP (box)": "55.1",
        "mAP (mask)": "44.3",
        "params (M)": "60.5",
        "FLOPs (B)": "248.4",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e-seg.pt",
        },
    },
    {
        "Model": "YOLOv10n-det",
        "Size (pixels)": "640",
        "mAP": "39.5",
        "params (M)": "2.3",
        "FLOPs (B)": "6.7",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt",
        },
    },
    {
        "Model": "YOLOv10s-det",
        "Size (pixels)": "640",
        "mAP": "46.8",
        "params (M)": "7.2",
        "FLOPs (B)": "21.6",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt",
        },
    },
    {
        "Model": "YOLOv10m-det",
        "Size (pixels)": "640",
        "mAP": "51.3",
        "params (M)": "15.4",
        "FLOPs (B)": "59.1",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt",
        },
    },
    {
        "Model": "YOLOv10l-det",
        "Size (pixels)": "640",
        "mAP": "53.4",
        "params (M)": "24.4",
        "FLOPs (B)": "120.3",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt",
        },
    },
    {
        "Model": "YOLOv10x-det",
        "Size (pixels)": "640",
        "mAP": "54.4",
        "params (M)": "29.5",
        "FLOPs (B)": "160.4",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt",
        },
    },
    {
        "Model": "YOLO11n-det",
        "Size (pixels)": "640",
        "mAP": "39.5",
        "params (M)": "2.6",
        "FLOPs (B)": "6.5",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        },
    },
    {
        "Model": "YOLO11s-det",
        "Size (pixels)": "640",
        "mAP": "47.0",
        "params (M)": "9.4",
        "FLOPs (B)": "21.5",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        },
    },
    {
        "Model": "YOLO11m-det",
        "Size (pixels)": "640",
        "mAP": "51.5",
        "params (M)": "20.1",
        "FLOPs (B)": "68.0",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        },
    },
    {
        "Model": "YOLO11l-det",
        "Size (pixels)": "640",
        "mAP": "53.4",
        "params (M)": "25.3",
        "FLOPs (B)": "86.9",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
        },
    },
    {
        "Model": "YOLO11x-det",
        "Size (pixels)": "640",
        "mAP": "54.7",
        "params (M)": "56.9",
        "FLOPs (B)": "194.9",
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        },
    },
    {
        "Model": "YOLO11n-pose",
        "Size (pixels)": "640",
        "mAP": "50.0",
        "params (M)": "2.9",
        "FLOPs (B)": "7.6",
        "meta": {
            "task_type": "pose estimation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
        },
    },
    {
        "Model": "YOLO11s-pose",
        "Size (pixels)": "640",
        "mAP": "58.9",
        "params (M)": "9.9",
        "FLOPs (B)": "23.2",
        "meta": {
            "task_type": "pose estimation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt",
        },
    },
    {
        "Model": "YOLO11m-pose",
        "Size (pixels)": "640",
        "mAP": "64.9",
        "params (M)": "20.9",
        "FLOPs (B)": "71.7",
        "meta": {
            "task_type": "pose estimation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt",
        },
    },
    {
        "Model": "YOLO11l-pose",
        "Size (pixels)": "640",
        "mAP": "66.1",
        "params (M)": "26.2",
        "FLOPs (B)": "90.7",
        "meta": {
            "task_type": "pose estimation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt",
        },
    },
    {
        "Model": "YOLO11x-pose",
        "Size (pixels)": "640",
        "mAP": "69.5",
        "params (M)": "58.8",
        "FLOPs (B)": "203.3",
        "meta": {
            "task_type": "pose estimation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt",
        },
    },
    {
        "Model": "YOLO11n-seg",
        "Size (pixels)": "640",
        "mAP (box)": "38.9",
        "mAP (mask)": "32.0",
        "params (M)": "2.9",
        "FLOPs (B)": "10.4",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        },
    },
    {
        "Model": "YOLO11s-seg",
        "Size (pixels)": "640",
        "mAP (box)": "46.6",
        "mAP (mask)": "37.8",
        "params (M)": "10.1",
        "FLOPs (B)": "35.5",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
        },
    },
    {
        "Model": "YOLO11m-seg",
        "Size (pixels)": "640",
        "mAP (box)": "51.5",
        "mAP (mask)": "41.5",
        "params (M)": "22.4",
        "FLOPs (B)": "123.3",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt",
        },
    },
    {
        "Model": "YOLO11l-seg",
        "Size (pixels)": "640",
        "mAP (box)": "53.4",
        "mAP (mask)": "42.9",
        "params (M)": "27.6",
        "FLOPs (B)": "142.2",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt",
        },
    },
    {
        "Model": "YOLO11x-seg",
        "Size (pixels)": "640",
        "mAP (box)": "54.7",
        "mAP (mask)": "43.8",
        "params (M)": "62.1",
        "FLOPs (B)": "319.0",
        "meta": {
            "task_type": "instance segmentation",
            "weights_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt",
        },
    },
]
