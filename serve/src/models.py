yolov8_models = [
    {
        "Model": "YOLOv8n-det",
        "Size (pixels)": "640",
        "mAP": "37.3",
        "params (M)": "3.2",
        "FLOPs (B)": "8.7",
        "meta": {
            "taskType": "object detection",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8n.pt",
        },
    },
    {
        "Model": "YOLOv8s-det",
        "Size (pixels)": "640",
        "mAP": "44.9",
        "params (M)": "11.2",
        "FLOPs (B)": "28.6",
        "meta": {
            "taskType": "object detection",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8s.pt",
        },
    },
    {
        "Model": "YOLOv8m-det",
        "Size (pixels)": "640",
        "mAP": "50.2",
        "params (M)": "25.9",
        "FLOPs (B)": "78.9",
        "meta": {
            "taskType": "object detection",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8m.pt",
        },
    },
    {
        "Model": "YOLOv8l-det",
        "Size (pixels)": "640",
        "mAP": "52.9",
        "params (M)": "43.7",
        "FLOPs (B)": "165.2",
        "meta": {
            "taskType": "object detection",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8l.pt",
        },
    },
    {
        "Model": "YOLOv8x-det",
        "Size (pixels)": "640",
        "mAP": "53.9",
        "params (M)": "68.2",
        "FLOPs (B)": "257.8",
        "meta": {
            "taskType": "object detection",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8x.pt",
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
            "taskType": "instance segmentation",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8n-seg.pt",
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
            "taskType": "instance segmentation",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8s-seg.pt",
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
            "taskType": "instance segmentation",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8m-seg.pt",
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
            "taskType": "instance segmentation",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8l-seg.pt",
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
            "taskType": "instance segmentation",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8x-seg.pt",
        },
    },
    {
        "Model": "YOLOv8n-pose",
        "Size (pixels)": "640",
        "mAP": "50.4",
        "params (M)": "3.3",
        "FLOPs (B)": "9.2",
        "meta": {
            "taskType": "pose estimation",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8n-pose.pt",
        },
    },
    {
        "Model": "YOLOv8s-pose",
        "Size (pixels)": "640",
        "mAP": "60.0",
        "params (M)": "11.6",
        "FLOPs (B)": "30.2",
        "meta": {
            "taskType": "pose estimation",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8s-pose.pt",
        },
    },
    {
        "Model": "YOLOv8m-pose",
        "Size (pixels)": "640",
        "mAP": "65.0",
        "params (M)": "26.4",
        "FLOPs (B)": "81.0",
        "meta": {
            "taskType": "pose estimation",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8m-pose.pt",
        },
    },
    {
        "Model": "YOLOv8l-pose",
        "Size (pixels)": "640",
        "mAP": "67.6",
        "params (M)": "44.4",
        "FLOPs (B)": "168.6",
        "meta": {
            "taskType": "pose estimation",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8l-pose.pt",
        },
    },
    {
        "Model": "YOLOv8x-pose",
        "Size (pixels)": "640",
        "mAP": "69.2",
        "params (M)": "69.4",
        "FLOPs (B)": "263.2",
        "meta": {
            "taskType": "pose estimation",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8x-pose.pt",
        },
    },
    {
        "Model": "YOLOv8x-pose-p6",
        "Size (pixels)": "640",
        "mAP": "71.6",
        "params (M)": "99.1",
        "FLOPs (B)": "1066.4",
        "meta": {
            "taskType": "pose estimation",
            "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8x-pose-p6.pt",
        },
    },
]
