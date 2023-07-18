import supervisely as sly
import os
from dotenv import load_dotenv
from ultralytics import YOLO
import torch
import cv2

# load credentials to enable API usage
team_id = 111 # pass your team id
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()
# pass checkpoint path in Team Files
remote_weights_path = "/yolov8_train/object detection/my_dataset/111/weights/best_101.pt" # pass your checkpoint path
# download checkpoint from Team Files
local_weights_path = os.path.join(os.getcwd(), "weights.pt")
if not os.path.exists(local_weights_path):
    api.file.download(
        team_id=team_id,
        remote_path=remote_weights_path,
        local_save_path=local_weights_path,
    )
# load model
model = YOLO(local_weights_path)
# define device
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
# load image
image_path = "input_image.jpg" # pass your image path
input_image = sly.image.read(image_path)
input_image = input_image[:, :, ::-1]
input_height, input_width = input_image.shape[:2]
# pass input image to model
predictions = model(
    source=input_image,
    conf=0.25,
    iou=0.7,
    half=False,
    device=device,
    max_det=300,
    agnostic_nms=False,
)
# visualize predictions
predictions_plotted = predictions[0].plot()
cv2.imwrite("predictions.jpg", predictions_plotted)