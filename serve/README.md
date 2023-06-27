
<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/yolov8/assets/12828725/3593ebe9-cdd8-4265-8217-f5781b6fb860"/>  

# Serve YOLOv8

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#example-apply-yolov8-to-image-in-labeling-tool">Example: apply YOLOv8 to image in labeling tool</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/yolov8/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov8)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/yolov8/serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/yolov8/serve.png)](https://supervise.ly)

</div>

# Overview

YOLOv8 is a powerful neural network architecture which provides both decent accuracy of predictions and high speed of inference. In comparison to YOLOv5, YOLOv8 uses anchor-free head (it allows to speed up non-max suppression (NMS) process), new backbone and new loss functions.

This app allows to apply YOLOv8 models to images and videos. App supports both models pretrained on COCO and models trained on custom datasets. Supported task types are object detection, instance segmentation and pose estimation.

# How To Run

## Pretrained models

**Step 1.** Select task type, pretrained model architecture and press the **Serve** button. Please, note that list of models in the table below depends on selected task type

![yolov8_pretrained_models](https://github.com/supervisely-ecosystem/yolov8/assets/91027877/2a15502d-8fb6-4059-afac-808ad938dd61)

**Step 2.** Wait for the model to deploy

![yolov8_deployed](https://github.com/supervisely-ecosystem/yolov8/assets/91027877/da175901-2667-4d4c-a8dd-5b0d94c4919b)

## Custom models

Copy model file path from Team Files and select task type:

https://user-images.githubusercontent.com/91027877/249001911-d92ac00e-bfa7-448d-bfd5-599b4ca3e415.mp4

# Example: apply YOLOv8 to image in labeling tool

Run **NN Image Labeling** app, connect to YOLOv8 app session, and click on "Apply model to image", or if you want to apply model only to the region within the bounding box, select the bbox and click on "Apply model to ROI":

https://user-images.githubusercontent.com/91027877/249003695-a1c0e6bb-8783-448f-86c0-0d8a4eccfae0.mp4

If you want to change model specific inference settings while working with the model in image labeling interface, go to **inference** tab in the settings section of **Apps** window, and change the parameters:

https://user-images.githubusercontent.com/91027877/249004380-e8a4758b-0356-4efc-a6cf-da146e0d3266.mp4

# Related apps

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployed NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>

- [Apply NN to Videos Project](https://ecosystem.supervise.ly/apps/apply-nn-to-videos-project) - app allows to label your videos using served Supervisely models.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://imgur.com/LDo8K1A.png" height="70px" margin-bottom="20px" />

# Acknowledgment

This app is based on the great work `YOLOv8` ([github](https://github.com/ultralytics/ultralytics)). ![GitHub Org's stars](https://img.shields.io/github/stars/ultralytics/ultralytics?style=social)




