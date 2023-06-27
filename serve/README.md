
<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/yolov8/assets/12828725/3593ebe9-cdd8-4265-8217-f5781b6fb860"/>  

# Serve YOLOv8

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a> •
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

This app allows to apply YOLOv8 models to images and videos. App supports to use both models pretrained on COCO and models trained on custom datasets.

# How To Run

## Pretrained models

**Step 1.** Select task type, pretrained model architecture and press the **Serve** button. Please, note that list of models in the table below depends on selected task type

![yolov8_pretrained_models](https://github.com/supervisely-ecosystem/yolov8/assets/91027877/2a15502d-8fb6-4059-afac-808ad938dd61)

**Step 2.** Wait for the model to deploy

![yolov8_deployed](https://github.com/supervisely-ecosystem/yolov8/assets/91027877/da175901-2667-4d4c-a8dd-5b0d94c4919b)

## Custom models

Copy model file path from Team Files and select task type:

https://github.com/supervisely-ecosystem/yolov8/assets/91027877/d92ac00e-bfa7-448d-bfd5-599b4ca3e415




