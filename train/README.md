
<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/yolov8/assets/12828725/2849410e-4922-4d23-93a7-2385d6d75426"/>  

# Train YOLOv8

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/yolov8/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov8)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/yolov8/train.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/yolov8/train.png)](https://supervise.ly)

</div>

# Overview

YOLOv8 is a powerful neural network architecture which provides both decent accuracy of predictions and high speed of inference. In comparison to YOLOv5, YOLOv8 uses anchor-free head (it allows to speed up non-max suppression (NMS) process), new backbone and new loss functions.

This app allows to train YOLOv8 model on selected dataset. You can define model checkpoint, data split method, training hyperparameters, data augmentation and many other features related to model training. App supports both models pretrained on COCO and models trained on custom datasets. Supported task types are object detection, instance segmentation and pose estimation.

# How To Run

Select images project, select GPU device in "Agent" field, click on RUN button:

https://user-images.githubusercontent.com/91027877/249008934-293b3176-d5f3-4edb-9816-15bffd3bb869.mp4

# Screenshot

![train_yolov8_screen](https://user-images.githubusercontent.com/91027877/249011925-4c4cb749-a04d-4613-bdf5-8ab26ec4efaa.png)


# Acknowledgment

This app is based on the great work `YOLOv8` ([github](https://github.com/ultralytics/ultralytics)). ![GitHub Org's stars](https://img.shields.io/github/stars/ultralytics/ultralytics?style=social)
