# YOLOv8 - ONNX Inference Examples

This folder contains examples of how to use the YOLOv8 ONNX model for inference.

## Detection demo

The `detection_demo.py` script demonstrates how to use the YOLOv8 ONNX model for object detection. The script takes an image as input and outputs the image with bounding boxes around detected objects.
You will need to install the required packages to run the script. You can install them using the following command:

```bash
pip install -r requirements.txt
```

To run the script, use the following command:

```bash
python detection_demo.py --model <path_to_onnx_model_file> --img <path_to_input_image> --output <path_to_output_image>
```

Default values for the arguments are:
--model `models/yolov8.onnx`
--img `iamges/image_01.jpg`
--output `output.jpg`

Optional arguments:
--conf `0.5` - confidence threshold
--iou `0.5` - IoU threshold
--visualize - flag to visualize the output


## Segmentation demo

The `segmentation_demo.py` script demonstrates how to use the YOLOv8 ONNX model for object detection and segmentation. The script takes an image as input and outputs the image with bounding boxes around detected objects and the segmented masks.
You will need to install the required packages to run the script. You can install them using the following command:

```bash
pip install -r requirements.txt
```
To run the script, use the following command:

```bash
python segmentation_demo.py --model <path_to_onnx_model_file> --img <path_to_input_image> --output <path_to_output_image>
```

Default values for the arguments are:
--model `models/yolov8.onnx`
--img `iamges/image_01.jpg`
--output `output.jpg`

Optional arguments:
--conf `0.25` - confidence threshold
--iou `0.45` - IoU threshold
--visualize - flag to visualize the output
