# Object Detection and Tracking

This project emphasizes providing practical code examples for easy experimentation with different detection models and tracking methods in video. Designed for versatility, it allows users to explore different detection models and tracking techniques for real-time video analysis. These examples are simple yet powerful solution for real-time video analysis.

## Purpose
The main objective of this project is to offer a comprehensive foundation for experimenting with multiple object detection models and tracking algorithms. By enabling easy integration, comparison, and customization, it helps users evaluate performance, fine-tune hyperparameters, and develop robust detection and tracking solutions.
This framework also helps exploring and analyzing the impact of various hyperparameter configurations on object detection and tracking performance. By providing a modular and adaptable implementation, this project enables researchers and enthusiasts to experiment with different settings, compare results, and gain deeper insights into how hyperparameters influence the accuracy, efficiency, and robustness of object detection and tracking systems.

## Features
- **Flexible Detection and Tracking**: Switch between different models and algorithms easily.
- **Customizable**: Adjust hyperparameters to suit specific use cases.
- **Object Tracking**: Implements the SORT algorithm to track the detected people across frames.
- **Visualization**: Displays bounding boxes with unique IDs for each detected and tracked object.

## Installation

To run this project, you need to install the required dependencies. You can install them using the following command:

```bash
pip install -r requirements.txt
```
requirements.txt
```
ultralytics  
opencv-python  
filterpy  
numpy  
matplotlib  
torch  
torchvision  
```

## How to run

1. clone the repository
2. Use your own video for object detection and tracking.
3. Run the detection and tracking script

## How It Works
1. Detection: The selected detection model (e.g., YOLOv8) identifies objects in each video frame and outputs bounding boxes with confidence scores. The project provides code examples demonstrating how to integrate and utilize different detection models.

2. Tracking: Various tracking algorithms (e.g., SORT, DeepSORT) assign unique IDs to detected objects and track their movement across frames. Code samples illustrate how to implement these algorithms and customize them for different use cases.

3. Visualization: Real-time visualization displays bounding boxes, object IDs, and tracking paths, making it easier to analyze object behavior. Example scripts show how to customize visual outputs for enhanced analysis.

## Contributing
Feel free to fork the repository, open issues, and submit pull requests. Contributions are welcome!

## Completion Status
### **Object Detection Algorithms**  
| Algorithm | Description | Implementation Status |
|-----------|-------------|-----------------------|
| **YOLO (You Only Look Once)** | A high-speed, high-accuracy object detection model. Suitable for real-time detection. | ✅ |
| **Faster R-CNN** | A high-accuracy object detection model. Suitable for complex objects. Since it detects objects in images, not videos, frames must be processed separately. |  |
| **SSD (Single Shot MultiBox Detector)** | A lightweight model suitable for real-time detection. | ❌ |
| **EfficientDet** | A model with an excellent balance between accuracy and speed. |  |
| **RetinaNet** | A model with strength in detecting small objects using Focal Loss. |  |

---

### **Object Tracking Algorithms**  
| Algorithm | Description | Implementation Status |
|-----------|-------------|-----------------------|
| **SORT (Simple Online and Realtime Tracking)** | A simple and efficient real-time tracking algorithm. |  |
| **DeepSORT** | An extension of SORT, utilizing deep learning for advanced tracking. |  |
| **ByteTrack** | Improves tracking performance by increasing detection confidence. |  |
| **Kalman Filter** | A classical tracking algorithm using state prediction and estimation. |  |
| **IOU Tracker** | Tracks objects based on Intersection Over Union (IOU) between bounding boxes. |  |
