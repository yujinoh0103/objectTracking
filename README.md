# Object Detection and Tracking

This project detects and tracks people in videos using YOLOv8 for object detection and SORT (Simple Online and Realtime Tracking) for tracking the detected objects across video frames. It is a simple yet powerful solution for real-time video analysis.

## Purpose
The primary objective of this project is to serve as a foundational framework for exploring and analyzing the impact of various hyperparameter configurations on object detection and tracking performance. By providing a modular and adaptable implementation, this project enables researchers and enthusiasts to experiment with different settings, compare results, and gain deeper insights into how hyperparameters influence the accuracy, efficiency, and robustness of object detection and tracking systems.

## Features
- **Person Detection**: Uses YOLOv8 to detect people in video frames.
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
```

## How to run

1. clone the repository
2. Download a sample video or use your own video for object detection and tracking.
3. Run the detection and tracking script

## How It Works
1. Object Detection with YOLOv8
YOLOv8 is a state-of-the-art object detection model that detects various objects, including people, in each video frame.
The model outputs bounding boxes and the confidence score for each detected object.
2. Object Tracking with SORT
SORT is a simple and efficient tracking algorithm that assigns a unique ID to each detected object and tracks it across frames based on its position.
The algorithm uses Kalman filtering and the Hungarian algorithm for data association.
3. Visualization
The detected and tracked objects are displayed with bounding boxes and unique IDs.
The tracking path can be visualized by drawing lines connecting the object's previous positions.
Results
The model tracks the movement of people in the video and maintains a consistent ID for each person across frames. The result is shown in the video, where each detected person is assigned a unique ID, and their movement is tracked over time.

## Contributing
Feel free to fork the repository, open issues, and submit pull requests. Contributions are welcome!
