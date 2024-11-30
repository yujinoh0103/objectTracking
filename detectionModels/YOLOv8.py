import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # YOLOv8 model (small version)

# Open the video file
video_path = 'samplevideo.mov'  # Video file path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the video ends

    # Perform object detection using YOLOv8
    results = model(frame)

    # Access detected boxes from results
    # results.boxes contains the bounding box information
    for result in results[0].boxes:  # Iterate through all detected boxes
        x1, y1, x2, y2 = map(int, result.xyxy)  # Bounding box coordinates (x1, y1, x2, y2)
        confidence = result.conf  # Confidence score
        class_id = int(result.cls)  # Class ID

        # Detect persons (person class has ID 0 in the COCO dataset)
        if class_id == 0:  # When a person is detected (class 0 in COCO)
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bounding box
            cv2.putText(frame, f"Person: {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the result on the screen
    cv2.imshow('YOLOv8 - Object Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
