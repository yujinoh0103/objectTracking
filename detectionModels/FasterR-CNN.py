import cv2
import torch
from torchvision import models, transforms
'''
Faster R-CNN doesn't directly process videos. It processes each frame individually, detecting objects within those frames.
'''
# Load the Faster R-CNN model (pretrained model)
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Open the video file
video_path = 'samplevideo.mov'
cap = cv2.VideoCapture(video_path)

# Get the frame size (width and height) for output video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

# Image preprocessing: convert image to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # If video ends, break the loop
    
    # Convert from OpenCV BGR format to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(rgb_frame).unsqueeze(0)  # Add batch dimension

    # Object detection with Faster R-CNN
    with torch.no_grad():
        predictions = model(img_tensor)  # Get predictions for the current frame
    
    # Process and visualize the results
    for element in range(len(predictions[0]['boxes'])):
        score = predictions[0]['scores'][element].item()  # Get confidence score for the prediction
        if score > 0.8:  # Use a threshold for confidence
            box = predictions[0]['boxes'][element].cpu().numpy().astype(int)  # Get bounding box coordinates
            label = predictions[0]['labels'][element].item()  # Get the label of the detected object
            # Draw bounding box around detected object
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    
    # Write the frame with bounding boxes to the output video
    out.write(frame)
    
    # Show the current frame with bounding boxes
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed

# Release the video capture and writer, and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
