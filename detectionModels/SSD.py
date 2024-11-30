from PIL import Image
import torch
import torchvision
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained SSD model
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Open the video file (or use webcam by passing 0 for the camera)
cap = cv2.VideoCapture("samplevideo.mov")  # Replace with 0 for webcam

# Get the video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR frame to RGB for PIL and then to Tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    
    # Preprocess the image (convert to Tensor and normalize)
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(pil_img)

    # Run the image through the model
    with torch.no_grad():
        prediction = model([img_tensor])

    # Extract the predictions (boxes, labels, and scores)
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    # Visualize the bounding boxes (only for scores greater than 0.5)
    for box, score in zip(boxes, scores):
        if score > 0.5:
            x_min, y_min, x_max, y_max = box.tolist()
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}', (int(x_min), int(y_min) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame with bounding boxes
    cv2.imshow("Object Detection", frame)

    # Write the frame into the video file
    out.write(frame)

    # Press 'q' to quit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
