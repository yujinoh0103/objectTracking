import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Load the pre-trained SSD model from COCO dataset
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Load an example image
url = "https://upload.wikimedia.org/wikipedia/commons/a/a1/Black_hole_SIM.jpg"  # Example image URL
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Preprocess the image (convert to Tensor and normalize)
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img)

# Run the image through the model
with torch.no_grad():
    prediction = model([img_tensor])

# Extract the predictions (boxes, labels, and scores)
boxes = prediction[0]['boxes']
labels = prediction[0]['labels']
scores = prediction[0]['scores']

# Visualize the bounding boxes
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(img)

# Draw the bounding boxes (only for scores greater than 0.5)
for box, score in zip(boxes, scores):
    if score > 0.5:
        x_min, y_min, x_max, y_max = box.tolist()
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                  fill=False, color='red', linewidth=2))
        ax.text(x_min, y_min, f'{score:.2f}', fontsize=10, color='white', backgroundcolor='red')

plt.show()
