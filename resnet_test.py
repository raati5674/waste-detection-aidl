import os
import random
import json
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor

# Define the directory where the model is saved
save_directory = "./model/resnet_50/v_1"

# Check if the directory exists
if not os.path.exists(save_directory):
    raise FileNotFoundError(f"The directory '{save_directory}' does not exist. Ensure the model is saved properly.")

# Load the model and processor from the local directory
model = AutoModelForImageClassification.from_pretrained(save_directory, ignore_mismatched_sizes=True )
processor = AutoProcessor.from_pretrained(save_directory)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the path to the dataset and annotations
images_root_dir = "data"
annotations_path = "data/train_annotations.json"

# Load annotations
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# Choose a random image from the dataset
random_image_info = random.choice(annotations['images'])
random_image_path = os.path.join(images_root_dir, random_image_info['file_name'])

# Load the random image
image = Image.open(random_image_path).convert('RGB')

# Preprocess the image
encoding = processor(images=image, return_tensors="pt")
pixel_values = encoding["pixel_values"].to(device)

# Make a prediction
model.eval()
with torch.no_grad():
    logits = model(pixel_values=pixel_values).logits
    prediction = torch.argmax(logits, dim=1).item()

# Retrieve id2label mapping from the model configuration
id2label = model.config.id2label

# Print the prediction
predicted_label = id2label[prediction]
print(f"Random Image Path: {random_image_path}")
print(f"Predicted Label: {predicted_label}")

# Optionally, display the image (requires a display environment)
try:
    image.show()
except:
    print("Image display is not supported in this environment.")
