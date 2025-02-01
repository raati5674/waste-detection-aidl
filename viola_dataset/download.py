import os
import csv
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

# Define paths
dataset_name = "viola77data/recycling-dataset"
base_dir = "viola_dataset"
images_dir = os.path.join(base_dir, "images")
annotations_file = os.path.join(base_dir, "annotations.csv")

# Load dataset
dataset = load_dataset(dataset_name, split="train")  # Adjust split if needed

# Create directories
os.makedirs(images_dir, exist_ok=True)

# Open CSV file to store annotations
with open(annotations_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "label"]) 

    # Iterate through dataset
    for idx, data in enumerate(dataset):
        if "image" not in data or "label" not in data:
            print(f"Skipping entry {idx} due to missing keys.")
            continue
        
        image_data = data["image"]
        label = data["label"]

        # If images are URLs, download them
        if isinstance(image_data, str):
            response = requests.get(image_data)
            image = Image.open(BytesIO(response.content))
        else:
            image = image_data  # Already a PIL Image

        # Save image
        image_path = os.path.join(images_dir, f"image_{idx}.jpg")
        image.save(image_path)

        # Write annotation
        writer.writerow([image_path, label])

print(f"Images saved in: {images_dir}")
print(f"Annotations saved in: {annotations_file}")
