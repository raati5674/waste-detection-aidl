import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

# Set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned ResNet-50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_classes = 11  # Number of classes in your dataset (update accordingly)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Modify last layer to match the number of classes
model.load_state_dict(torch.load('./model/resnet-50_v1/fine_tuned_resnet50.pth'))  # Load the saved weights
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define class mapping (the same mapping you used for training)
class_mapping = {
    0: "aluminium",
    1: "batteries",
    2: "cardboard",
    3: "disposable_plates",
    4: "glass",
    5: "hard_plastic",
    6: "paper",
    7: "paper_towel",
    8: "polystyrene",
    9: "soft_plastics",
    10: "takeaway_cups",
}

# Load the annotations.csv to map real labels to class names
annotations_df = pd.read_csv('./viola_dataset/annotations_updated.csv')

# Randomly select an image
random_index = random.randint(0, len(annotations_df) - 1)
random_image_path = annotations_df.iloc[random_index]["image_path"]
real_label_index = annotations_df.iloc[random_index]["label"]
real_label = class_mapping[real_label_index]

# Load and preprocess the imageS
image = Image.open(os.path.join(random_image_path))

# Define the preprocessing transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the input size expected by ResNet-50
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
])

# Apply transformations to the image
image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Make the prediction
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted_label_index = torch.max(outputs, 1)  # Get the index of the max logit (predicted label)
    predicted_label = class_mapping[predicted_label_index.item()]

# Display the image and prediction results
plt.imshow(image)
plt.title(f"Image ID: {random_image_path}\nReal Label: {real_label}\nPredicted Label: {predicted_label}")
plt.axis('off')  # Hide axes
plt.show()

# Print the results in the console
print(f"Image ID: {random_image_path}")
print(f"Real Label: {real_label}")
print(f"Predicted Label: {predicted_label}")
