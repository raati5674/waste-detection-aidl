import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor

# Use the processor's feature extractor for preprocessing
processor = AutoProcessor.from_pretrained("microsoft/resnet-50")

class TACOClassificationDataset(Dataset):
    def __init__(self, annotations_file, root_dir, processor):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.root_dir = root_dir
        self.processor = processor
        self.image_annotations = self.annotations['images']
        self.annotations_mapping = {
            image['id']: image['file_name'] for image in self.image_annotations
        }
        self.labels_mapping = {
            category['id']: category['name'] for category in self.annotations['categories']
        }
        self.annotations_by_image = {}
        for annotation in self.annotations['annotations']:
            if annotation['image_id'] not in self.annotations_by_image:
                self.annotations_by_image[annotation['image_id']] = []
            self.annotations_by_image[annotation['image_id']].append(annotation['category_id'])

    def __len__(self):
        return len(self.image_annotations)

    def __getitem__(self, idx):
        image_info = self.image_annotations[idx]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        # Preprocess the image using the processor
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)  # Remove batch dimension

        # Get the label (for simplicity, pick the first category ID associated with the image)
        image_id = image_info['id']
        label = self.annotations_by_image[image_id][0]

        # Convert label to index
        label_index = list(self.labels_mapping.keys()).index(label)

        return pixel_values, label_index

# Paths to your dataset
train_annotations_path = "data/train_annotations.json"
val_annotations_path = "data/validation_annotations.json"
test_annotations_path = "data/test_annotations.json"
images_root_dir = "data"


# Update the DataLoader and Dataset Initialization
train_dataset = TACOClassificationDataset(train_annotations_path, images_root_dir, processor)
val_dataset = TACOClassificationDataset(val_annotations_path, images_root_dir, processor)
test_dataset = TACOClassificationDataset(test_annotations_path, images_root_dir, processor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 2: Load Pretrained ResNet-50
num_classes = len(train_dataset.labels_mapping)
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-50", num_labels=num_classes, ignore_mismatched_sizes=True
)

# Step 3: Define Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training and Validation
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    # Validation
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(pixel_values=images).logits
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    print(f"Validation Accuracy: {total_correct / total_samples:.4f}")

# Step 4: Test the Model
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = model(pixel_values=images).logits
        predictions = torch.argmax(outputs, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

print(f"Test Accuracy: {total_correct / total_samples:.4f}")