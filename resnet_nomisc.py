import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor

# 6 new classes:
RECLASSIFY_MAPPING = {
    # Plastic
    "Bottle cap": "Plastic",
    "Blister pack": "Plastic",
    "Bottle": "Plastic",
    "Lid": "Plastic",
    "Other plastic": "Plastic",
    "Plastic bag & wrapper": "Plastic",
    "Plastic container": "Plastic",
    "Plastic gloves": "Plastic",
    "Plastic utensils": "Plastic",
    "Squeezable tube": "Plastic",
    "Straw": "Plastic",
    "Styrofoam piece": "Plastic",
    # Metal
    "Aluminium foil": "Metal",
    "Can": "Metal",
    "Pop tab": "Metal",
    "Scrap metal": "Metal",
    "Battery": "Metal",
    # Glass
    "Broken glass": "Glass",
    "Glass jar": "Glass",
    # Paper/Cardboard
    "Carton": "Paper/Cardboard",
    "Paper": "Paper/Cardboard",
    "Paper bag": "Paper/Cardboard",
    # Organic
    "Food waste": "Organic",
    "Rope & strings": "Organic",
    # Miscellaneous
    "Shoe": "Miscellaneous",
    "Cigarette": "Miscellaneous",
    "Unlabeled litter": "Miscellaneous",
}

# Use the processor's feature extractor for preprocessing
processor = AutoProcessor.from_pretrained("microsoft/resnet-50")

class TACOClassificationDataset(Dataset):
    def __init__(self, annotations_file, root_dir, processor, reclassify_mapping):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.root_dir = root_dir
        self.processor = processor
        self.image_annotations = self.annotations['images']
        self.reclassify_mapping = reclassify_mapping

        # Build mapping for image IDs and new labels
        self.image_id_to_file_name = {
            image['id']: image['file_name'] for image in self.image_annotations
        }
        self.category_id_to_name = {
            category['id']: category['name'] for category in self.annotations['categories']
        }
        self.image_id_to_labels = {}
        for annotation in self.annotations['annotations']:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            category_name = self.category_id_to_name[category_id]
            new_category = self.reclassify_mapping.get(category_name, "Miscellaneous")
            if new_category != "Miscellaneous":  # Exclude Miscellaneous
                if image_id not in self.image_id_to_labels:
                    self.image_id_to_labels[image_id] = []
                self.image_id_to_labels[image_id].append(new_category)

        # Filter out images with only "Miscellaneous" labels
        self.filtered_image_annotations = [
            image for image in self.image_annotations if image['id'] in self.image_id_to_labels
        ]

        # Finalize unique label mapping
        self.new_labels = sorted(set(
            label for labels in self.image_id_to_labels.values() for label in labels
        ))
        self.label_to_index = {label: idx for idx, label in enumerate(self.new_labels)}

    def __len__(self):
        return len(self.filtered_image_annotations)

    def __getitem__(self, idx):
        image_info = self.filtered_image_annotations[idx]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        # Preprocess the image using the processor
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)  # Remove batch dimension

        # Get the label (select the first reclassified label)
        image_id = image_info['id']
        labels = self.image_id_to_labels[image_id]
        label = labels[0]  # Take the first associated label
        label_index = self.label_to_index[label]

        return pixel_values, label_index

# Paths to your dataset
train_annotations_path = "data/train_annotations.json"
val_annotations_path = "data/validation_annotations.json"
test_annotations_path = "data/test_annotations.json"
images_root_dir = "data"

# Update the DataLoader and Dataset Initialization
train_dataset = TACOClassificationDataset(train_annotations_path, images_root_dir, processor, RECLASSIFY_MAPPING)
val_dataset = TACOClassificationDataset(val_annotations_path, images_root_dir, processor, RECLASSIFY_MAPPING)
test_dataset = TACOClassificationDataset(test_annotations_path, images_root_dir, processor, RECLASSIFY_MAPPING)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 2: Load Pretrained ResNet-50
num_classes = len(RECLASSIFY_MAPPING.values()) - 1  # Exclude Miscellaneous
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-50", num_labels=num_classes, ignore_mismatched_sizes=True
)

# Step 3: Define Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# Training and Validation
num_epochs = 15
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print batch-level information
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
            print(f"Sample Labels: {labels[:5].cpu().numpy()}")
            print(f"Sample Predictions: {torch.argmax(outputs[:5], dim=1).cpu().numpy()}")

    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(pixel_values=images).logits
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Print validation predictions for the first batch
            if epoch == 0 and batch_idx == 0:
                print(f"**Validation Labels:** {labels[:5].cpu().numpy()}")
                print(f"**Validation Predictions:** {predictions[:5].cpu().numpy()}")

    print(f"**Validation Accuracy: ** {total_correct / total_samples:.4f}")

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

#SAVING THE MODEL W/ THE LABELS
#id2label and label2id mappings
id2label = {idx: label for label, idx in train_dataset.label_to_index.items()}
label2id = {label: idx for idx, label in id2label.items()}

# Add the mappings to the model configuration
model.config.id2label = id2label
model.config.label2id = label2id

model_directory = "model/resnet_50/v_1"

# Save the model and processor
model.save_pretrained(model_directory)
processor.save_pretrained(model_directory)