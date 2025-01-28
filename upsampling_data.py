import os
import json
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# TensorBoard:

logs_base_dir = "resnet-50"
os.makedirs(logs_base_dir, exist_ok=True)

tb_rs_up = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_UPSAMPLING/')



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

# Dataset Class with Augmentation
class TACOClassificationDataset(Dataset):
    def __init__(self, annotations_file, root_dir, processor, reclassify_mapping, augment=False):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.root_dir = root_dir
        self.processor = processor
        self.reclassify_mapping = reclassify_mapping
        self.augment = augment

        # Load image and label mappings
        self.image_annotations = self.annotations['images']
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
            if new_category != "Miscellaneous":
                if image_id not in self.image_id_to_labels:
                    self.image_id_to_labels[image_id] = []
                self.image_id_to_labels[image_id].append(new_category)

        # Filter images
        self.filtered_image_annotations = [
            image for image in self.image_annotations if image['id'] in self.image_id_to_labels
        ]

        # Finalize labels
        self.new_labels = sorted(set(
            label for labels in self.image_id_to_labels.values() for label in labels
        ))
        self.label_to_index = {label: idx for idx, label in enumerate(self.new_labels)}
        self.class_counts = {label: 0 for label in self.new_labels}
        for labels in self.image_id_to_labels.values():
            for label in labels:
                self.class_counts[label] += 1

        # Compute class weights
        self.class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(list(range(len(self.new_labels)))),  # Convert to numpy.ndarray
            y=np.array([self.label_to_index[label] for labels in self.image_id_to_labels.values() for label in labels])  # Convert labels to numpy.ndarray
        )
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)

    def __len__(self):
        return len(self.filtered_image_annotations)

    def __getitem__(self, idx):
        image_info = self.filtered_image_annotations[idx]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        # Apply augmentations for minority classes
        if self.augment:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                transforms.ToTensor(),
            ])
            image = transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Preprocess the image
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)  # Remove batch dimension

        # Get label
        image_id = image_info['id']
        labels = self.image_id_to_labels[image_id]
        label = labels[0]
        label_index = self.label_to_index[label]

        return pixel_values, label_index

# Paths to your dataset
train_annotations_path = "data/train_annotations.json"
val_annotations_path = "data/validation_annotations.json"
test_annotations_path = "data/test_annotations.json"
images_root_dir = "data"

# Datasets and WeightedRandomSampler
train_dataset = TACOClassificationDataset(train_annotations_path, images_root_dir, processor, RECLASSIFY_MAPPING, augment=True)
val_dataset = TACOClassificationDataset(val_annotations_path, images_root_dir, processor, RECLASSIFY_MAPPING)
test_dataset = TACOClassificationDataset(test_annotations_path, images_root_dir, processor, RECLASSIFY_MAPPING)

train_weights = [train_dataset.class_weights[label] for _, label in train_dataset]
train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights))

train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model with Dropout
num_classes = len(train_dataset.new_labels)
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-50", num_labels=num_classes, ignore_mismatched_sizes=True
)
model.classifier.add_module("dropout", torch.nn.Dropout(p=0.25))

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# Training and Validation
num_epochs = 7
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
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

    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

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



#SAVING THE MODEL W/ THE LABELS
#id2label and label2id mappings
id2label = {idx: label for label, idx in train_dataset.label_to_index.items()}
label2id = {label: idx for idx, label in id2label.items()}

# Add the mappings to the model configuration
model.config.id2label = id2label
model.config.label2id = label2id

model_directory = "model/resnet_50/v_2"

# Save the model and processor
model.save_pretrained(model_directory)
processor.save_pretrained(model_directory)