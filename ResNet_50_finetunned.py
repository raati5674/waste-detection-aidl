import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor
from tqdm import tqdm

# Step 1: Setup and Load Data
class TACOClassificationDataset(Dataset):
    def __init__(self, annotations_file, root_dir, processor, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.root_dir = root_dir
        self.processor = processor
        self.transform = transform
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

        if self.transform:
            image = self.transform(image)

        # Get the label (for simplicity, pick the first category ID associated with the image)
        image_id = image_info['id']
        label = self.annotations_by_image[image_id][0]

        # Convert label to index
        label_index = list(self.labels_mapping.keys()).index(label)

        return image, label_index

# Paths to your dataset
train_annotations_path = "data/train_annotations.json"
val_annotations_path = "data/validation_annotations.json"
test_annotations_path = "data/test_annotations.json"
images_root_dir = "data"

# Define transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Augmentation
    transforms.RandomRotation(10),  # Augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Processor (AutoProcessor handles feature extraction)
processor = AutoProcessor.from_pretrained("microsoft/resnet-50")

# Load datasets
train_dataset = TACOClassificationDataset(train_annotations_path, images_root_dir, processor, transform)
val_dataset = TACOClassificationDataset(val_annotations_path, images_root_dir, processor, transform)
test_dataset = TACOClassificationDataset(test_annotations_path, images_root_dir, processor, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 2: Load Pretrained ResNet-50
num_classes = len(train_dataset.labels_mapping)
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-50", num_labels=num_classes, ignore_mismatched_sizes=True
)

# Step 3: Define Training Loop
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Freeze all layers except the final classifier layer
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last classifier layer (the fully connected layer)
for param in model.classifier.parameters():
    param.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()

# Use a smaller learning rate and add weight decay to the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)

# Training and Validation
num_epochs = 20  # Increase epochs for better fine-tuning
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct_train = 0
    total_samples_train = 0

    # Training Loop with tqdm progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training") as train_progress:
        for batch in train_progress:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy for training
            predictions = torch.argmax(outputs, dim=1)
            total_correct_train += (predictions == labels).sum().item()
            total_samples_train += labels.size(0)

            train_progress.set_postfix(loss=total_loss / len(train_loader),
                                       accuracy=total_correct_train / total_samples_train)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
    print(f"Training Accuracy: {total_correct_train / total_samples_train:.4f}")

    # Validation
    model.eval()
    total_correct_val = 0
    total_samples_val = 0

    with tqdm(val_loader, desc="Validation") as val_progress:
        with torch.no_grad():
            for batch in val_progress:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                outputs = model(pixel_values=images).logits
                predictions = torch.argmax(outputs, dim=1)

                total_correct_val += (predictions == labels).sum().item()
                total_samples_val += labels.size(0)

                val_progress.set_postfix(accuracy=total_correct_val / total_samples_val)

    print(f"Validation Accuracy: {total_correct_val / total_samples_val:.4f}")

# Step 4: Test the Model
model.eval()
total_correct_test = 0
total_samples_test = 0
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = model(pixel_values=images).logits
        predictions = torch.argmax(outputs, dim=1)
        total_correct_test += (predictions == labels).sum().item()
        total_samples_test += labels.size(0)

print(f"Test Accuracy: {total_correct_test / total_samples_test:.4f}")