import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='./runs/experiment_1')

# Paths
base_dir = "./viola_dataset/"
train_file = os.path.join(base_dir, "train.csv")
val_file = os.path.join(base_dir, "val.csv")
test_file = os.path.join(base_dir, "test.csv")


# Path to the updated annotations file
annotations_file = os.path.join(base_dir, "annotations_updated.csv")

# Load the annotations file for mapped classes
annotations_df = pd.read_csv(annotations_file)

# Load the datasets
train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

# Extract unique class names from the annotations file (assuming 'class_name' column contains the names)
class_names = annotations_df["class_name"].unique()

# Create a mapping from class names to numerical labels
class_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}

# Now, map the class names in train, val, and test datasets to numerical labels
train_df["label"] = train_df["class_name"].map(class_mapping)
val_df["label"] = val_df["class_name"].map(class_mapping)
test_df["label"] = test_df["class_name"].map(class_mapping)

print("Class Mapping:", class_mapping)

# Define the image transformations for training and validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),       # Resize images to 224x224 (ResNet-50 input size)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained ResNet mean & std
])

# Custom Dataset Class to load images and labels
class WasteDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_path"]
        label = self.dataframe.iloc[idx]["label"]
        image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format

        if self.transform:
            image = self.transform(image)
        
        return image, label

# Create DataLoaders for train, validation, and test
train_dataset = WasteDataset(train_df, transform)
val_dataset = WasteDataset(val_df, transform)
test_dataset = WasteDataset(test_df, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Modify the final layer to match the number of classes
num_classes = len(class_mapping)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Check if a GPU is available and move the model accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

        # Log the training loss and accuracy every 100 steps
        if step % 100 == 0:
            writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + step)
            writer.add_scalar("Training Accuracy", correct_preds / total_preds, epoch * len(train_loader) + step)

    train_accuracy = correct_preds / total_preds
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {train_accuracy * 100:.2f}%")

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    val_accuracy = correct_preds / total_preds
    print(f"Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Log the validation loss and accuracy
    writer.add_scalar("Validation Loss", val_loss / len(val_loader), epoch)
    writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_resnet50.pth') 

# Close the writer
writer.close()

print("Training complete, model saved!")
