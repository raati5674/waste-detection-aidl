from transformers import ViTForImageClassification, ViTImageProcessor
from torch.utils.data import DataLoader
import torch

# Load the pretrained ViT model and processor
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(CUSTOM_CATEGORIES),  # Number of custom categories
    id2label={i: category for i, category in enumerate(CUSTOM_CATEGORIES.keys())},
    label2id={category: i for i, category in enumerate(CUSTOM_CATEGORIES.keys())}
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define transforms compatible with ViT
def preprocess_image(image):
    return processor(images=image, return_tensors="pt").pixel_values

# Create a dataloader for training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop (simplified)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        images, labels = batch['image'], batch['label']
        images = preprocess_image(images).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


