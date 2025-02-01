import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
base_dir = "viola_dataset"
annotations_file = os.path.join(base_dir, "annotations_updated.csv")

# Load annotations
df = pd.read_csv(annotations_file)

# Ensure the dataset has the required columns
if "image_path" not in df.columns or "class_name" not in df.columns:
    raise ValueError("CSV file must contain 'image_path' and 'class_name' columns.")

# Split dataset (80% train, 10% val, 10% test)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["class_name"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["class_name"], random_state=42)

# Save the splits
train_df.to_csv(os.path.join(base_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(base_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(base_dir, "test.csv"), index=False)

# Print split summary
print(f"Dataset split completed!")
print(f"Train set: {len(train_df)} images")
print(f"Validation set: {len(val_df)} images")
print(f"Test set: {len(test_df)} images")

print(f"Train split saved to: {os.path.join(base_dir, 'train.csv')}")
print(f"Validation split saved to: {os.path.join(base_dir, 'val.csv')}")
print(f"Test split saved to: {os.path.join(base_dir, 'test.csv')}")