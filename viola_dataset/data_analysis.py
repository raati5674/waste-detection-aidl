import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Paths
base_dir = "viola_dataset"
images_dir = os.path.join(base_dir, "images")
annotations_file = os.path.join(base_dir, "annotations.csv")

# Class mapping
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

# Load annotations
df = pd.read_csv(annotations_file)

# Check if required columns exist
if "image_path" not in df.columns or "label" not in df.columns:
    raise ValueError("CSV file must contain 'image_path' and 'label' columns.")

# Ensure labels are integers before mapping
df["label"] = df["label"].astype(int)

# Create a new column with mapped class names
df["class_name"] = df["label"].map(class_mapping)

# Save the updated CSV file
updated_annotations_file = os.path.join(base_dir, "annotations_updated.csv")
df.to_csv(updated_annotations_file, index=False)

# Count occurrences of each class
class_counts = Counter(df["class_name"])

# Convert to DataFrame for sorting and analysis
df_counts = pd.DataFrame(class_counts.items(), columns=["Class", "Count"])
df_counts = df_counts.sort_values(by="Count", ascending=False)

# Print dataset statistics
total_images = len(df)
num_classes = len(class_counts)

print("Dataset Summary:")
print(f"Total Images: {total_images}")
print(f"Number of Classes: {num_classes}")
print(df_counts)

# Plot class distribution
plt.figure(figsize=(10, 6))
plt.bar(df_counts["Class"], df_counts["Count"], color="skyblue")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Class Distribution in Dataset")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save plot
plot_path = os.path.join(base_dir, "class_distribution.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.show()

print(f"Updated annotations saved to: {updated_annotations_file}")
print(f"Class distribution plot saved to: {plot_path}")
