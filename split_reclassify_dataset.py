import json
from sklearn.model_selection import train_test_split
import argparse
import os

# Mapping of original TACO categories to new categories
CATEGORY_MAPPING = {
    "Plastic": [
        "Bottle cap", "Blister pack", "Bottle", "Lid", "Other plastic",
        "Plastic bag & wrapper", "Plastic container", "Plastic gloves",
        "Plastic utensils", "Squeezable tube", "Straw", "Styrofoam piece"
        
    ],
    "Metal": [
        "Aluminium foil", "Can", "Pop tab", "Scrap metal", "Battery"
    ],
    "Glass": [
        "Broken glass", "Glass jar"
    ],
    "Paper/Cardboard": [
        "Carton", "Paper", "Paper bag"
    ],
    "Organic": [
        "Food waste", "Rope & strings"
    ],
    "Miscellaneous": [
        "Shoe", "Cigarette", "Unlabeled litter"
    ]
}

# Reverse mapping for quick lookups (original category -> new category)
CATEGORY_LOOKUP = {}
for new_category, original_categories in CATEGORY_MAPPING.items():
    for original_category in original_categories:
        CATEGORY_LOOKUP[original_category] = new_category

# Parse arguments
parser = argparse.ArgumentParser(description='Split dataset into training, validation and testing sets')
parser.add_argument('--dataset_dir', required=True, help='Path to dataset annotations', type=str)
parser.add_argument('--test_percentage', required=False, help='Percentage of images used for the testing set', type=float, default=0.10)
parser.add_argument('--val_percentage', required=False, help='Percentage of images used for the validation set', type=float, default=0.10)
parser.add_argument('--seed', required=False, help='Random seed for the split', type=int, default=123)
parser.add_argument('--verbose', required=False, help='Print information about the split', type=bool, default=False)

args = parser.parse_args()

# Get annotations path
ann_input_path = os.path.join(args.dataset_dir, 'annotations.json')

# Check if the annotations file exists
assert os.path.exists(ann_input_path), 'Annotations file not found'
if args.verbose: print('Annotations file found...')

# Load COCO annotations
with open(ann_input_path, 'r') as f:
    coco_data = json.load(f)
if args.verbose: print('Annotations file loaded...')

# Reclassify categories in COCO annotations
def reclassify_categories(coco_data):
    # Create new categories list based on the 6 new categories
    new_categories = [
        {"id": i + 1, "name": category} for i, category in enumerate(CATEGORY_MAPPING.keys())
    ]
    new_category_ids = {category["name"]: category["id"] for category in new_categories}

    # Update annotations with new categories
    for annotation in coco_data["annotations"]:
        # Find the original category name
        original_category_id = annotation["category_id"]
        original_category_name = next(
            (cat["name"] for cat in coco_data["categories"] if cat["id"] == original_category_id), None
        )

        # Find the new category for this annotation
        new_category_name = CATEGORY_LOOKUP.get(original_category_name, "Miscellaneous")
        annotation["category_id"] = new_category_ids[new_category_name]

    # Replace the old categories with the new categories
    coco_data["categories"] = new_categories
    return coco_data

if args.verbose: print('Reclassifying categories...')
coco_data = reclassify_categories(coco_data)
if args.verbose: print('Categories reclassified...')

# Get image IDs
image_ids = [image['id'] for image in coco_data['images']]

# Split COCO annotations based on image IDs in training, validation and testing sets
train_val_ids, test_ids = train_test_split(image_ids, test_size=args.test_percentage, random_state=args.seed)
train_ids, val_ids = train_test_split(train_val_ids, test_size=args.val_percentage/(1-args.test_percentage), random_state=args.seed)
if args.verbose: print('Annotations split...')

# Define a function to filter data based on image IDs
def filter_coco_data(image_ids, coco_data):
    return {
        'info': coco_data['info'],
        'images': [image for image in coco_data['images'] if image['id'] in image_ids],
        'annotations': [annotation for annotation in coco_data['annotations'] if annotation['image_id'] in image_ids],
        'licenses': coco_data['licenses'],
        'categories': coco_data['categories']
    }

# Create new annotations for training, validation and test sets
if args.verbose: print('Filtering annotations according to the split...')
train_dataset = filter_coco_data(train_ids, coco_data)
val_dataset = filter_coco_data(val_ids, coco_data)
test_dataset = filter_coco_data(test_ids, coco_data)
if args.verbose: print('Filtering completed...')

# Save the split COCO annotations in different files
train_output_path = os.path.join(args.dataset_dir,'train_annotations.json')
val_output_path = os.path.join(args.dataset_dir,'validation_annotations.json')
test_output_path = os.path.join(args.dataset_dir,'test_annotations.json')

if args.verbose: print('Creating train_annotations.json...')
with open(train_output_path, 'w') as f:
    json.dump(train_dataset, f, indent=4)

if args.verbose: print('Creating validation_annotations.json...')
with open(val_output_path, 'w') as f:
    json.dump(val_dataset, f, indent=4)

if args.verbose: print('Creating test_annotations.json...')
with open(test_output_path, 'w') as f:
    json.dump(test_dataset, f, indent=4)

if args.verbose: print(f'JSONs created in {args.dataset_dir}. Completed!')
