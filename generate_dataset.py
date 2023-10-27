import yaml
from sklearn.model_selection import train_test_split
import os

data_root = '/media/niccolo/DATA/DND-Diko-WWWR/Challenge/DND-Diko-WWWR'  # path to dataset

metadata_path = os.path.join(data_root, "WW2020", 'labels_trainval.yml')

# Load the data
with open(metadata_path, 'r') as file:
    data = yaml.safe_load(file)

# Convert dictionary to lists for easier processing
images, labels = list(data.keys()), list(data.values())

# Split: 80% train, 20% validation
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

# Convert lists back to dictionaries
train_data = dict(zip(train_images, train_labels))
val_data = dict(zip(val_images, val_labels))

# Write the splits into separate .yml files
with open('labels_train.yml', 'w') as file:
    yaml.dump(train_data, file)

with open('labels_val.yml', 'w') as file:
    yaml.dump(val_data, file)

# Generate .txt files containing only the image paths for each split
with open('train_paths.txt', 'w') as file:
    file.write('\n'.join(train_images))

with open('val_paths.txt', 'w') as file:
    file.write('\n'.join(val_images))

print("Data has been stratified and written to separate .yml and .txt files.")
