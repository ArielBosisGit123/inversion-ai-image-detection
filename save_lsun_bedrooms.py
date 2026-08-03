import os
from datasets import load_dataset
import pandas as pd

REAL_IMAGES_PATH = r"dataset/real_images"
LABELS_CSV_PATH = r"dataset/labels.csv"

NUM_IMAGES = 50

dataset_id = "pcuenq/lsun-bedrooms"
dataset_name = "lsun-bedrooms"
dataset = load_dataset(dataset_id, split="train")

# If .csv file is empty, create columns
expected_columns = ["file_name", "AI", "origin_name"]
try:
    labels_df = pd.read_csv(LABELS_CSV_PATH)
except pd.errors.EmptyDataError:
    labels_df = pd.DataFrame(columns=expected_columns)

new_label_data = []

# Getting images from the HuggingFace dataset
for idx, item in enumerate(dataset):
        if idx >= NUM_IMAGES:
                break

        image_name = f"{dataset_name}{idx}.png"

        print(f"Saving image {idx+1}/{NUM_IMAGES}: {image_name}")

        image = item["image"]
        image = image.convert("RGB")
        image = image.resize((512, 512))

        image.save(os.path.join(REAL_IMAGES_PATH, image_name))

        new_label_data.append({"file_name": image_name,
                               "AI": 0,
                               "origin_name": dataset_name})

# Saving data for labels.csv
new_df = pd.DataFrame(new_label_data)
final_df = pd.concat([labels_df, new_df], ignore_index=True)
final_df.to_csv(LABELS_CSV_PATH, index=False)