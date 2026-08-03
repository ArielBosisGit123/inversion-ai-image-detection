import os
import random
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
from pathlib import Path
import pandas as pd

SYNTHETIC_IMAGES_PATH = r"dataset/synthetic_images"
LABELS_CSV_PATH = r"dataset/labels.csv"

NUM_INFERENCE_STEPS = 40

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Downloading ADM-G-512 subfolder from HuggingFace...")
repo_path = snapshot_download(repo_id="BiliSakura/ADM-diffusers", allow_patterns="ADM-G-512/*")
model_dir = Path(repo_path) / "ADM-G-512"

model_name = "ADM-diffusers"

pipe = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        custom_pipeline=str(model_dir / "pipeline.py"),
        trust_remote_code=True
)
pipe.to(device)

# Different clases compatable with the model
imagenet_classes = [
        "lion", "sports car", "castle", "space shuttle", "golden retriever",
        "monastery", "fire engine", "convertible", "airliner", "fountain"
]

prompt_num = 50
prompts = []

counter = 0
while counter < prompt_num:
        class_label = random.choice(imagenet_classes)
        prompts.append(class_label)
        counter += 1

expected_columns = ["file_name", "AI", "origin_name"]
try:
        labels_df = pd.read_csv(LABELS_CSV_PATH)
except pd.errors.EmptyDataError:
        labels_df = pd.DataFrame(columns=expected_columns)

new_label_data = []

# Image generation
for idx, class_label in enumerate(prompts):
        image_name = f"{model_name}{idx}.png"

        print(f"Generating and saving image {idx+1}/{prompt_num}: {image_name}")

        output = pipe(class_labels=class_label, num_inference_steps=NUM_INFERENCE_STEPS)
        image = output.images[0]

        image.save(os.path.join(SYNTHETIC_IMAGES_PATH, image_name))

        new_label_data.append({"file_name": image_name,
                               "AI": 1,
                               "origin_name": model_name})

# Saving labels.csv data
new_df = pd.DataFrame(new_label_data)
final_df = pd.concat([labels_df, new_df], ignore_index=True)
final_df.to_csv(LABELS_CSV_PATH, index=False)