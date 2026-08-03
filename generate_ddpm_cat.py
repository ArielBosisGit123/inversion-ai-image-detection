import os
import torch
from diffusers import DiffusionPipeline
import pandas as pd
from PIL import Image

SYNTHETIC_IMAGES_PATH = r"dataset/synthetic_images"
LABELS_CSV_PATH = r"dataset/labels.csv"

NUM_INFERENCE_STEPS = 40
NUM_IMAGES_TO_GENERATE = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "google/ddpm-cat-256"
model_name = "ddpm-cat-256"

print(f"Loading {model_id} from HuggingFace...")
# DiffusionPipeline will automatically infer this is a DDPMPipeline (unconditional)
pipe = DiffusionPipeline.from_pretrained(model_id)
pipe.to(device)

expected_columns = ["file_name", "AI", "origin_name"]
try:
        labels_df = pd.read_csv(LABELS_CSV_PATH)
except pd.errors.EmptyDataError:
        labels_df = pd.DataFrame(columns=expected_columns)
except FileNotFoundError:
        labels_df = pd.DataFrame(columns=expected_columns)

new_label_data = []

# Image generation
for idx in range(NUM_IMAGES_TO_GENERATE):
        image_name = f"{model_name}{idx}.png"

        print(f"Generating and saving image {idx+1}/{NUM_IMAGES_TO_GENERATE}: {image_name}")

        output = pipe(num_inference_steps=NUM_INFERENCE_STEPS)
        image = output.images[0]

        # Resize to 512x512
        image = image.resize((512, 512), Image.Resampling.BICUBIC)

        os.makedirs(SYNTHETIC_IMAGES_PATH, exist_ok=True)
        image.save(os.path.join(SYNTHETIC_IMAGES_PATH, image_name))

        new_label_data.append({"file_name": image_name,
                               "AI": 1,
                               "origin_name": model_name})

# Saving labels.csv data
new_df = pd.DataFrame(new_label_data)
final_df = pd.concat([labels_df, new_df], ignore_index=True)

os.makedirs(os.path.dirname(LABELS_CSV_PATH), exist_ok=True)
final_df.to_csv(LABELS_CSV_PATH, index=False)
print("Generation complete and labels updated!")