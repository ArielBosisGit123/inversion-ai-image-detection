import os
import random
import torch
from diffusers import StableDiffusionPipeline
import pandas as pd

SYNTHETIC_IMAGES_PATH = r"dataset/synthetic_images"
LABELS_CSV_PATH = r"dataset/labels.csv"

NUM_INFERENCE_STEPS = 40

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.device(device)

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
model_name = "stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to(device)

# Creating prompts for image generation
views = ["A wide angle shot of", "A close up shot of"]
subjects = [
          "a majestic lion", "a futuristic cyberpunk city", "a cute domestic cat",
          "a cottage in a green forest", "an astronaut floating in space",
         "a vintage red sports car", "a hot cup of coffee on a rainy day",
         "a high-tech humanoid robot", "a snow-covered mountain peak", "a medieval castle"
]
styles = [
         "photorealistic, highly detailed, 8k resolution, DSLR",
          "beautiful oil painting, artistic brush strokes",
          "dreamy watercolor illustration, soft colors",
          "anime style, vibrant colors, clean lines",
          "cinematic lighting, dramatic atmosphere, dark background"
]

prompt_num = 50
prompts = []

counter = 0
while counter < prompt_num:
        view = random.choice(views)
        subject = random.choice(subjects)
        style = random.choice(styles)

        prompt = f"{view} {subject}, {style}."

        if not prompt in prompts:
                prompts.append(prompt)
                counter += 1

# If .csv file is empty, create columns
expected_columns = ["file_name", "AI", "origin_name"]
try:
        labels_df = pd.read_csv(LABELS_CSV_PATH)
except pd.errors.EmptyDataError:
        labels_df = pd.DataFrame(columns=expected_columns)

new_label_data = []

# Image generation
for idx, prompt in enumerate(prompts):
        image_name = f"{model_name}{idx}.png"

        print(f"Generating and saving image {idx+1}/{prompt_num}: {image_name}")

        with torch.inference_mode():
                output = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, height=512, width=512)
                image = output.images[0]

        image.save(os.path.join(SYNTHETIC_IMAGES_PATH, image_name))

        new_label_data.append({"file_name": image_name,
                               "AI": 1,
                               "origin_name": model_name})

# Saving data for labels.csv
new_df = pd.DataFrame(new_label_data)
final_df = pd.concat([labels_df, new_df], ignore_index=True)
final_df.to_csv(LABELS_CSV_PATH, index=False)