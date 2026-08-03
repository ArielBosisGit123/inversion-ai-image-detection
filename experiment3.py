import os
from PIL import Image
import pandas as pd
from utils import calculate_mse, manipulate_img, manipulations
import time

REAL_IMAGES_PATH = r"dataset/real_images"
SYNTHETIC_IMAGES_PATH = r"dataset/synthetic_images"
LABELS_CSV_PATH = r"dataset/labels.csv"
RESULTS_CSV_PATH = r"experiment3/results.csv"

NUM_IMAGES = 100 # 1/2 synthetic, 1/2 real
NUM_INFERENCE_STEPS = 25

labels_df = pd.read_csv(LABELS_CSV_PATH)
results_data = []

for i in range(NUM_IMAGES):
        if i < NUM_IMAGES / 2:
                img_name = f"stable-diffusion-v1-5{i}.png"
                img_path = os.path.join(SYNTHETIC_IMAGES_PATH, img_name)
        else:
                img_name = f"lsun-bedrooms{i - NUM_IMAGES // 2}.png"
                img_path = os.path.join(REAL_IMAGES_PATH, img_name)

        img = Image.open(img_path)

        for manipulation in manipulations:
                manipulated_img = manipulate_img(img, manipulation["type"], manipulation["val"])
                manipulation_label = manipulation["label"]

                mse = calculate_mse(manipulated_img, NUM_INFERENCE_STEPS)

                print(f"[{i+1}/{NUM_IMAGES}] '{img_name}', {manipulation_label} - MSE: {mse:.5f}")

                is_ai = labels_df.loc[labels_df["file_name"].str.contains(img_name), "AI"].to_list()[0] # 0 (real) or 1 (AI)

                results_data.append({
                        "file_name": img_name,
                        "AI": is_ai,
                        "MSE": mse,
                        "manipulation": manipulation_label
                })

results_df = pd.DataFrame(results_data)
results_df.to_csv(RESULTS_CSV_PATH, index=False)