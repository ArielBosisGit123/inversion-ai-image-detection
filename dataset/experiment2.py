import os
from PIL import Image
import pandas as pd
from utils import calculate_mse
import time

REAL_IMAGES_PATH = r"dataset/real_images"
SYNTHETIC_IMAGES_PATH = r"dataset/synthetic_images"
LABELS_CSV_PATH = r"dataset/labels.csv"
RESULTS_CSV_PATH = r"experiment2/results.csv"

NUM_IMAGES = 100 # 1/2 synthetic, 1/2 real
NUM_INFERENCE_STEPS_LIST = [10, 25, 50]

labels_df = pd.read_csv(LABELS_CSV_PATH)
results_data = []

for i in range(NUM_IMAGES):
        if i < NUM_IMAGES / 2:
                img_name = f"stable-diffusion-v1-5{i}.png"
                img_path = os.path.join(SYNTHETIC_IMAGES_PATH, img_name)
        else:
                img_name = f"lsun-bedrooms{i - NUM_IMAGES // 2}.png"
                img_path = os.path.join(REAL_IMAGES_PATH, img_name)

        for num_steps in NUM_INFERENCE_STEPS_LIST:
                start_time = time.perf_counter()

                img = Image.open(img_path)
                mse = calculate_mse(img, num_steps)

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                print(f"[{i+1}/{NUM_IMAGES}] '{img_name}', num_steps = {num_steps} - MSE: {mse:.5f}")

                is_ai = labels_df.loc[labels_df["file_name"].str.contains(img_name), "AI"].to_list()[0] # 0 (real) or 1 (AI)

                results_data.append({
                        "file_name": img_name,
                        "AI": is_ai,
                        "MSE": mse,
                        "steps": num_steps,
                        "time": elapsed_time
                })

results_df = pd.DataFrame(results_data)
results_df.to_csv(RESULTS_CSV_PATH, index=False)