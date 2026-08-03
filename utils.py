import os
import torch
from PIL import Image, ImageFilter
import numpy as np
import io
from IPython.display import display
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler

manipulations = [
     {"type": "none", "val": 0, "label": "none"},
     {"type": "jpeg", "val": 85, "label": "JPEG-Quality-85"},
     {"type": "jpeg", "val": 70, "label": "JPEG-Quality-70"},
     {"type": "crop", "val": 20, "label": "Crop-20%"},
     {"type": "crop", "val": 40, "label": "Crop-40%"},
     {"type": "rotation", "val": 20, "label": "Rotation-20deg"},
     {"type": "rotation", "val": 40, "label": "Rotation-40deg"},
     {"type": "resize", "val": 140, "label": "Scale-140%"},
     {"type": "resize", "val": 60, "label": "Scale-60%"},
     {"type": "blur", "val": 0.5, "label": "Blur-Radius-0.5px"},
     {"type": "blur", "val": 1, "label": "Blur-Radius-1px"},
]

def manipulate_img(img: Image.Image, type, value):
        img = img.convert("RGB")

        if type == "none":
                return img

        elif type == "jpeg":
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=int(value))
                buffer.seek(0)
                return Image.open(buffer).resize((512, 512), Image.Resampling.BICUBIC)

        elif type == "crop":
                width, height = img.size
                crop_width = int(width * (1 - float(value) / 100.0))
                crop_height = int(height * (1 - float(value) / 100.0))

                left = (width - crop_width) // 2
                top = (height - crop_height) // 2
                right = left + crop_width
                bottom = top + crop_height

                return img.crop((left, top, right, bottom)).resize((512, 512), Image.Resampling.BICUBIC)

        elif type == "rotation":
                angle = float(value)
                rotated_img = img.rotate(angle=angle, resample=Image.Resampling.BICUBIC)

                width, height = rotated_img.size
                crop_size = int(min(width, height) / (np.abs(np.sin(np.radians(angle))) + np.abs(np.cos(np.radians(angle)))))
                left = (width - crop_size) // 2
                top = (height - crop_size) // 2

                return rotated_img.crop((left, top, left + crop_size, top + crop_size)).resize((512, 512), Image.Resampling.BICUBIC)

        elif type == "resize":
                width, height = img.size
                resized_img = img.resize((int(width * float(value) / 100.0), int(height * float(value) / 100.0)), Image.Resampling.BICUBIC)
                return resized_img.resize((512, 512), Image.Resampling.BICUBIC)

        elif type == "blur":
                return img.filter(ImageFilter.GaussianBlur(radius=float(value))).resize((512, 512), Image.Resampling.BICUBIC)

def display_img_manipulations(img_path):
        test_img = Image.open(img_path)

        grouped_results = {}

        print("Processing images...")
        for manipulation in manipulations:
                manipulated_img = manipulate_img(test_img, manipulation["type"], manipulation["val"])
                manipulation_type = manipulation["type"]

                if manipulation_type not in grouped_results:
                        grouped_results[manipulation_type] = []

                grouped_results[manipulation_type].append({
                "img": manipulated_img,
                "label": manipulation["label"]
                })

        types = list(grouped_results.keys())
        rows = len(types) # Row for each manipulation type
        cols = max(len(items) for items in grouped_results.values()) # Amount of columns will be the manipulation type with the most values

        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))

        for r, manipulation_type in enumerate(types):
                items = grouped_results[manipulation_type]

                for c in range(cols):
                        ax = axes[r, c]

                        if c < len(items):
                                ax.imshow(items[c]["img"])
                                ax.set_title(items[c]["label"], fontsize=6, fontweight="bold", pad=4)
                                ax.axis("off")
                else:
                        ax.axis("off")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.02, hspace=0.12)
        fig.suptitle("Image Manipulations", fontsize=10, fontweight="bold", y=1.0)
        plt.show()


def calculate_mse(img: Image.Image, inference_steps):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = "runwayml/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe.to(device)

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) # Denoising scheduler for reconstruction
        inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config) # Inverse scheduler for inversion

        inverse_scheduler.set_timesteps(inference_steps, device=device) # Setting the amount of timesteps in the inversion process

        img = img.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
        img = np.array(img).astype(np.float32) / 127.5 - 1.0 # Converting into array, where every value is inside [-1, 1]
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device) # Converting into tensor

        # Encoding to latent space VAE
        with torch.inference_mode():
                x_0 = pipe.vae.encode(img_tensor).latent_dist.mean
                x_0 *= pipe.vae.config.scaling_factor

        # Text embeddings (no prompt - unconditional)
        text_inputs = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")

        with torch.inference_mode():
                embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

        # Inversion
        x_t = x_0.clone()
        for t in inverse_scheduler.timesteps:
                with torch.inference_mode():
                        noise_prediction = pipe.unet(x_t, t, encoder_hidden_states=embeddings).sample
                x_t = inverse_scheduler.step(noise_prediction, t, x_t).prev_sample

        # Inversion
        with torch.inference_mode():
                recon_output = pipe(prompt="", latents=x_t, num_inference_steps=inference_steps, guidance_scale=3.0, output_type="latent")
                x_0_hat = recon_output.images

        # Calculating MSE
        diff = torch.abs(x_0 - x_0_hat)
        mse = (diff ** 2).mean().item()

        return mse