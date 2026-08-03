# AI Image Detection Using Diffusion Inversion

## Experiments

* **1. White-Box:** Evaluating detection using `stable-diffusion-v1-5`.
* **2. Inference Steps:** Comparing accuracy across 10, 25, and 50 steps.
* **3. Robustness:** Testing robustness under JPEG compression, blur, crop, rotation, and resizing.
* **4. Black-Box:** Generalization across images from multiple models (`SD v1.5`, `SD v2.1`, `ADM`, `DDPM`).

## Data & Stack

* **Data:** 50 natural images (LSUN Bedrooms) + 200 synthetic images (50 per generator model).
* **Resources:** Python 3.12, PyTorch, HuggingFace `diffusers`, PIL/OpenCV, NumPy, Pandas, SciPy, Matplotlib.