import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

RESULTS_CSV_PATH = (
    r"/content/drive/MyDrive/inversion-ai-image-detection/experiment1/results.csv"
)
results_df = pd.read_csv(RESULTS_CSV_PATH)

# Threshold and accuracy
num_images = len(results_df)
mse_scores = results_df["MSE"]
true_ai = results_df["AI"]
best_threshold = 0.0
thresholds = [0.017]
min_errors = float("inf")

for idx, thresh in enumerate(thresholds):
    predictions_ai = (mse_scores <= thresh).astype(int)
    total_errors = (predictions_ai != true_ai).sum()
    if total_errors < min_errors:
        min_errors = total_errors
        best_threshold = thresh

print(
    f"Amount of errors: {min_errors}, threshold: {best_threshold}, accuracy: {100 - 100 * min_errors/num_images}%"
)

# Mann-Whitney U test
synthetic_group = results_df[results_df["AI"] == 1]["MSE"]
real_group = results_df[results_df["AI"] == 0]["MSE"]
stat, p_value = mannwhitneyu(synthetic_group, real_group, alternative="two-sided")
print(f"p-value: {p_value}")

# p-value asterisks
if p_value < 0.001:
    sig_label = "* * *"
elif p_value < 0.01:
    sig_label = "* *"
elif p_value < 0.05:
    sig_label = "*"
else:
    sig_label = "ns"

# Histogram
plt.figure(figsize=(7, 5))
plt.hist(
    mse_scores[true_ai == 0],
    bins=14,
    alpha=0.5,
    color="#4e79a7",
    label="lsun-bedrooms",
)
plt.hist(
    mse_scores[true_ai == 1],
    bins=14,
    alpha=0.5,
    color="#f28e2c",
    label="stable-diffusion-v1-5",
)


# Draw significance bracket
x1, x2 = 0.007, 0.035
y_base = 23.0  # Height above highest bin
tick_h = 0.8  # Height of the bracket ticks

# Bracket lines
plt.plot(
        [x1, x1, x2, x2],
        [y_base, y_base + tick_h, y_base + tick_h, y_base],
        color="black",
        lw=1.5,
)
# Text
plt.text(
        (x1 + x2) / 2,
        y_base + tick_h,
        sig_label,
        ha="center",
        va="bottom",
        fontsize=13,
        weight="bold",
)

plt.ylim(0, 26)

plt.title("Histogram of MSE Values")
plt.xlabel("MSE Value")
plt.ylabel("Frequency")
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(RESULTS_CSV_PATH), "hist1.png"), dpi=1000)
plt.show()