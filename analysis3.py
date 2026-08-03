import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV_PATH = r"experiment3/results.csv"
df = pd.read_csv(RESULTS_CSV_PATH, names=["file_name", "AI", "MSE", "manipulation"])

df["MSE"] = pd.to_numeric(df["MSE"], errors='coerce')
df["AI"] = pd.to_numeric(df["AI"], errors='coerce')
df = df.dropna(subset=["MSE", "AI"])

# Thresholds
thresholds = [0.0228]
global_threshold = thresholds[0]

categories = df["manipulation"].unique()
results_summary = []

for cat in categories:
    subset = df[df["manipulation"] == cat]
    y_true = subset["AI"].values
    y_pred = (subset["MSE"].values <= global_threshold).astype(int)

    accuracy = (y_pred == y_true).mean() * 100
    results_summary.append({"Manipulation": cat, "Accuracy": accuracy})

results_df = pd.DataFrame(results_summary)

# Order
custom_order = [
    "none",
    "JPEG-Quality-85", "JPEG-Quality-70",
    "Blur-Radius-0.5px", "Blur-Radius-1px",
    "Crop-20%", "Crop-40%",
    "Rotation-20deg", "Rotation-40deg",
    "Scale-140%", "Scale-60%"
]

results_df["Manipulation"] = pd.Categorical(
    results_df["Manipulation"], categories=custom_order, ordered=True
)
results_df = results_df.sort_values("Manipulation")

# Colors
color_map = {
    "none": "#808080",
    "JPEG": "#4e79a7",
    "Blur": "#59a14f",
    "Crop": "#e15759",
    "Rotation": "#f28e2c",
    "Scale": "#b07aa1"
}

def get_color(label):
    for key in color_map:
        if key in label: return color_map[key]
    return "#9c9c9c"

bar_colors = [get_color(cat) for cat in results_df["Manipulation"]]

# Plot figure
plt.figure(figsize=(12, 6))
bars = plt.bar(results_df["Manipulation"], results_df["Accuracy"], color=bar_colors, alpha=1.0, edgecolor="none")

# Data labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}%",
             ha="center", va="bottom", fontsize=10)

plt.title(f"Manipulation Accuracies (Threshold = {global_threshold:.4f})", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(0, 110)
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(RESULTS_CSV_PATH), "robustness.png"), dpi=1000)
plt.show()