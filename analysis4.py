import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV_PATH = r"experiment4/results.csv"

results_df = pd.read_csv(RESULTS_CSV_PATH)

real_df = results_df[results_df["AI"] == 0]

ai_origins = results_df[results_df["AI"] == 1]["origin_name"].unique()

# Threshold
best_threshold = None
best_avg_accuracy = -1
best_model_accuracies = {}

for threshold in np.unique(results_df["MSE"]):

        real_correct = (real_df["MSE"] > threshold).sum()

        model_accuracies = {}

        for origin in ai_origins:
                ai_df = results_df[results_df["origin_name"] == origin]

                ai_correct = (ai_df["MSE"] <= threshold).sum()

                accuracy = (
                (real_correct + ai_correct)
                / (len(real_df) + len(ai_df))
                 * 100
                )

                model_accuracies[origin] = accuracy

        avg_accuracy = np.mean(list(model_accuracies.values()))

        if avg_accuracy > best_avg_accuracy:
                best_avg_accuracy = avg_accuracy
                best_threshold = threshold
                best_model_accuracies = model_accuracies

print(f"Best Threshold: {best_threshold:.6f}")
print(f"Average Accuracy: {best_avg_accuracy:.2f}%")

# Plor figure
plt.figure(figsize=(8, 5))

origins = list(best_model_accuracies.keys())[::-1]
accuracies = list(best_model_accuracies.values())[::-1]

# colors
bar_colors = [
        "#808080",
        "#4e79a7",
        "#59a14f",
        "#e15759"
]

bars = plt.bar(origins, accuracies, color=bar_colors)

plt.ylim(0, 105)
plt.ylabel("Accuracy (%)")
plt.title(f"Black-box Accuracies (Threshold = {best_threshold:.4f})")

for bar, acc in zip(bars, accuracies):
        plt.text(
                bar.get_x() + bar.get_width() / 2,
                acc + 1,
                f"{acc:.1f}%",
                ha="center"
        )

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(RESULTS_CSV_PATH), "black_box.png"), dpi=1000)
plt.show()