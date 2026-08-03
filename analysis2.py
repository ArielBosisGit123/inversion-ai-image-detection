import numpy as np
import pandas as pd

RESULTS_CSV_PATH = r"experiment2/results.csv"
results_df = pd.read_csv(RESULTS_CSV_PATH)

step_levels = sorted(results_df["steps"].unique())
avg_times = []
best_accuracies = []

# Time and accuracy
for step in step_levels:
        subset = results_df[results_df["steps"] == step]

        # Average time
        mean_time = subset["time"].mean()
        mean_ai_mse = subset[subset["AI"] == 1]["MSE"].mean()
        mean_real_mse = subset[subset["AI"] == 0]["MSE"].mean()
        avg_times.append(mean_time)

        # Best threshold
        mse_scores = subset["MSE"].values
        true_ai = subset["AI"].values
        thresholds = np.unique(mse_scores)

        min_errors = float("inf")
        best_thresh = 0.0

        for thresh in thresholds:
                predictions_ai = (mse_scores <= thresh).astype(int)
                total_errors = (predictions_ai != true_ai).sum()
                if total_errors < min_errors:
                        min_errors = total_errors
                        best_thresh = thresh

        # Accuracy
        num_images = len(subset)
        accuracy = 100 - (100 * min_errors / num_images)
        best_accuracies.append(accuracy)

        # Print table row
        print(f"Steps: {step:2d} | Avg Time: {mean_time:.2f}s | Avg AI / Real MSE: {mean_ai_mse:.4f} / {mean_real_mse:.4f} | Best Threshold: {best_thresh:.4f} | Accuracy: {accuracy:.1f}%")