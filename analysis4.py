import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

RESULTS_CSV_PATH = r"experiment4/results.csv"

results_df = pd.read_csv(RESULTS_CSV_PATH)

real_df = results_df[results_df["AI"] == 0]
ai_origins = results_df[results_df["AI"] == 1]["origin_name"].unique()

# Find the optimal threshold, maximizing average accuracy
best_threshold = None
best_avg_accuracy = -1
best_model_accuracies = {}

for threshold in np.unique(results_df["MSE"]):

	real_correct = (real_df["MSE"] > threshold).sum()
	model_accuracies = {}

	for origin in ai_origins:
		ai_df = results_df[results_df["origin_name"] == origin]
		ai_correct = (ai_df["MSE"] <= threshold).sum()
		accuracy = ((real_correct + ai_correct) / (len(real_df) + len(ai_df)) * 100)
		model_accuracies[origin] = accuracy

	avg_accuracy = np.mean(list(model_accuracies.values()))

	if avg_accuracy > best_avg_accuracy:
		best_avg_accuracy = avg_accuracy
		best_threshold = threshold
		best_model_accuracies = model_accuracies

print(f"Best Threshold: {best_threshold:.6f}")
print(f"Average Accuracy: {best_avg_accuracy:.2f}%")


# Calculate balanced acc, F1 score, ROC & AUC

# AI (1) if MSE <= threshold, Real (0) if MSE > threshold
y_true = results_df["AI"]
y_pred = (results_df["MSE"] <= best_threshold).astype(int)

y_scores = -results_df["MSE"]

bal_acc = balanced_accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_scores)

print(f"\n--- Advanced Metrics ---")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"F1 Score:          {f1:.4f}")
print(f"ROC-AUC Score:     {roc_auc:.4f}")

# Plotting

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy per model
origins = list(best_model_accuracies.keys())[::-1]
accuracies = list(best_model_accuracies.values())[::-1]
bar_colors = ["#808080", "#4e79a7", "#59a14f", "#e15759"]

bars = axes[0].bar(origins, accuracies, color=bar_colors)
axes[0].tick_params(axis='x', rotation=20)
axes[0].set_ylim(0, 105)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title(f"Black-box Accuracies (Threshold = {best_threshold:.4f})")

for bar, acc in zip(bars, accuracies):
	axes[0].text(
		bar.get_x() + bar.get_width() / 2,
		acc + 1,
		f"{acc:.1f}%",
		ha="center"
	)

metrics_text = f"Balanced Acc: {bal_acc:.4f}\nF1 Score: {f1:.4f}"
axes[0].text(
	0.05, 0.95,
	metrics_text,
	transform=axes[0].transAxes,
	fontsize=11,
	verticalalignment='top',
	bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real (0)", "AI (1)"])
disp.plot(ax=axes[1], cmap="Blues", colorbar=False)
axes[1].set_title("Confusion Matrix")

# ROC curve & AUC
fpr, tpr, _ = roc_curve(y_true, y_scores)
axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[2].set_xlim([0.0, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('Receiver Operating Characteristic (ROC)')
axes[2].legend(loc="lower right")

plt.tight_layout()




save_dir = os.path.dirname(RESULTS_CSV_PATH)
if save_dir:
	os.makedirs(save_dir, exist_ok=True)
	plt.savefig(os.path.join(save_dir, "black_box.png"), dpi=1000)

plt.show()