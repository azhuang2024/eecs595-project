import matplotlib.pyplot as plt
import numpy as np

# Define model names and corresponding data
model_names = ["Random", "All Fiction", "Fine-tuned BERT"]

# Data for each metric by model (precision, recall, F1-score)
# Model A, Model B, Model C data for precision, recall, F1-score
precision_data = [0.85, 0.78, 0.80]
recall_data = [0.75, 0.85, 0.70]
f1_score_data = [0.80, 0.82, 0.75]

# Define bar width and positions for each group
bar_width = 0.25
x = np.arange(len(model_names))  # x-positions for each group

# Create bar plots for each metric, grouped by model
plt.bar(x - bar_width, precision_data, width=bar_width, label="Precision", align="center")
plt.bar(x, recall_data, width=bar_width, label="Recall", align="center")
plt.bar(x + bar_width, f1_score_data, width=bar_width, label="F1-Score", align="center")

# Add labels, title, and legend
plt.xlabel("Models")
plt.ylabel("Scores")
plt.title("Precision, Recall, F1-Score for Baselines and Model")
plt.xticks(x, model_names)  # Set the labels for each group
plt.legend()  # Display the legend with metric labels

# Show the bar plot
plt.savefig("graph.png")
