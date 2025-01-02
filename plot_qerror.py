import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import numpy as np

# File paths and workload names
results_dir = "results"
workloads = ["synthetic", "scale", "job-light"]

# Function to load and compute Q-Error
def load_qerror_from_results(file_path, model_name, workload_name):
    qerrors = []
    with open(file_path, "r") as f:
        for line in f:
            predicted, true = line.strip().split(",")
            predicted = float(predicted.strip("[]"))
            true = float(true)

            # Compute Q-Error
            if predicted > true and true > 0:
                qerror = predicted / true
            elif true > predicted and predicted > 0:
                qerror = true / predicted
            else:
                qerror = np.inf  # Avoid division by zero

            qerrors.append({"QError": qerror, "Model": model_name, "Workload": workload_name})
    return qerrors

# Load Q-Error results for a specific workload
def load_qerrors_for_workload(results_dir, workload):
    data = []
    # Our Model Results
    our_model_file = os.path.join(results_dir, f"predictions_h_mscn{workload}.csv")
    if os.path.exists(our_model_file):
        data.extend(load_qerror_from_results(our_model_file, "Our Model", workload))

    # MSCN Results
    mscn_file = os.path.join(results_dir, f"predictions_mscn{workload}.csv")
    if os.path.exists(mscn_file):
        data.extend(load_qerror_from_results(mscn_file, "MSCN", workload))

    return pd.DataFrame(data)

# Generate and save boxplot for a specific workload
def generate_boxplot_for_workload(results_dir, workload):
    os.makedirs("plots", exist_ok=True)

    print(f"Processing workload: {workload}")
    qerror_data = load_qerrors_for_workload(results_dir, workload)

    if qerror_data.empty:
        print(f"No Q-Error data found for {workload}. Skipping...")
        return

    # Plot boxplot for the specific workload
    sns.set(style="whitegrid", palette="muted")
    plt.figure(figsize=(8, 6))

    # Custom color mapping
    color_palette = {"Our Model": "green", "MSCN": "blue"}

    # Create boxplot
    sns.boxplot(
        x="Model",
        y="QError",
        data=qerror_data,
        palette=color_palette,
        showfliers=True
    )

    # Use log scale for QError
    plt.yscale("log")

    # Bold axis labels
    plt.ylabel("QError", fontsize=14, fontweight="bold")
    plt.xlabel("Model", fontsize=14, fontweight="bold")

    # Add legend
    legend_handles = [
        mpatches.Patch(color="green", label="Our Model"),
        mpatches.Patch(color="blue", label="MSCN")
    ]
    plt.legend(handles=legend_handles, loc="upper left", fontsize=12)

    # Save the plot
    output_file = f"plots/qerror_{workload}.png"
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Box plot for workload '{workload}' saved to '{output_file}'")

# Main function
def main():
    for workload in workloads:
        generate_boxplot_for_workload(results_dir, workload)
    print("All box plots generated successfully!")

if __name__ == "__main__":
    main()
