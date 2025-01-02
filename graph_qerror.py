import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# File paths and workload names
results_dir = "results"
workloads = ["synthetic", "scale", "job-light"]

# Function to load and compute Q-Error
def load_qerror_from_results(file_path, model_name, workload_name):
    """
    Reads the prediction file and computes Q-Errors.
    :param file_path: Path to the predictions file.
    :param model_name: Name of the model (C1 or MSCN).
    :param workload_name: Workload name.
    :return: List of Q-Error values.
    """
    qerrors = []
    with open(file_path, "r") as f:
        for line in f:
            predicted, true = line.strip().split(",")
            predicted = float(predicted.strip("[]"))  # Remove brackets
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

# Load Q-Error results for all models and workloads
def load_qerrors_for_workload(results_dir, workload):
    """
    Loads Q-Errors for a specific workload for both models.
    """
    data = []
    # C1 Results
    c1_file = os.path.join(results_dir, f"predictions_h_mscn{workload}.csv")
    if os.path.exists(c1_file):
        data.extend(load_qerror_from_results(c1_file, "C1 (Our Model)", workload))

    # MSCN Results
    mscn_file = os.path.join(results_dir, f"predictions_mscn{workload}.csv")
    if os.path.exists(mscn_file):
        data.extend(load_qerror_from_results(mscn_file, "MSCN (Original)", workload))

    return pd.DataFrame(data)

# Generate histograms or line graphs for each workload
def generate_histograms(results_dir, workloads):
    os.makedirs("plots", exist_ok=True)

    for workload in workloads:
        print(f"Processing workload: {workload}")
        # Load Q-Error data for the specific workload
        qerror_data = load_qerrors_for_workload(results_dir, workload)

        if qerror_data.empty:
            print(f"No Q-Error data found for {workload}. Skipping...")
            continue

        # Plot histogram for the current workload
        sns.set(style="whitegrid", palette="muted")
        plt.figure(figsize=(10, 6))

        # Plot histogram for each model
        sns.histplot(
            data=qerror_data, 
            x="QError", 
            hue="Model", 
            bins=50, 
            log_scale=True, 
            kde=False, 
            element="step",
            palette={"C1 (Our Model)": "blue", "MSCN (Original)": "orange"}
        )

        plt.title(f"Q-Error Histogram Comparison ({workload})")
        plt.xlabel("QError (log scale)")
        plt.ylabel("Frequency")
        plt.legend(title="Model", labels=["C1 (Our Model)", "MSCN (Original)"], loc="upper right")

        # Save the plot
        output_file = f"plots/qerror_histogram_{workload}.png"
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        print(f"Histogram saved to '{output_file}'")

# Main function
def main():
    generate_histograms(results_dir, workloads)
    print("All histograms generated successfully!")

if __name__ == "__main__":
    main()
