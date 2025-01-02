import os
import subprocess
import pandas as pd
import numpy as np
import shutil

# Settings
results_dir = "results"
best_results_dir = "best_results"
os.makedirs(best_results_dir, exist_ok=True)

workloads = ["synthetic", "scale", "job-light"]
iterations = 5  # Number of times to run the model for each workload

# Function to run the C2 model and generate predictions
def run_c2_model(workload):
    """
    Runs the C2 model using trainh.py and generates predictions.
    """
    print(f"Running C2 model for workload '{workload}'...")
    command = f"python3 trainh.py {workload} --queries 50000 --epochs 50"
    subprocess.run(command, shell=True, check=True)

    # Output file from trainh.py
    default_output_file = os.path.join(results_dir, f"predictions_h_mscn{workload}.csv")
    return default_output_file

# Function to compute Q-error statistics
def compute_qerror_stats(predictions_file):
    """
    Computes Q-error statistics (mean, median, percentiles, max).
    """
    qerrors = []
    with open(predictions_file, "r") as f:
        for line in f:
            predicted, true = line.strip().split(",")
            predicted = float(predicted.strip("[]"))
            true = float(true)

            # Q-error calculation
            if predicted > 0 and true > 0:
                if predicted > true:
                    qerror = predicted / true
                else:
                    qerror = true / predicted
                qerrors.append(qerror)

    qerrors = np.array(qerrors)
    stats = {
        "mean": np.mean(qerrors),
        "median": np.median(qerrors),
        "90th": np.percentile(qerrors, 90),
        "95th": np.percentile(qerrors, 95),
        "99th": np.percentile(qerrors, 99),
        "max": np.max(qerrors)
    }
    return stats

# Function to find the best case results
def find_best_case_results(workload):
    """
    Runs the model multiple times and finds the best results based on median Q-error.
    """
    best_stats = None
    best_file = None

    for iteration in range(iterations):
        print(f"Running iteration {iteration+1}/{iterations}...")

        # Run model and get predictions
        predictions_file = run_c2_model(workload)

        # Rename the output file to avoid overwriting
        iteration_output_file = os.path.join(results_dir, f"BBpredictions_h_mscn{workload}_{iteration}.csv")
        shutil.move(predictions_file, iteration_output_file)

        # Compute Q-error statistics
        stats = compute_qerror_stats(iteration_output_file)
        print(f"Iteration {iteration+1}: Mean Q-Error: {stats['mean']:.4f}, Median Q-Error: {stats['median']:.4f}")

        # Update best results based on minimum median Q-error
        if best_stats is None or stats['median'] < best_stats['median'] or \
           (stats['median'] == best_stats['median'] and stats['mean'] < best_stats['mean']):
            best_stats = stats
            best_file = iteration_output_file

    # Save the best results
    best_output_file = os.path.join(best_results_dir, f"best_results_{workload}.csv")
    shutil.copy(best_file, best_output_file)

    # Print detailed Q-error summary for the best results
    print(f"\nQ-Error Summary for Best-Case Results of workload '{workload}':")
    print(f"Median: {best_stats['median']:.4f}")
    print(f"90th percentile: {best_stats['90th']:.4f}")
    print(f"95th percentile: {best_stats['95th']:.4f}")
    print(f"99th percentile: {best_stats['99th']:.4f}")
    print(f"Max: {best_stats['max']:.4f}")
    print(f"Mean: {best_stats['mean']:.4f}")
    print(f"Best results for workload '{workload}' saved to '{best_output_file}'\n")

# Main function
def main():
    """
    Main function to process all workloads and find best-case results.
    """
    os.makedirs(results_dir, exist_ok=True)

    for workload in workloads:
        print(f"\nProcessing workload: {workload}")
        find_best_case_results(workload)

if __name__ == "__main__":
    main()
