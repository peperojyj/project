import argparse
import csv
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from h_mscn.data import get_train_datasets
from h_mscn.model import SetConvWithAttention

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)

def qerror_loss(preds, targets, min_val, max_val):
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    qerror = []
    for i in range(len(targets)):
        if preds[i] > targets[i]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    
    qerror = torch.stack(qerror)
    return torch.mean(qerror)

def predict(model, data_loader):
    preds = []
    t_total = 0.0

    model.eval()
    with torch.no_grad():  # Disable gradient updates during testing
        for batch_idx, data_batch in enumerate(data_loader):
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

            samples, predicates, joins, targets = (
                Variable(samples),
                Variable(predicates),
                Variable(joins),
                Variable(targets),
            )
            sample_masks, predicate_masks, join_masks = (
                Variable(sample_masks),
                Variable(predicate_masks),
                Variable(join_masks),
            )

            t = time.time()
            outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            t_total += time.time() - t

            for i in range(outputs.data.shape[0]):
                preds.append(outputs.data[i])

    return preds, t_total

def train_and_evaluate(workload_name, learning_rate, hidden_units, batch_size, num_epochs):
    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        100000, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts

    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConvWithAttention(sample_feats, predicate_feats, join_feats, hidden_units, max_num_predicates)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    print(f"Workload: {workload_name}, LR: {learning_rate}, HU: {hidden_units}, Batch Size: {batch_size}, Epochs: {num_epochs}")
    print("Number of training samples:", len(labels_train))
    print("Number of validation samples:", len(labels_test))

    model.train()
    for epoch in range(num_epochs):
        loss_total = 0.0

        for batch_idx, data_batch in enumerate(train_data_loader):
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

            samples, predicates, joins, targets = (
                Variable(samples),
                Variable(predicates),
                Variable(joins),
                Variable(targets),
            )
            sample_masks, predicate_masks, join_masks = (
                Variable(sample_masks),
                Variable(predicate_masks),
                Variable(join_masks),
            )

            optimizer.zero_grad()
            outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)

            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_total / len(train_data_loader):.4f}")

    preds_train, _ = predict(model, train_data_loader)
    preds_test, _ = predict(model, test_data_loader)

    train_loss = qerror_loss(torch.tensor(preds_train), torch.tensor(labels_train).float(), min_val, max_val).item()
    val_loss = qerror_loss(torch.tensor(preds_test), torch.tensor(labels_test).float(), min_val, max_val).item()
    return train_loss, val_loss

def hyperparameter_tuning(workloads, learning_rates, hidden_units, batch_sizes, epochs_list, results_file):
    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Workload", "Learning Rate", "Hidden Units", "Batch Size", "Epochs", "Train Loss", "Validation Loss"])

    for workload in workloads:
        for lr in learning_rates:
            for hu in hidden_units:
                for bs in batch_sizes:
                    for epochs in epochs_list:
                        print(f"Starting experiment: Workload={workload}, LR={lr}, HU={hu}, Batch Size={bs}, Epochs={epochs}")
                        train_loss, val_loss = train_and_evaluate(workload, lr, hu, bs, epochs)
                        with open(results_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([workload, lr, hu, bs, epochs, train_loss, val_loss])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=str, nargs="+", required=True, help="List of workloads to train on.")
    parser.add_argument("--learning_rates", type=float, nargs="+", required=True, help="List of learning rates.")
    parser.add_argument("--hidden_units", type=int, nargs="+", required=True, help="List of hidden unit sizes.")
    parser.add_argument("--batch_sizes", type=int, nargs="+", required=True, help="List of batch sizes.")
    parser.add_argument("--epochs", type=int, nargs="+", default=[100], help="List of epochs to try.")
    parser.add_argument("--results_file", type=str, default="tuning_results.csv", help="File to store tuning results.")
    args = parser.parse_args()

    hyperparameter_tuning(
        args.workloads, args.learning_rates, args.hidden_units, args.batch_sizes, args.epochs, args.results_file
    )

if __name__ == "__main__":
    main()
