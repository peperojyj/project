import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from mscn.data import get_train_datasets
from mscn.model import SetConvWithAttention

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)

def qerror_loss(preds, targets, min_val, max_val):
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)
    qerror = torch.max(preds / targets, targets / preds)
    return torch.mean(qerror)

def print_qerror(preds, labels):
    qerrors = [max(p / l, l / p) for p, l in zip(preds, labels)]
    qerrors = np.array(qerrors)
    print(f"Median: {np.median(qerrors):.4f}")
    print(f"90th percentile: {np.percentile(qerrors, 90):.4f}")
    print(f"95th percentile: {np.percentile(qerrors, 95):.4f}")
    print(f"99th percentile: {np.percentile(qerrors, 99):.4f}")
    print(f"Max: {np.max(qerrors):.4f}")
    print(f"Mean: {np.mean(qerrors):.4f}")

def train_and_predict(workload_name, num_queries, num_epochs, batch_size, hid_units, cuda):
    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_dataset, test_dataset = get_train_datasets(
        num_queries, num_materialized_samples)

    table2vec, column2vec, op2vec, join2vec = dicts
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConvWithAttention(sample_feats, predicate_feats, join_feats, hid_units, max_num_predicates, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Number of training samples: {len(labels_train)}")
    print(f"Number of validation samples: {len(labels_test)}")

    for epoch in range(num_epochs):
        loss_total = 0.0
        for data_batch in train_data_loader:
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch
            if cuda:
                samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()

            optimizer.zero_grad()
            outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            loss = qerror_loss(outputs, targets, min_val, max_val)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_total / len(train_data_loader):.6f}")

    preds_train, labels_train_unnorm = [], []
    preds_test, labels_test_unnorm = [], []
    model.eval()

    # Training Data Evaluation
    for data_batch in train_data_loader:
        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch
        if cuda:
            samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()

        outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks).cpu().detach().numpy()
        preds_train.extend(unnormalize_torch(torch.tensor(outputs), min_val, max_val).numpy())
        labels_train_unnorm.extend(unnormalize_torch(targets, min_val, max_val).cpu().numpy())

    # Validation Data Evaluation
    for data_batch in test_data_loader:
        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch
        if cuda:
            samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()

        outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks).cpu().detach().numpy()
        preds_test.extend(unnormalize_torch(torch.tensor(outputs), min_val, max_val).numpy())
        labels_test_unnorm.extend(unnormalize_torch(targets, min_val, max_val).cpu().numpy())

    print("\nQ-Error training set:")
    print_qerror(preds_train, labels_train_unnorm)

    print("\nQ-Error validation set:")
    print_qerror(preds_test, labels_test_unnorm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload_name", help="Name of workload (synthetic, scale, or job-light)")
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    train_and_predict(args.workload_name, args.queries, args.epochs, args.batch, args.hid, args.cuda)

if __name__ == "__main__":
    main()
