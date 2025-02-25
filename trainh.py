import argparse
import time
import numpy as np
import torch
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from h_mscn.util import *
from h_mscn.data import get_train_datasets, load_data, make_dataset
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

def weighted_qerror_loss(preds, targets, min_val, max_val, threshold=10.0, penalty_factor=2.0):
    # Unnormalize predictions and targets
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    qerror = []
    for i in range(len(targets)):
        q = preds[i] / targets[i] if preds[i] > targets[i] else targets[i] / preds[i]
        
        # Apply penalty factor if Q-error exceeds the threshold
        if q > threshold:
            q *= penalty_factor  # Penalize high Q-errors more
        qerror.append(q)

    qerror = torch.stack(qerror)
    return torch.mean(qerror)


def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.0

    model.eval()
    with torch.no_grad():  # Disable gradient updates during testing
        for batch_idx, data_batch in enumerate(data_loader):
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

            if cuda:
                samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()

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

def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []

    for i in range(len(preds_unnorm)):
        pred = preds_unnorm[i].item() if isinstance(preds_unnorm[i], np.ndarray) else preds_unnorm[i]
        label = labels_unnorm[i].item() if isinstance(labels_unnorm[i], np.ndarray) else labels_unnorm[i]
        label = float(label) if isinstance(label, str) else label

        if pred > label:
            qerror.append(pred / label)
        else:
            qerror.append(label / pred)

    qerror = np.array(qerror, dtype=np.float64)

    print("Median: {:.4f}".format(np.median(qerror)))
    print("90th percentile: {:.4f}".format(np.percentile(qerror, 90)))
    print("95th percentile: {:.4f}".format(np.percentile(qerror, 95)))
    print("99th percentile: {:.4f}".format(np.percentile(qerror, 99)))
    print("Max: {:.4f}".format(np.max(qerror)))
    print("Mean: {:.4f}".format(np.mean(qerror)))

def train_and_predict(workload_name, num_queries, num_epochs, batch_size, hid_units, cuda):
    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts

    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConvWithAttention(sample_feats, predicate_feats, join_feats, hid_units, max_num_predicates, max_num_joins)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # Adaptive learning rate

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        loss_total = 0.0
        entropy_loss_total = 0.0  # Track entropy loss

        for batch_idx, data_batch in enumerate(train_data_loader):
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

            if cuda:
                samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
            samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
                targets)
            sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
                join_masks)

            optimizer.zero_grad()
            outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            #loss = qerror_loss(outputs, targets.float(), min_val, max_val)

            loss = weighted_qerror_loss(outputs, targets.float(), min_val, max_val, threshold=10.0, penalty_factor=2.0)

            loss_total += loss.item()
            loss.backward()  # Use the combined loss for backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Clip gradients to a max norm of 0.5
            optimizer.step()
        
        scheduler.step(loss_total / len(train_data_loader))  # Update learning rate
        print("Epoch {}/{}, Q-Error Loss: {:.6f}".format(epoch + 1, num_epochs, loss_total / len(train_data_loader)))

    preds_train, t_total = predict(model, train_data_loader, cuda)
    print("Prediction time per training sample: {:.6f} ms".format(t_total / len(labels_train) * 1000))

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per validation sample: {:.6f} ms".format(t_total / len(labels_test) * 1000))

    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

    print("\nQ-Error training set:")
    print_qerror(preds_train_unnorm, labels_train_unnorm)

    print("\nQ-Error validation set:")
    print_qerror(preds_test_unnorm, labels_test_unnorm)

    # Test workload evaluation
    print("\nEvaluating on workload: {}".format(workload_name))
    file_name = "workloads/" + workload_name
    joins, predicates, tables, samples, label = load_data(file_name, num_materialized_samples)

    samples_test = encode_samples(tables, samples, table2vec)
    predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    print("Number of test samples: {}".format(len(labels_test)))

    max_num_predicates = max([len(p) for p in predicates_test])
    max_num_joins = max([len(j) for j in joins_test])

    test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per test sample: {:.6f} ms".format(t_total / len(labels_test) * 1000))

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    print("\nQ-Error for workload '{}':".format(workload_name))
    print_qerror(preds_test_unnorm, label)

     # Write predictions
    file_name = "results/predictions_h_mscn" + workload_name + ".csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        for i in range(len(preds_test_unnorm)):
            f.write(str(preds_test_unnorm[i]) + "," + label[i] + "\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload_name", help="Name of the workload (synthetic, scale, job-light)")
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hid", type=int, default=512)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    train_and_predict(args.workload_name, args.queries, args.epochs, args.batch, args.hid, args.cuda)

if __name__ == "__main__":
    main()