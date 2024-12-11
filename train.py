import argparse
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F

from mscn.util import *
from mscn.data import get_train_datasets, load_data, make_dataset
from mscn.model import SetConvWithAttention

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

def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.0

    model.eval()
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

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))

def train_and_predict(workload_name, num_queries, num_epochs, batch_size, hid_units, cuda):
    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts

    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConvWithAttention(sample_feats, predicate_feats, join_feats, hid_units, max_num_predicates)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    model.train()
    for epoch in range(num_epochs):
        loss_total = 0.0

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
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)

            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {}/{}, Loss: {:.6f}".format(epoch + 1, num_epochs, loss_total / len(train_data_loader)))

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
    print("")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload_name", help="Name of the workload (synthetic, scale, job-light)")
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    train_and_predict(args.workload_name, args.queries, args.epochs, args.batch, args.hid, args.cuda)

if __name__ == "__main__":
    main()
