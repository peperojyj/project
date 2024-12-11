import csv
import torch
from torch.utils.data import dataset
from mscn.util import *

def load_data(file_name, num_materialized_samples):
    joins, predicates, tables, samples, label = [], [], [], [], []

    with open(file_name + ".csv", 'r') as f:
        data_raw = list(csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            label.append(row[3])
    print("Loaded queries")

    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(file_name + ".bitmaps", 'rb') as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")

    predicates = [list(chunks(d, 3)) for d in predicates]
    return joins, predicates, tables, samples, label

def load_and_encode_train_data(num_queries, num_materialized_samples):
    file_name_queries = "data/train"
    file_name_column_min_max_vals = "data/column_min_max_vals.csv"
    joins, predicates, tables, samples, label = load_data(file_name_queries, num_materialized_samples)

    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)

    table_names = get_all_table_names(tables)
    table2vec, idx2table = get_set_encoding(table_names)

    operators = get_all_operators(predicates)
    op2vec, idx2op = get_set_encoding(operators)

    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    with open(file_name_column_min_max_vals, 'r') as f:
        data_raw = list(csv.reader(f, delimiter=','))
        column_min_max_vals = {row[0]: [float(row[1]), float(row[2])] for row in data_raw[1:]}

    samples_enc = encode_samples(tables, samples, table2vec)
    predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    label_norm, min_val, max_val = normalize_labels(label)

    num_train = int(num_queries * 0.9)
    num_test = num_queries - num_train

    samples_train, predicates_train, joins_train, labels_train = (
        samples_enc[:num_train],
        predicates_enc[:num_train],
        joins_enc[:num_train],
        label_norm[:num_train],
    )

    samples_test, predicates_test, joins_test, labels_test = (
        samples_enc[num_train:num_train + num_test],
        predicates_enc[num_train:num_train + num_test],
        joins_enc[num_train:num_train + num_test],
        label_norm[num_train:num_train + num_test],
    )

    max_num_joins = max(max(len(j) for j in joins_train), max(len(j) for j in joins_test))
    max_num_predicates = max(max(len(p) for p in predicates_train), max(len(p) for p in predicates_test))

    dicts = [table2vec, column2vec, op2vec, join2vec]
    train_data = [samples_train, predicates_train, joins_train]
    test_data = [samples_test, predicates_test, joins_test]
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data

def make_dataset(samples, predicates, joins, labels, max_num_joins, max_num_predicates):
    def pad_and_stack(data, max_len, feature_dim):
        tensors, masks = [], []
        for item in data:
            tensor = np.vstack(item)
            num_pad = max_len - tensor.shape[0]
            tensor = np.pad(tensor, ((0, num_pad), (0, 0)), 'constant')
            mask = np.pad(np.ones_like(item).mean(1, keepdims=True), ((0, num_pad), (0, 0)), 'constant')
            tensors.append(np.expand_dims(tensor, 0))
            masks.append(np.expand_dims(mask, 0))
        return torch.FloatTensor(np.vstack(tensors)), torch.FloatTensor(np.vstack(masks))

    sample_tensors, sample_masks = pad_and_stack(samples, max_num_joins + 1, samples[0][0].shape[0])
    predicate_tensors, predicate_masks = pad_and_stack(predicates, max_num_predicates, predicates[0][0].shape[0])
    join_tensors, join_masks = pad_and_stack(joins, max_num_joins, joins[0][0].shape[0])

    return dataset.TensorDataset(
        sample_tensors, predicate_tensors, join_tensors, torch.FloatTensor(labels),
        sample_masks, predicate_masks, join_masks,
    )

def get_train_datasets(num_queries, num_materialized_samples):
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = load_and_encode_train_data(
        num_queries, num_materialized_samples)
    train_dataset = make_dataset(*train_data, labels=labels_train, max_num_joins=max_num_joins,
                                 max_num_predicates=max_num_predicates)
    test_dataset = make_dataset(*test_data, labels=labels_test, max_num_joins=max_num_joins,
                                max_num_predicates=max_num_predicates)
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_dataset, test_dataset
