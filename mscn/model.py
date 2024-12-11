import torch
import torch.nn as nn
import torch.nn.functional as F

class SetConvWithAttention(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units, max_num_predicates):
        super(SetConvWithAttention, self).__init__()

        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)

        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)

        self.max_num_predicates = max_num_predicates
        self.attention_query = nn.Linear(hid_units, hid_units)
        self.attention_key = nn.Linear(hid_units, hid_units)
        self.attention_value = nn.Linear(hid_units, hid_units)
        self.attention_dropout = nn.Dropout(0.2)

        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)

        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        # Process sample features
        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask
        hid_sample = torch.sum(hid_sample, dim=1)
        sample_norm = sample_mask.sum(1)
        hid_sample = hid_sample / sample_norm

        # Process predicate features
        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))

        # Attention mechanism for predicates
        query = self.attention_query(hid_predicate)
        key = self.attention_key(hid_predicate)
        value = self.attention_value(hid_predicate)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (hid_predicate.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attended_predicates = torch.matmul(attention_weights, value)

        hid_predicate = hid_predicate + attended_predicates  # Residual connection
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1)
        predicate_norm = predicate_mask.sum(1)
        hid_predicate = hid_predicate / predicate_norm

        # Process join features
        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1)
        join_norm = join_mask.sum(1)
        hid_join = hid_join / join_norm

        # Combine all features
        hid = torch.cat((hid_sample, hid_predicate, hid_join), dim=1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out