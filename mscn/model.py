import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, mask):
        attn_output, _ = self.multihead_attn(x, x, x, key_padding_mask=mask)
        return attn_output

class SetConvWithAttention(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units, max_num_predicates, num_heads=4):
        super(SetConvWithAttention, self).__init__()
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)

        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)

        self.predicate_attention = AttentionLayer(dim=hid_units, num_heads=num_heads)

        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)

        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        # Process sample features
        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask
        hid_sample = torch.sum(hid_sample, dim=1) / (sample_mask.sum(dim=1) + 1e-6)

        # Process predicate features
        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))

        # Apply attention
        predicate_mask_flat = ~predicate_mask.squeeze(-1).bool()
        attn_predicates = self.predicate_attention(hid_predicate, predicate_mask_flat)
        hid_predicate = torch.sum(attn_predicates, dim=1) / (predicate_mask.sum(dim=1) + 1e-6)

        # Process join features
        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1) / (join_mask.sum(dim=1) + 1e-6)

        # Combine all features
        hid = torch.cat((hid_sample, hid_predicate, hid_join), dim=1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out
