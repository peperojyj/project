import torch
import torch.nn as nn
import torch.nn.functional as F

class SetConv(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units, max_num_predicates):
        super(SetConv, self).__init__()
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)

        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)

        self.max_num_predicates = max_num_predicates
        self.predicate_corr_mlp1 = nn.Linear(max_num_predicates * max_num_predicates, hid_units)
        self.predicate_corr_mlp2 = nn.Linear(hid_units, hid_units)

        self.corr_weight = nn.Parameter(torch.tensor(0.01))  # Start with a small value
        self.corr_dropout = nn.Dropout(0.2)  # Dropout for correlation features

        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)

        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)  # Removed direct inclusion of `hid_corr`
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
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1)
        predicate_norm = predicate_mask.sum(1)
        hid_predicate = hid_predicate / predicate_norm

        # Process predicate correlations
        predicate_corr = torch.bmm(predicates, predicates.transpose(1, 2))  # Pairwise correlations
        predicate_corr = torch.abs(predicate_corr)  # Non-negative values

        # Apply threshold for noise filtering
        threshold = 0.1  # Adjustable
        predicate_corr = torch.where(predicate_corr > threshold, predicate_corr, torch.tensor(0.0, device=predicate_corr.device))

        # Normalize correlation matrix
        predicate_corr /= (predicate_corr.norm(dim=(1, 2), keepdim=True) + 1e-6)

        # Flatten and pass through MLP
        predicate_corr_flat = predicate_corr.view(predicate_corr.size(0), -1)
        expected_size = self.max_num_predicates * self.max_num_predicates
        if predicate_corr_flat.size(1) < expected_size:
            pad_size = expected_size - predicate_corr_flat.size(1)
            predicate_corr_flat = F.pad(predicate_corr_flat, (0, pad_size))
        elif predicate_corr_flat.size(1) > expected_size:
            predicate_corr_flat = predicate_corr_flat[:, :expected_size]

        hid_corr = F.relu(self.predicate_corr_mlp1(predicate_corr_flat))
        hid_corr = F.relu(self.predicate_corr_mlp2(hid_corr))
        hid_corr = self.corr_dropout(hid_corr)  # Apply dropout to `hid_corr`

        # Weighted sum with learnable scalar
        hid_predicate += self.corr_weight * hid_corr

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
