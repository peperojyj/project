import torch
import torch.nn as nn
import torch.nn.functional as F

class SetConvWithAttention(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units, max_num_predicates, max_num_joins):
        super(SetConvWithAttention, self).__init__()

        # MLP layers for sample features
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)

        # MLP layers for predicate features
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)

        # MLP layers for join features
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)

        # Self-Attention for predicates and joins
        self.predicate_self_attn = nn.MultiheadAttention(embed_dim=hid_units, num_heads=1, batch_first=True)
        self.join_self_attn = nn.MultiheadAttention(embed_dim=hid_units, num_heads=1, batch_first=True)

        # Cross-Attention for query -> predicates and query -> joins
        self.cross_attention_pred = nn.MultiheadAttention(embed_dim=hid_units, num_heads=1, batch_first=True)
        self.cross_attention_join = nn.MultiheadAttention(embed_dim=hid_units, num_heads=1, batch_first=True)

        # Query projection to align dimensions
        self.query_projection = nn.Linear(hid_units * 2, hid_units)  # Reduces combined_query (1024) to hid_units (512)

        # Output MLP layers
        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        # Process sample features
        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask
        hid_sample = torch.sum(hid_sample, dim=1) / sample_mask.sum(1)

        # Process predicate features
        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))

        # Process join features
        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))

        # Ensure predicate and join masks are 2-D
        predicate_mask = predicate_mask.squeeze(-1) if predicate_mask.dim() == 3 else predicate_mask
        join_mask = join_mask.squeeze(-1) if join_mask.dim() == 3 else join_mask

        # Self-Attention: refine predicates
        refined_predicates, _ = self.predicate_self_attn(
            hid_predicate, hid_predicate, hid_predicate, key_padding_mask=~predicate_mask.bool()
        )

        # Self-Attention: refine joins
        refined_joins, _ = self.join_self_attn(
            hid_join, hid_join, hid_join, key_padding_mask=~join_mask.bool()
        )

        # Aggregate refined predicates
        predicate_mask_expanded = predicate_mask.unsqueeze(-1).expand_as(refined_predicates)
        predicate_agg = torch.sum(refined_predicates * predicate_mask_expanded, dim=1) / (predicate_mask.sum(1, keepdim=True) + 1e-8)

        # Aggregate refined joins
        join_mask_expanded = join_mask.unsqueeze(-1).expand_as(refined_joins)
        join_agg = torch.sum(refined_joins * join_mask_expanded, dim=1) / (join_mask.sum(1, keepdim=True) + 1e-8)

        # Combine query vector
        combined_query = torch.cat((predicate_agg, join_agg), dim=-1)  # [batch_size, hidden_dim * 2]
        query_vector = self.query_projection(combined_query).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Cross-Attention: Query -> Refined Predicates
        attended_predicates, _ = self.cross_attention_pred(
            query_vector, refined_predicates, refined_predicates, key_padding_mask=~predicate_mask.bool()
        )

        # Cross-Attention: Query -> Refined Joins
        attended_joins, _ = self.cross_attention_join(
            query_vector, refined_joins, refined_joins, key_padding_mask=~join_mask.bool()
        )

        # Remove sequence dimension after attention
        attended_predicates = attended_predicates.squeeze(1)
        attended_joins = attended_joins.squeeze(1)

        # Combine all features
        combined_features = torch.cat((hid_sample, attended_predicates, attended_joins), dim=1)

        # Final MLP for prediction
        hid = F.relu(self.out_mlp1(combined_features))
        out = torch.sigmoid(self.out_mlp2(hid))

        return out