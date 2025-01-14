import torch
import torch.nn as nn
import torch.nn.functional as F

class SetConvWithAttention(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units, max_num_predicates, max_num_joins):
        super(SetConvWithAttention, self).__init__()

        # MLPs for processing sample, predicate, and join features
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)

        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)

        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)

        # Enhanced multi-head attention
        self.predicate_self_attn = nn.MultiheadAttention(embed_dim=hid_units, num_heads=4, batch_first=True)
        self.join_self_attn = nn.MultiheadAttention(embed_dim=hid_units, num_heads=4, batch_first=True)
        self.cross_attention_pred = nn.MultiheadAttention(embed_dim=hid_units, num_heads=4, batch_first=True)
        self.cross_attention_join = nn.MultiheadAttention(embed_dim=hid_units, num_heads=4, batch_first=True)

        # Positional embeddings
        self.predicate_positional_embedding = nn.Parameter(torch.randn(max_num_predicates, hid_units))
        self.join_positional_embedding = nn.Parameter(torch.randn(max_num_joins, hid_units))

        # Dropout for regularization
        self.predicate_attn_dropout = nn.Dropout(0.1)
        self.join_attn_dropout = nn.Dropout(0.1)
        self.cross_pred_attn_dropout = nn.Dropout(0.1)
        self.cross_join_attn_dropout = nn.Dropout(0.1)

        # Query projection and output layers
        self.query_projection = nn.Linear(hid_units * 2, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        # Process sample features
        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask
        hid_sample = torch.sum(hid_sample, dim=1) / (sample_mask.sum(1) + 1e-8)

        # Process predicate features
        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))

        # Process join features
        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))

        # Adjust positional embeddings dynamically
        batch_size, num_predicates, _ = hid_predicate.size()
        batch_size, num_joins, _ = hid_join.size()

        predicate_pos_embedding = self.predicate_positional_embedding[:num_predicates, :]
        join_pos_embedding = self.join_positional_embedding[:num_joins, :]

        # Pad embeddings if needed
        if num_predicates > predicate_pos_embedding.size(0):
            predicate_pos_embedding = F.pad(predicate_pos_embedding, (0, 0, 0, num_predicates - predicate_pos_embedding.size(0)))
        if num_joins > join_pos_embedding.size(0):
            join_pos_embedding = F.pad(join_pos_embedding, (0, 0, 0, num_joins - join_pos_embedding.size(0)))

        # Expand embeddings to batch size
        predicate_pos_embedding = predicate_pos_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        join_pos_embedding = join_pos_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        hid_predicate = hid_predicate + predicate_pos_embedding
        hid_join = hid_join + join_pos_embedding

        # Ensure key_padding_mask is 2-D
        predicate_mask = predicate_mask.squeeze(-1) if predicate_mask.dim() == 3 else predicate_mask
        join_mask = join_mask.squeeze(-1) if join_mask.dim() == 3 else join_mask

        # Self-attention on predicates and joins
        refined_predicates, _ = self.predicate_self_attn(
            hid_predicate, hid_predicate, hid_predicate, key_padding_mask=~predicate_mask.bool()
        )
        refined_predicates = self.predicate_attn_dropout(refined_predicates)

        refined_joins, _ = self.join_self_attn(
            hid_join, hid_join, hid_join, key_padding_mask=~join_mask.bool()
        )
        refined_joins = self.join_attn_dropout(refined_joins)

        # Aggregate refined features
        predicate_mask_expanded = predicate_mask.unsqueeze(-1).expand_as(refined_predicates)
        predicate_agg = torch.sum(refined_predicates * predicate_mask_expanded, dim=1) / (predicate_mask.sum(1, keepdim=True) + 1e-8)

        join_mask_expanded = join_mask.unsqueeze(-1).expand_as(refined_joins)
        join_agg = torch.sum(refined_joins * join_mask_expanded, dim=1) / (join_mask.sum(1, keepdim=True) + 1e-8)

        # Combine query vector
        combined_query = torch.cat((predicate_agg, join_agg), dim=-1)
        query_vector = self.query_projection(combined_query).unsqueeze(1)

        # Cross-attention
        attended_predicates, _ = self.cross_attention_pred(
            query_vector, refined_predicates, refined_predicates, key_padding_mask=~predicate_mask.bool()
        )
        attended_predicates = self.cross_pred_attn_dropout(attended_predicates)

        attended_joins, _ = self.cross_attention_join(
            query_vector, refined_joins, refined_joins, key_padding_mask=~join_mask.bool()
        )
        attended_joins = self.cross_join_attn_dropout(attended_joins)

        # Remove sequence dimension
        attended_predicates = attended_predicates.squeeze(1)
        attended_joins = attended_joins.squeeze(1)

        # Final feature combination
        combined_features = torch.cat((hid_sample, attended_predicates, attended_joins), dim=1)

        # Final MLP for prediction
        hid = F.relu(self.out_mlp1(combined_features))
        out = torch.sigmoid(self.out_mlp2(hid))

        return out