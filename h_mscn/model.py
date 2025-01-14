import torch
import torch.nn as nn
import torch.nn.functional as F

class SetConvWithAttention(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, hid_units, max_num_predicates, max_num_joins):
        super(SetConvWithAttention, self).__init__()

        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)

        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)

        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)

        # Self-Attention for predicates and joins
        self.predicate_self_attn = nn.MultiheadAttention(embed_dim=hid_units, num_heads=1, batch_first=True)
        self.join_self_attn = nn.MultiheadAttention(embed_dim=hid_units, num_heads=1, batch_first=True)

        # Cross-Attention for query -> predicates and query -> joins
        self.cross_attention_pred = nn.MultiheadAttention(embed_dim=hid_units, num_heads=1, batch_first=True)
        self.cross_attention_join = nn.MultiheadAttention(embed_dim=hid_units, num_heads=1, batch_first=True)

        # Entropy tracking attributes
        self.predicate_entropy = 0.0
        self.join_entropy = 0.0
        self.cross_pred_entropy = 0.0
        self.cross_join_entropy = 0.0

        # Added dropout layers for attention
        self.predicate_attn_dropout = nn.Dropout(0.1)
        self.join_attn_dropout = nn.Dropout(0.1)
        self.cross_pred_attn_dropout = nn.Dropout(0.1)
        self.cross_join_attn_dropout = nn.Dropout(0.1) 
    
        self.query_projection = nn.Linear(hid_units * 2, hid_units)  # Reduces combined_query (1024) to hid_units (512)

        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    '''def compute_entropy(self, attn_weights):
        # Compute entropy for attention weights.
        attn_weights = attn_weights + 1e-8  # Avoid log(0)
        entropy = -torch.sum(attn_weights * torch.log(attn_weights), dim=-1)  # Sum over keys
       return entropy'''

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
       
        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask
        hid_sample = torch.sum(hid_sample, dim=1) / sample_mask.sum(1)

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))

        # Ensure predicate and join masks are 2-D
        predicate_mask = predicate_mask.squeeze(-1) if predicate_mask.dim() == 3 else predicate_mask
        join_mask = join_mask.squeeze(-1) if join_mask.dim() == 3 else join_mask

        # Self-Attention: refine predicates
        refined_predicates, pred_attn_weights = self.predicate_self_attn(
            hid_predicate, hid_predicate, hid_predicate, key_padding_mask=~predicate_mask.bool()
        )
        '''self.predicate_entropy = self.compute_entropy(pred_attn_weights)  #### Track predicate entropy'''

        # Self-Attention: refine joins
        refined_joins, join_attn_weights = self.join_self_attn(
            hid_join, hid_join, hid_join, key_padding_mask=~join_mask.bool()
        )
        '''self.join_entropy = self.compute_entropy(join_attn_weights)  #### Track join entropy'''


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
        attended_predicates, cross_pred_attn_weights = self.cross_attention_pred(
            query_vector, refined_predicates, refined_predicates, key_padding_mask=~predicate_mask.bool()
        )
        '''self.cross_pred_entropy = self.compute_entropy(cross_pred_attn_weights)  #### Track cross-predicate entropy'''

        # Cross-Attention: Query -> Refined Joins
        attended_joins, cross_join_attn_weights = self.cross_attention_join(
            query_vector, refined_joins, refined_joins, key_padding_mask=~join_mask.bool()
        )
        '''self.cross_join_entropy = self.compute_entropy(cross_join_attn_weights)  #### Track cross-join entropy'''

        # Remove sequence dimension after attention
        attended_predicates = attended_predicates.squeeze(1) 
        attended_joins = attended_joins.squeeze(1)

        # Combine all features
        combined_features = torch.cat((hid_sample, attended_predicates, attended_joins), dim=1)

        # Final MLP for prediction
        hid = F.relu(self.out_mlp1(combined_features))
        out = torch.sigmoid(self.out_mlp2(hid))

        return out
    
    '''def entropy_loss(self, lambda_entropy):
        # Compute the entropy regularization loss 
        total_entropy = torch.mean(self.predicate_entropy) + torch.mean(self.join_entropy) + \
                        torch.mean(self.cross_pred_entropy) + torch.mean(self.cross_join_entropy)
        return -lambda_entropy * total_entropy  # Negative sign to maximize entropy'''
