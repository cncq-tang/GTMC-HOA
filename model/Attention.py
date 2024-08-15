import torch.nn as nn
from model.Dot_Attention import ScaledDotProductAttention


class Attention(nn.Module):

    def __init__(self, num_views, comm_feature_dim, num_classes, dropout=0.1):
        super(Attention, self).__init__()
        self.comm_feature_dim = comm_feature_dim
        self.num_views = num_views
        self.num_classes = num_classes

        self.linear_q = nn.Linear(self.comm_feature_dim, self.comm_feature_dim)
        self.linear_k = nn.Linear(self.comm_feature_dim, self.comm_feature_dim)
        self.linear_v = nn.Linear(self.num_classes, self.num_classes)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.SoftPlus = nn.Softplus()

    def forward(self, query, key, value):
        residual = value
        batch_size = query.shape[0]
        # linear projection
        query_reshaped = query.view(-1, self.comm_feature_dim)
        query = self.linear_q(query_reshaped)
        query = query.view(batch_size, self.num_views, self.comm_feature_dim)

        key_reshaped = key.view(-1, self.comm_feature_dim)
        key = self.linear_k(key_reshaped)
        key = key.view(batch_size, self.num_views, self.comm_feature_dim)

        value_reshaped = value.view(-1, self.num_classes)
        value = self.linear_v(value_reshaped)
        value = value.view(batch_size, self.num_views, self.num_classes)

        # scaled dot product attention
        scale = key.size(-1) ** (-0.5)
        attention = self.dot_product_attention(query, key, value, scale)
        # dropout
        output = self.dropout(attention)
        # add residual
        output = residual + output

        evidences_attention = {}
        for v_num in range(self.num_views):
            evidence_view = output[:, v_num, :]
            evidences_attention[v_num] = self.SoftPlus(evidence_view)
        return evidences_attention
