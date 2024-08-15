import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.SoftPlus = nn.Softplus()

    def forward(self, q, k, v, scale):
        similarity = torch.bmm(q, k.transpose(1, 2))
        if scale:
            similarity = similarity * scale
        similarity = (self.SoftPlus(similarity) + 1e-8)
        row_sum = torch.sum(similarity, dim=2, keepdim=True)
        similarity = similarity / row_sum
        # add dropout
        similarity = self.dropout(similarity)
        attention = torch.bmm(similarity, v)
        return attention
