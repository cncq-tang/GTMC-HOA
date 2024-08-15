import torch.nn as nn
import torch.nn.functional as F


class ViewBlock(nn.Module):

    def __init__(self, v_num, input_feature_dim, comm_feature_dim):
        super(ViewBlock, self).__init__()
        self.v_num = v_num
        self.fc_extract_comm = nn.Linear(input_feature_dim, comm_feature_dim)
        self.fc_private = nn.Linear(input_feature_dim, comm_feature_dim)

    def forward(self, x_view):
        x_private = F.relu(self.fc_private(x_view))
        x_comm_feature = F.relu(self.fc_extract_comm(x_view))
        return x_private, x_comm_feature
