import torch.nn as nn
from utilities.loss_function import *


class RCML(nn.Module):

    def __init__(self, num_views, num_classes, lambda_epochs, gamma):
        super(RCML, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.lambda_epochs = lambda_epochs
        self.gamma = gamma

    def forward(self, evidences, y, global_step):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        evidence_a = evidences[0]
        for i in range(1, self.num_views):
            evidence_a = (evidences[i] + evidence_a) / 2
        loss = get_loss(evidences, evidence_a, y, global_step, self.num_classes, self.lambda_epochs, self.gamma, device)
        return evidences, evidence_a, loss
