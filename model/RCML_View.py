import torch.nn as nn
from utilities.loss_function import *
from EvidenceCollector import EvidenceCollector


class RCML_View(nn.Module):

    def __init__(self, num_classes, comm_feature_dim, lambda_epochs, gamma):
        super(RCML_View, self).__init__()
        self.num_classes = num_classes
        self.comm_feature_dim = comm_feature_dim
        self.lambda_epochs = lambda_epochs
        self.gamma = gamma
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(self.comm_feature_dim, self.num_classes) for _ in range(2)])

    def forward(self, comm_feature, private_feature, y, global_step):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        view_feature = dict()
        view_feature[0] = comm_feature
        view_feature[1] = private_feature
        evidences = self.infer(view_feature)
        evidence_a = (evidences[0] + evidences[1]) / 2
        loss = get_loss(evidences, evidence_a, y, global_step, self.num_classes, self.lambda_epochs, self.gamma, device)
        return evidence_a, loss

    def infer(self, x):
        evidence = dict()
        for num in range(2):
            evidence[num] = self.EvidenceCollectors[num](x[num])
        return evidence
