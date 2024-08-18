import torch
import torch.nn as nn
from utilities.common_loss import KL_loss
from utilities.common_loss import logit_ML_loss
from model.RCML_View import RCML_View
from model.Attention import Attention
from model.RCML import RCML


class GTMC_HOA(nn.Module):

    def __init__(self, view_blocks, comm_feature_dim, num_classes, lambda_epochs, args, device):
        super(GTMC_HOA, self).__init__()
        self.view_blocks = nn.Sequential()
        self.view_blocks_codes = []
        for view_block in view_blocks:
            self.view_blocks.add_module(str(view_block.v_num), view_block)
            self.view_blocks_codes.append(str(view_block.v_num))

        self.args = args
        self.comm_feature_dim = comm_feature_dim
        self.num_views = len(self.view_blocks)
        self.num_classes = num_classes
        self.device = device

        self.fc_comm_extract = nn.Linear(self.comm_feature_dim, self.comm_feature_dim)
        self.discriminator = nn.Linear(self.comm_feature_dim, self.num_views)
        self.fc_comm_predictor = nn.Linear(self.comm_feature_dim, self.num_classes)

        self.RCML_views = nn.ModuleList(
            [RCML_View(self.num_classes, [self.comm_feature_dim], lambda_epochs, args.gamma) for _ in
             range(self.num_views)])

        self.attention = Attention(self.num_views, self.comm_feature_dim, self.num_classes)

        self.RCML = RCML(self.num_views, self.num_classes, lambda_epochs, args.gamma)

    def forward(self, x, labels, epoch):
        loss = 0.0  # All loss
        batch_size = x[0].shape[0]
        view_features_dict = self.extract_view_features(x)
        comm_feature = torch.zeros(batch_size, self.comm_feature_dim).to(self.device)
        final_features = torch.zeros(batch_size, self.num_views, self.comm_feature_dim).to(self.device)
        final_evidences = torch.zeros(batch_size, self.num_views, self.num_classes).to(self.device)

        GAN_loss = 0.0
        comm_ML_loss = 0.0
        for view_code, view_feature in view_features_dict.items():
            view_code = int(view_code)
            view_comm_feature = self.fc_comm_extract(view_feature[1])
            comm_feature += view_comm_feature

            GAN_loss += self.calculate_GAN_loss(view_comm_feature, view_code)
            comm_prediction = self.fc_comm_predictor(view_comm_feature)
            comm_loss = logit_ML_loss(comm_prediction, labels, self.num_classes)
            comm_ML_loss += comm_loss
        comm_feature /= self.num_views

        RCML_views_loss = 0.0
        for view_code, view_feature in view_features_dict.items():
            view_code = int(view_code)
            final_features[:, view_code, :] = view_feature[0] + comm_feature
            evidence_view, RCML_view_loss = self.RCML_views[view_code](comm_feature, view_feature[0], labels, epoch)
            final_evidences[:, view_code, :] = evidence_view
            RCML_views_loss += RCML_view_loss

        evidences_attention = self.attention(final_features, final_features, final_evidences)
        evidences, evidence_a, RCML_loss = self.RCML(evidences_attention, labels, epoch)

        # L(adv)
        GAN_loss /= self.num_views
        GAN_loss = torch.exp(-GAN_loss)
        # L(cml)
        comm_ML_loss /= self.num_views
        # L(com)
        loss += (GAN_loss + comm_ML_loss) * self.args.delta
        # L(spe)
        orthogonal_regularization = self.calculate_orthogonal_loss(view_features_dict, comm_feature)
        orthogonal_regularization /= self.num_views
        loss += orthogonal_regularization * self.args.eta
        # L(H1)
        RCML_views_loss /= self.num_views
        loss += RCML_views_loss
        # L(H2)
        loss += RCML_loss

        return evidences, evidence_a, loss

    def extract_view_features(self, x):
        view_features_dict = {}
        for view_blcok_code in self.view_blocks_codes:
            view_x = x[int(view_blcok_code)]
            view_block = self.view_blocks.__getattr__(view_blcok_code)
            view_features = view_block(view_x)
            view_features_dict[view_blcok_code] = view_features
        return view_features_dict

    def calculate_GAN_loss(self, view_comm_feature, view_code):
        pre_distributions = self.discriminator(view_comm_feature)
        true_distributions = torch.zeros(pre_distributions.shape).to(self.device)
        true_distributions[:, view_code] = 1.0
        loss = KL_loss(pre_distributions, true_distributions)
        return loss

    def calculate_orthogonal_loss(self, view_features_dict, comm_feature):
        loss = 0.0
        comm_feature_T = comm_feature.t()
        for _, view_feature in view_features_dict.items():
            item = view_feature[0].mm(comm_feature_T)
            item = item ** 2
            item = item.sum()
            loss += item
        loss /= (comm_feature.shape[0] * comm_feature.shape[0])
        return loss
