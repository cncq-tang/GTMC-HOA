import torch
import torch.nn as nn
import torch.nn.functional as F


def KL_loss(predictions, true_distributions):
    predictions = F.log_softmax(predictions, dim=1)
    KL = (true_distributions * predictions).sum()
    KL = -1.0 * KL / predictions.shape[0]
    return KL


def logit_ML_loss(view_predictions, labels, num_class):
    true_labels = F.one_hot(labels, num_classes=num_class)
    true_labels = true_labels.to(torch.float32)
    view_predictions_sig = torch.sigmoid(view_predictions)
    criterion = nn.BCELoss()
    ML_loss = criterion(view_predictions_sig, true_labels)
    return ML_loss
