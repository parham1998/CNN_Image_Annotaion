# =============================================================================
# Import required libraries
# =============================================================================
import numpy as np
import torch
from torch.nn.modules.loss import _Loss


class MultiLabelLoss(_Loss):
    def __init__(self,
                 gamma_neg=0,
                 gamma_pos=0,
                 neg_margin=0,
                 eps=1e-8):
        super(MultiLabelLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.neg_margin = neg_margin
        self.eps = eps

    def forward(self, outputs, targets):
        p = torch.sigmoid(outputs)
        los_pos = targets * torch.log(p.clamp(min=self.eps)) * ((1 - p) ** self.gamma_pos)
        #
        if self.neg_margin is not None and self.neg_margin > 0:
            # probability margin
            p_m = (torch.sigmoid(outputs) - self.neg_margin).clamp(min=0)
            los_neg = (1 - targets) * torch.log((1 - p_m).clamp(min=self.eps)) * (p_m ** self.gamma_neg)
        else:
            los_neg = (1 - targets) * torch.log((1 - p).clamp(min=self.eps)) * (p ** self.gamma_neg)
        loss = los_pos + los_neg
        return -loss.mean()


# log sum exponential pairwise loss
class LSEPLoss(_Loss):
    def __init__(self):
        super(LSEPLoss, self).__init__()

    def forward(self, outputs, targets):
        loss = 0
        for batch_idx in range(targets.size(0)):
            t = targets[batch_idx]
            o = outputs[batch_idx]
            positive = [np.argwhere(t.detach().cpu().numpy() == 1)]
            negative = [np.argwhere(t.detach().cpu().numpy() == 0)]
            pos_exms = o[positive]
            neg_exms = o[negative].reshape(-1)
            loss += torch.log(1 +
                              torch.sum(torch.exp(neg_exms - pos_exms)))
        return loss / targets.size(0)
