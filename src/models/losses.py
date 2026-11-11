import torch.nn as nn
import torch

class NegativePearsonCorrelation(nn.Module):

    def __init__(self):

        super().__init__()

        self.eps = 1e-6

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred, target: (B, N)
        """

        norm_pred = pred - pred.mean(dim=1, keepdim=True)
        norm_target = target - target.mean(dim=1, keepdim=True)

        num = torch.sum((norm_pred * norm_target), dim=1)
        den = torch.sqrt(torch.sum((norm_pred ** 2), dim=1) + self.eps) * torch.sqrt(torch.sum((norm_target ** 2), dim=1) + self.eps)

        loss = 1 - (num / den).mean()

        return loss
