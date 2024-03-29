import torch
from torch import nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def calc_cosinsimilarity(self, x1, x2):
        return self.cos(x1, x2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_cosinsimilarity(anchor, positive)
        distance_negative = self.calc_cosinsimilarity(anchor, negative)

        losses = torch.relu(- torch.mean(distance_positive) +
                            torch.mean(distance_negative) + self.margin)

        return losses.mean()


class ContrastiveLoss(nn.Module):
    """
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def calc_cosinsimilarity(self, x1, x2):
        return self.cos(x1, x2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_cosinsimilarity(anchor, positive)
        distance_negative = self.calc_cosinsimilarity(anchor, negative)
        loss_contrastive = torch.mean(torch.pow(distance_negative, 2)) + torch.mean(torch.pow(torch.relu(self.margin - distance_positive), 2))
        return torch.mean(loss_contrastive)
