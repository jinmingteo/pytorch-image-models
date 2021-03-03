import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class WeightedSoftTargetCrossEntropy(nn.Module):
    '''
    input: weights in tensor and cuda()
    '''
    def __init__(self, weight=None):
        super(WeightedSoftTargetCrossEntropy, self).__init__()
        self.weight = weight
        if self.weight is None:
            print ("Warning: Using WeightedSoftTargetCrossEntropy without weights!")

    def forward(self, x, target):
        loss = torch.sum(self.weight * -target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
