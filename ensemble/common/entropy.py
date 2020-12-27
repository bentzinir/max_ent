from torch import nn
from torch.nn import functional as F


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, dim):
        b = F.softmax(x, dim=dim) * F.log_softmax(x, dim=dim)
        b = -1.0 * b.sum(dim)
        return b
