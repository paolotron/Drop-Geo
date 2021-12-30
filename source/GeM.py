import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as function


class GeM(nn.Module):
    def __init__(self, in_size=256, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.full((in_size,), p))
        self.in_size = in_size
        self.eps = eps

    def forward(self, x):
        x = torch.clamp(x, min=self.eps)
        x = torch.pow(x, self.p.unsqueeze(-1).unsqueeze(-1))
        x = function.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = torch.pow(x, 1./(self.p.unsqueeze(-1).unsqueeze(-1)))
        return x
