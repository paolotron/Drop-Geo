import torch
import logging
import torchvision
import torch.nn as nn
import torch.nn.functional as function
import numpy as np

FAST = False  # Change this to false if gpu memory is not enough


class NetVlad(nn.Module):
    def __init__(self, num_clusters=8, dim=256, alpha=100.0, normalize_input=True):
        super(NetVlad, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clusters, train_desc):
        clsts_assign = clusters / np.linalg.norm(clusters, axis=1, keepdims=True)
        dots = np.dot(clsts_assign, train_desc.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending
        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clusters))
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clsts_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        n, c = x.shape[:2]

        if self.normalize_input:
            x = function.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(n, self.num_clusters, -1)
        soft_assign = function.softmax(soft_assign, dim=1)

        x_flatten = x.view(n, c, -1)

        if FAST:
            # calculate residuals to each clusters
            residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) \
                - self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign.unsqueeze(2)
            vlad = residual.sum(dim=-1)
        else:
            vlad = torch.zeros([n, self.num_clusters, c], dtype=x.dtype, layout=x.layout, device=x.device)
            for c in range(self.num_clusters):  # slower than non-looped, but lower memory usage
                residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                           self.centroids[c:c + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
                residual *= soft_assign[:, c:c + 1, :].unsqueeze(2)
                vlad[:, c:c + 1, :] = residual.sum(dim=-1)

        vlad = function.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = function.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
