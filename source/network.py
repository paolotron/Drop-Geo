import torch
import logging
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from CBAM import CBAMBlock
from NetVlad import NetVlad
from GeM import GeM


class GeoLocalizationNet(nn.Module):
    """The model is composed of a backbone and an aggregation layer.
    The backbone is a (cropped) ResNet-18, and the aggregation is a L2
    normalization followed by max pooling.
    """

    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.encoder_dim = 256
        self.attention = None

        if args.attention:
            self.attention = CBAMBlock(channel=256)
            self.attention.init_weights()

        if args.netvlad_clusters is not None:
            self.aggregation = nn.Sequential(L2Norm(),
                                             NetVlad(dim=self.encoder_dim, num_clusters=args.netvlad_clusters))
        elif args.gem_power is not None:
            self.aggregation = nn.Sequential(L2Norm(),
                                             GeM(p=args.gem_power),
                                             Flatten())
        else:
            self.aggregation = nn.Sequential(L2Norm(),
                                             torch.nn.AdaptiveAvgPool2d(1),
                                             Flatten())

    def forward(self, x):
        x = self.backbone(x)
        if self.attention:
            x = self.attention(x)
        x = self.aggregation(x)

        return x


def get_backbone(args):
    backbone = torchvision.models.resnet18(pretrained=True)
    for name, child in backbone.named_children():
        if name == "layer3":
            break
        for params in child.parameters():
            params.requires_grad = False
    logging.debug("Train only conv4 of the ResNet-18 (remove conv5), freeze the previous ones")
    layers = list(backbone.children())[:-3]
    backbone = torch.nn.Sequential(*layers)
    if args.netvlad_clusters is not None:
        args.features_dim = args.netvlad_clusters * 256  # Number of features outputted by the vlad layer (clusters * conv4 chan)
    else:
        args.features_dim = 256  # Number of channels in conv4
    return backbone


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
