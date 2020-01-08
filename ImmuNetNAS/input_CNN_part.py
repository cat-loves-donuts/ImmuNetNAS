import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

"Define input layer"
class Input_CNN_block(nn.Module):
    def __init__(self, in_put, out_put):
        super(Input_CNN_block, self).__init__()
        self.Input_CNN = nn.Sequential(
            nn.Conv2d(in_channels=in_put, out_channels=out_put, kernel_size=1, stride=1, padding=0,
                             bias=True),
            nn.BatchNorm2d(out_put),
            nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.Input_CNN(x)
        return x