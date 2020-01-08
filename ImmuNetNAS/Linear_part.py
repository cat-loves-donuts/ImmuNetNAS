import random
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

"define linear layer"
class Linear_block(nn.Module):
    def __init__(self, in_put, out_put):
        super(Linear_block, self).__init__()

        self.linear = torch.nn.Linear(in_put,out_put)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x