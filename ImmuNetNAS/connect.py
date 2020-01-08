import random
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

"connect two models"
class connection(nn.Sequential):
    def __init__(self, model1, model2):
        super(connection, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        out = self.model1(x)
        out = self.model2(out)
        return out

