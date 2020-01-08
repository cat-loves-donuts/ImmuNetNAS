import torch.nn as nn
from mutation import mutate_conv

"generate thee layers by input codes"
class _conv_layer(nn.Sequential):
    def __init__(self, in_put, out_put, layer_type):
        super(_conv_layer, self).__init__()
        if layer_type ==1 or layer_type ==2 or layer_type ==3 or layer_type ==4:
            self.add_module("conv", mutate_conv(in_put, out_put, layer_type))

            self.add_module("BN1", nn.BatchNorm2d(out_put))
            self.add_module("ReLU1", nn.ReLU(inplace=True))
        else:
            self.add_module("conv", mutate_conv(in_put, out_put, layer_type))
            self.add_module("BN1", nn.BatchNorm2d(out_put))

"Generate defult pooling layers"
class _Pooling_layer(nn.Sequential):
    def __init__(self, in_put, out_put):
        super(_Pooling_layer, self).__init__()
        self.add_module("conv",nn.Conv2d(in_channels = in_put, out_channels = out_put, kernel_size= 1, stride= 1, padding = 0, bias = True))
        self.add_module("BN1", nn.BatchNorm2d(out_put))
        self.add_module("ReLU1", nn.ReLU(inplace=True))
        self.add_module("pooling", nn.AvgPool2d(kernel_size=3, stride=1, padding= 1))

"generate cell and define the foward function"
class CNN_final_block(nn.Module):
    def __init__(self, layer_list, stage_num, structure, in_put, out_put):   #stage_num : 3
        super(CNN_final_block, self).__init__()                                                           #structure : [0,1,1,0,1,0]
        self.stage_num = stage_num
        self.structure = structure
        self.in_put = in_put
        self.out_put = out_put
        self.layer_list = layer_list

        block = []
        num_picture = out_put
        num_to_pooling = num_picture * (stage_num+1)
        for i in range(stage_num):
            block.append(_conv_layer(self.out_put, self.out_put, self.layer_list[i]))
        block.append(_Pooling_layer(in_put=num_picture, out_put=num_picture))
        #block.append(nn.BatchNorm2d(num_picture))

        self.block = nn.ModuleList(block)

    "define forward function and skipping connection"
    def forward(self, x):
        sample_structure = []
        sample_structure.append([9])
        sample_structure.append([9])
        real_input = x
        out_layer = []
        out_layer.append(self.block[0](real_input))
        temp_structure = self.structure
        for i in range(0, self.stage_num):
            sample_structure.append(temp_structure[:i + 1])
            temp_structure = temp_structure[i + 1:]
        for i in range(2,len(sample_structure)):
            n = 0
            middle_input = 0
            for j in range(0, len(sample_structure[i])):
                n = n + sample_structure[i][j]
            if n == 0:
                middle_input = real_input
            if n > 0:
                for j in range(0, len(sample_structure[i])):
                    if sample_structure[i][j] == 1:
                        middle_input = middle_input + out_layer[j]
            out_layer.append(self.block[i-1](middle_input))
        #out = self.block[(self.stage_num+1)](out_layer[self.stage_num-1])
        out = out_layer[-1]
        return out


