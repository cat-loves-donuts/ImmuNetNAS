import random
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

def mutate_conv(in_put, out_put, layer_type):
    if layer_type == 1:
        mutate = nn.Conv2d(in_channels=in_put, out_channels=out_put, kernel_size=1, stride=1, padding=0,
                             bias=True)
    if layer_type == 2:
        mutate = nn.Conv2d(in_channels=in_put, out_channels=out_put, kernel_size=3, stride=1, padding=1,
                             bias=True)
    if layer_type == 3:
        mutate = nn.Conv2d(in_channels=in_put, out_channels=out_put, kernel_size=5, stride=1, padding=2,
                             bias=True)
    if layer_type == 4:
        mutate = nn.Conv2d(in_channels=in_put, out_channels=out_put, kernel_size=7, stride=1, padding=3,
                             bias=True)
    if layer_type == 5:
        mutate = nn.AvgPool2d(kernel_size=3, stride=1, padding= 1)
    if layer_type == 6:
        mutate = nn.AvgPool2d(kernel_size=5, stride=1, padding= 2)
    if layer_type == 7:
        mutate = nn.MaxPool2d(kernel_size=3, stride=1, padding= 1)
    if layer_type == 8:
        mutate = nn.MaxPool2d(kernel_size=5, stride=1, padding= 2)

    return mutate


def structure_mutate(structure, affinity, amount, population, index):

    k1 = 0.1
    k2 = 0.2
    average_affinity = (float(affinity[-1][1])+float(affinity[0][1]))/2
    if affinity[index][1] >= average_affinity:
        mutate_rate = k1*(affinity[0][1]-affinity[index][1])/(affinity[0][1]-average_affinity)
    else:
        mutate_rate = k2
    mutate_number = mutate_rate * population

    mutate_final_layer = mutate_number*0.2
    mutate_rest_layer = mutate_number*0.5
    mutate_layer_type = mutate_number*0.3

    if mutate_layer_type<1:
        mutate_layer_type = 1
    if mutate_rest_layer<1:
        mutate_rest_layer = 1
    if mutate_final_layer<1:
        mutate_final_layer = 1

    if index <= (amount * 0.2):
        a = [i + 1 for i in range(0, len(structure[index][0]))]
        for l in range(0,int(mutate_final_layer)):
            num = random.choice(a)
            if structure[index][1][-num] == 1:
                structure[index][1][-num] = 0
            elif structure[index][1][-num] == 0:
                structure[index][1][-num] = 1
            a.remove(num)

    elif (amount * 0.2) < index <= (amount * 0.3):
        a = [i + 1 for i in range(0, len(structure[index][0]))]
        b = [i for i in range(0,len(structure[index][1][:-5]))]
        for m in range(0,int(mutate_final_layer)):
            num = random.choice(a)
            if structure[index][1][-num] == 1:
                structure[index][1][-num] = 0
            elif structure[index][1][-num] == 0:
                structure[index][1][-num] = 1
            a.remove(num)
        for n in range(0,int(mutate_rest_layer)):
            num = random.choice(b)
            if structure[index][1][num] == 1:
                structure[index][1][num] = 0
            elif structure[index][1][-num] == 0:
                structure[index][1][num] = 1
            b.remove(num)

    else:
        q = 0
        a = [i + 1 for i in range(0, len(structure[index][0]))]
        b = [i for i in range(0, len(structure[index][1][:-5]))]
        for o in range(0,int(mutate_final_layer)):
            num = random.choice(a)
            if structure[index][1][-num] == 1:
                structure[index][1][-num] = 0
            elif structure[index][1][-num] == 0:
                structure[index][1][-num] = 1
            a.remove(num)
        for p in range(0,int(mutate_rest_layer)):
            num = random.choice(b)
            if structure[index][1][num] == 1:
                structure[index][1][num] = 0
            elif structure[index][1][-num] == 0:
                structure[index][1][num] = 1
            b.remove(num)
        for q in range(0,int(mutate_layer_type)):
            num = random.choice(a)
            changed_type = random.choice([1,2,3,4,5,6,7])
            structure[index][0][num-1] = changed_type
            a.remove(num)
    return structure[index]


