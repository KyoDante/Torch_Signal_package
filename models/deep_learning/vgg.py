import torch
from torch import nn, optim
import torchvision
import time


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

def vgg_block(num_convs, in_channels, out_channels):
        blk = []
        for i in range(num_convs):
            if i ==0:
                blk.append(nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1))
            else:
                blk.append(nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1))
            blk.append(nn.ReLU())
        blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*blk)

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()

    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module('vgg_block_' + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    
    net.add_module('fc', nn.Sequential(
        FlattenLayer(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 10)
    ))
    return net

def make_vgg():
    conv_arch = ((1 ,1, 64), (1, 64, 128),
    (2,128,256),(2,256,512),(2,512,512))
    fc_features = 512 * 7 * 7
    fc_hidden_units = 4096
    net = vgg(conv_arch, fc_features, fc_hidden_units)
    
    return net