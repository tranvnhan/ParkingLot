import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
import numpy as np
import time
import numpy as np

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

        # remove unnecessary layers
        del self.classifier
        del self.avgpool
        del self.features[30]
        del self.features[29]
        del self.features[28]
        del self.features[27]
        del self.features[26]
        del self.features[25]
        del self.features[24]
        del self.features[23]
        del self.features[22]
        del self.features[21]
        del self.features[20]
        del self.features[19]
        del self.features[18]
        del self.features[17]
        
        self.end_range = 3 # only ((0, 5), (5, 10), (10, 17)) for vgg16

    def forward(self, x):
        # output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(self.end_range):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            # output["x%d" % (idx + 1)] = x

        # output = output['x5']
        output = x
        return output
        

class VGGNet_bn(VGG):
    def __init__(self, pretrained=True, model='vgg16_bn', requires_grad=True, show_params=False):
        super().__init__(make_layers(cfg[model], batch_norm=True))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

        # remove unnecessary layers
        del self.classifier
        del self.avgpool
        del self.features[43]
        del self.features[42]
        del self.features[41]
        del self.features[40]
        del self.features[39]
        del self.features[38]
        del self.features[37]
        del self.features[36]
        del self.features[35]
        del self.features[34]
        del self.features[33]
        del self.features[32]
        del self.features[31]
        del self.features[30]
        del self.features[29]
        del self.features[28]
        del self.features[27]
        del self.features[26]
        del self.features[25]
        del self.features[24]
        
        self.end_range = 3 # only ((0, 7), (7, 14), (14, 24)) for vgg16_bn

    def forward(self, x):
        # output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(self.end_range):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            # output["x%d" % (idx + 1)] = x

        # output = output['x5']
        output = x
        return output


ranges = {
    'vgg11':    ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13':    ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16':    ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg16_bn': ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)),
    'vgg19':    ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg16_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
