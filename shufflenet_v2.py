# Author Lingge Li from XJTU(446049454@qq.com)
# ShuffleNet_v2 network


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init


'''
def g_name(name, m):
    m.g_name = name
    return m

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    #Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
    def forward(self, x):
        n, c, h, w = x.size()
        x =  x.view(n, self.groups, c // self.groups, h, w).permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
        return x

def channel_shuffle(name, groups):
    return g_name(name, ChannelShuffle(groups))
'''
#Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
def channel_shuffle(x, groups):
    bacth_size, channels, height, width = x.size()
    x = x.view(bacth_size, groups, channels // groups, height, width)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(bacth_size, -1, height, width)
    return x

def conv_bn_relu(input_channels, output_channels, kernel_size, stride, padding, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias = False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True)
    )

class ShuffleNet_unit(nn.Module):
    def __init__(self, input_channels, output_channels, stride, benchmodel):
        super(ShuffleNet_unit, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride

        assert stride in [1, 2]

        half_channels = output_channels // 2

        if self.benchmodel == 'c':
            self.branch2 = nn.Sequential(
                #pointwise conv
                nn.Conv2d(half_channels, half_channels, 1, 1, 0, bias = False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True),
                #depthwise conv
                nn.Conv2d(half_channels, half_channels, 3, stride, 1, groups=half_channels, bias = False),
                nn.BatchNorm2d(half_channels),
                #pointwise conv
                nn.Conv2d(half_channels, half_channels, 1, 1, 0, bias = False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential(
                #depthwise conv
                nn.Conv2d(input_channels, input_channels, 3, stride, 1, groups=input_channels, bias = False),
                nn.BatchNorm2d(input_channels),
                #pointwise conv
                nn.Conv2d(input_channels, half_channels, 1, 1, 0, bias = False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True)
            )
            self.branch2 = nn.Sequential(
                #pointwise conv
                nn.Conv2d(input_channels, half_channels, 1, 1, 0, bias = False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True),
                #depthwise conv
                nn.Conv2d(half_channels, half_channels, 3, stride, 1, groups=half_channels, bias = False),
                nn.BatchNorm2d(half_channels),
                #pointwise conv
                nn.Conv2d(half_channels, half_channels, 1, 1, 0, bias = False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True)
            )
    
    def concat(self, x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)
    
    def forward(self, x):
        if self.benchmodel == 'c':
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self.concat(x1, self.branch2(x2))
        elif self.benchmodel == 'd':
            #print(self.branch1(x).size(), self.branch2(x).size())
            out = self.concat(self.branch1(x), self.branch2(x))

        return channel_shuffle(out, 2)

class ShuffleNet_v2(nn.Module):
    def __init__(self, num_classes = 1000, input_size = 224, width_mult = 1.0):
        super(ShuffleNet_v2, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        else: 
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        
        #build first layer
        input_channels = self.stage_out_channels[1]
        self.conv1 = conv_bn_relu(3, input_channels, kernel_size = 3, stride = 2, padding = 1)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.features = []

        #build ShuffleNet_unit
        for id_stage in range(len(self.stage_repeats)):
            num_repeat = self.stage_repeats[id_stage]
            output_channels = self.stage_out_channels[id_stage+2]
            for i in range(num_repeat):
                if i == 0:
                    self.features.append(ShuffleNet_unit(input_channels, output_channels, 2, 'd'))
                else:
                    self.features.append(ShuffleNet_unit(input_channels, output_channels, 1, 'c'))
                input_channels = output_channels
        
        self.features = nn.Sequential(*self.features)

        #build last several layers
        self.conv5 = conv_bn_relu(input_channels, self.stage_out_channels[-1], kernel_size = 1, stride = 1, padding = 0)
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))

        #build classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], num_classes, bias = True))

    def forward(self, x):
        #print('\tIn Model: input size', x.size())
        x = self.conv1(x)
        #print(x.size())
        x = self.maxpool(x)
        #print(x.size())
        x = self.features(x)
        #print(x.size())
        x = self.conv5(x)
        #print(x.size())
        x = self.globalpool(x)
        #print(x.size())
        x = x.view(-1, self.stage_out_channels[-1])
        #print(x.size())
        x = self.classifier(x)
        #print('output size: ', x.size())
        return x


'''
# Test the model
shufflenet_v2.eval()
correct = 0
total = 0

for images, labels in testLoader:
    images = Variable(images).cuda()
    labels = labels.cuda()
    outputs = shufflenet_v2(images)
    _, predicted = torch.max(outputs.data, 1)
    
    #labels.size(0): bacth_size
    total = total + labels.size(0)
    print(type(predicted), type(labels))
    correct = correct + (predicted == labels).sum()

    print('Accuracy of the network on the val images: %d %%' % (100 * correct / total))
'''







