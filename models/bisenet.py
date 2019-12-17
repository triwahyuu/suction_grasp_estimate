## PSPNet model definition
## Modified from:
## https://github.com/ooooverflow/BiSeNet/blob/master/model/
## https://arxiv.org/abs/1808.00897

import torch
from torch import nn
from torchvision import models
from .efficientnet import effbisenet

import warnings
warnings.filterwarnings(action='ignore')


class ResNet(nn.Module):
    def __init__(self, arch='resnet18',pretrained=True):
        super().__init__()

        _resnet = getattr(models, arch)
        features = _resnet(pretrained=pretrained)
        
        self.conv1 = features.conv1
        self.bn1 = features.bn1
        self.relu = features.relu
        self.maxpool1 = features.maxpool
        self.layer1 = features.layer1
        self.layer2 = features.layer2
        self.layer3 = features.layer3
        self.layer4 = features.layer4
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail

def _build_contextpath_resnet(model_name):
    avail_resnet = [('resnet' + str(n)) for n in [18, 34, 50, 101]]
    if model_name not in avail_resnet:
        raise ValueError('BiseNet backend should be one of: ' + ', '.join(avail_resnet))
    
    model = ResNet(arch=model_name)
    return model

def _build_contextpath_effnet(model_name):
    avail_effnet = [('efficientnet-b' + str(n)) for n in range(6)]
    if model_name not in avail_effnet:
        raise ValueError('BiseNet backend with EfficientNet should be one of: ' + \
            ', '.join(avail_effnet))
    
    model, in_size = effbisenet(model_name, pretrained=True)
    return model, in_size

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), \
            'in_channels ({}) and out_channels ({}) should all be {}'.format(self.in_channels, self.out_channels, x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        
        self.in_channels = in_channels 

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, input_1, input_2):
        input_1 = torch.nn.functional.interpolate(input_1, \
            size=(input_2.size(2), input_2.size(2)), mode='bilinear')
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class BiSeNet(nn.Module):
    def __init__(self, num_classes, att_size, bise_size, ffm_size):
        super(BiSeNet, self).__init__()

        ## attention module
        self.attention_refinement_module1 = AttentionRefinementModule(att_size, att_size)
        self.attention_refinement_module2 = AttentionRefinementModule(bise_size*2, bise_size*2)
        ## supervision block
        self.supervision1 = nn.Conv2d(in_channels=att_size, out_channels=num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(in_channels=bise_size*2, out_channels=num_classes, kernel_size=1)
        ## build feature fusion module
        self.feature_fusion_module = FeatureFusionModule(num_classes, ffm_size)

        ## build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, sx, cx1, cx2, tail):
        ## sx -> spatial_path output
        ## cx1, cx2, tail -> contex_path output
        ## output of context path
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        
        ## upsampling
        cx1 = torch.nn.functional.interpolate(cx1, scale_factor=2, mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, scale_factor=4, mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)

        if self.training == True:
            cx1_sup = torch.nn.functional.interpolate(self.supervision1(cx1), scale_factor=8, mode='bilinear')
            cx2_sup = torch.nn.functional.interpolate(self.supervision2(cx2), scale_factor=8, mode='bilinear')

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
        result = self.conv(result)

        if self.training:
            return result, cx1_sup, cx2_sup
        return result