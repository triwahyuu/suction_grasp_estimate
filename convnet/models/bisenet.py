## PSPNet model definition
## Modified from:
## https://github.com/ooooverflow/BiSeNet/blob/master/model/
## https://arxiv.org/abs/1808.00897

import torch
from torch import nn
from torchvision import models
import warnings
warnings.filterwarnings(action='ignore')


class ResNet(torch.nn.Module):
    def __init__(self, arch='resnet18',pretrained=True):
        super().__init__()
        features = None
        if arch == 'resnet18':
            features = models.resnet18(pretrained=pretrained)
        elif arch == 'resnet34':
            features = models.resnet34(pretrained=pretrained)
        elif arch == 'resnet50':
            features = models.resnet50(pretrained=pretrained)
        elif arch == 'resnet101':
            features = models.resnet101(pretrained=pretrained)
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

def build_contextpath(arch):
    model = None
    if arch in ['resnet101', 'resnet50', 'resnet18', 'resnet34']:
        model = ResNet(arch=arch)
    return model

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class SpatialPath(torch.nn.Module):
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

class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from context path) + 1024(from spatial path) + 2048(from spatial path)
        # resnet18  1024 = 256(from context path) + 256(from spatial path) + 512(from spatial path)
        self.in_channels = in_channels 

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path):
        super().__init__()
        # build spatial path
        # self.spatial_path = Spatial_path()

        # build attention refinement module  for resnet 101
        if context_path in ['resnet101', 'resnet50']:
            self.attention_refinement_module1 = AttentionRefinementModule(2048, 2048)
            self.attention_refinement_module2 = AttentionRefinementModule(4096, 4096)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 3328)

        elif context_path in ['resnet18', 'resnet34']:
            # build attention refinement module  for resnet 18
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)
        else:
            raise Exception('Error: context_path %s is unsupported' % (context_path))

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)


    def forward(self, sx, cx1, cx2, tail):
        # output of context path
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, scale_factor=2, mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, scale_factor=4, mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
        result = self.conv(result)

        return result