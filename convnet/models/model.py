## Suction Model

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from .rfnet import rfnet101, rfnet50
from .rfnet_lw import rfnet_lw101, rfnet_lw50
from .pspnet import PSPNet

## source:
# https://github.com/foolwood/deepmask-pytorch/blob/master/models/DeepMask.py
class SymmetricPad2d(nn.Module):
    def __init__(self, padding):
        super(SymmetricPad2d, self).__init__()
        self.padding = padding
        try:
            self.pad_l, self.pad_b, self.pad_r, self.pad_t = padding
        except:
            self.pad_l, self.pad_b, self.pad_r, self.pad_t = [padding,]*4

    def forward(self, input):
        assert len(input.shape) == 4, "only Dimension=4 implemented"
        h = input.shape[2] + self.pad_t + self.pad_b
        w = input.shape[3] + self.pad_l + self.pad_r
        assert w >= 1 and h >= 1, "input is too small"
        output = torch.zeros(input.shape[0], input.shape[1], h, w).to(input.device)
        c_input = input
        if self.pad_t < 0:
            c_input = c_input.narrow(2, -self.pad_t, c_input.shape[2] + self.pad_t)
        if self.pad_b < 0:
            c_input = c_input.narrow(2, 0, c_input.shape[2] + self.pad_b)
        if self.pad_l < 0:
            c_input = c_input.narrow(3, -self.pad_l, c_input.shape[3] + self.pad_l)
        if self.pad_r < 0:
            c_input = c_input.narrow(3, 0, c_input.shape[3] + self.pad_r)

        c_output = output
        if self.pad_t > 0:
            c_output = c_output.narrow(2, self.pad_t, c_output.shape[2] - self.pad_t)
        if self.pad_b > 0:
            c_output = c_output.narrow(2, 0, c_output.shape[2] - self.pad_b)
        if self.pad_l > 0:
            c_output = c_output.narrow(3, self.pad_l, c_output.shape[3] - self.pad_l)
        if self.pad_r > 0:
            c_output = c_output.narrow(3, 0, c_output.shape[3] - self.pad_r)

        c_output.copy_(c_input)

        assert w >= 2*self.pad_l and w >= 2*self.pad_r and h >= 2*self.pad_t and h >= 2*self.pad_b
        "input is too small"
        for i in range(self.pad_t):
            output.narrow(2, self.pad_t-i-1, 1).copy_(output.narrow(2, self.pad_t+i, 1))
        for i in range(self.pad_b):
            output.narrow(2, output.shape[2] - self.pad_b + i, 1).copy_(
                output.narrow(2, output.shape[2] - self.pad_b - i-1, 1))
        for i in range(self.pad_l):
            output.narrow(3, self.pad_l-i-1, 1).copy_(output.narrow(3, self.pad_l+i, 1))
        for i in range(self.pad_r):
            output.narrow(3, output.shape[3] - self.pad_r + i, 1).copy_(
                output.narrow(3, output.shape[3] - self.pad_r - i-1, 1))
        return output

def updatePadding(net, nn_padding):
    typename = torch.typename(net)

    if typename.find('Sequential') >= 0 or typename.find('Bottleneck') >= 0:
        modules_keys = list(net._modules.keys())
        for i in reversed(range(len(modules_keys))):
            subnet = net._modules[modules_keys[i]]
            out = updatePadding(subnet, nn_padding)
            if out != -1:
                p = out
                in_c, out_c, k, s, _, d, g, b = \
                    subnet.in_channels, subnet.out_channels, \
                    subnet.kernel_size[0], subnet.stride[0], \
                    subnet.padding[0], subnet.dilation[0], \
                    subnet.groups, subnet.bias,
                conv_temple = nn.Conv2d(in_c, out_c, k, stride=s, padding=0,
                                        dilation=d, groups=g, bias=b)
                conv_temple.weight = subnet.weight
                conv_temple.bias = subnet.bias
                if p > 1:
                    net._modules[modules_keys[i]] = nn.Sequential(SymmetricPad2d(p), conv_temple)
                else:
                    net._modules[modules_keys[i]] = nn.Sequential(nn_padding(p), conv_temple)
    else:
        if typename.find('torch.nn.modules.conv.Conv2d') >= 0:
            k_sz, p_sz = net.kernel_size[0], net.padding[0]
            if ((k_sz == 3) or (k_sz == 7)) and p_sz != 0:
                return p_sz
    return -1


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
        return x

## Suction Model with RefineNet Backbone
class SuctionRefineNet(nn.Module):
    def __init__(self, options):
        super(SuctionRefineNet, self).__init__()
        self.arch = options.arch
        self.n_class = options.n_class

        self.rgb_trunk = self._create_trunk()
        self.depth_trunk = self._create_trunk()

        self.feature = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(0.25),
            nn.Conv2d(128, options.n_class, kernel_size=3, stride=1, padding=1, bias=True)
        )
        updatePadding(self.feature, nn.ReflectionPad2d)

    def forward(self, rgbd_input):
        rgb_feature = self.rgb_trunk(rgbd_input[0])
        depth_feature = self.depth_trunk(rgbd_input[1])

        # concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 1)

        out = self.feature(rgbd_parallel)
        return out
    
    def _create_trunk(self):
        rfnet = rfnet101(self.n_class, pretrained=True)
        if self.arch == 'rfnet50':
            rfnet = rfnet50(self.n_class, pretrained=True)
            
        updatePadding(rfnet, nn.ReflectionPad2d)
        m = rfnet
        return m


## Suction Model with Light-Weight RefineNet Backbone
class SuctionRefineNetLW(nn.Module):
    def __init__(self, options):
        super(SuctionRefineNetLW, self).__init__()
        self.arch = options.arch
        self.n_class = options.n_class

        self.rgb_trunk = self._create_trunk()
        self.depth_trunk = self._create_trunk()

        self.feature = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(0.25),
            nn.Conv2d(128, options.n_class, kernel_size=3, stride=1, padding=1, bias=True)
        )
        updatePadding(self.feature, nn.ReflectionPad2d)

    def forward(self, rgbd_input):
        rgb_feature = self.rgb_trunk(rgbd_input[0])
        depth_feature = self.depth_trunk(rgbd_input[1])

        ## concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 1)

        out = self.feature(rgbd_parallel)
        return out
    
    def _create_trunk(self):
        rfnet = rfnet_lw101(self.n_class, pretrained=True)
        if self.arch == 'rfnet50':
            rfnet = rfnet_lw50(self.n_class, pretrained=True)
        
        updatePadding(rfnet, nn.ReflectionPad2d)
        m = rfnet
        return m

## Suction Model with ResNet18 or ResNet34 backbone
class SuctionModel18(nn.Module):
    def __init__(self, options):
        super(SuctionModel18, self).__init__()
        self.arch = options.arch
        self.rgb_trunk = self._create_trunk()
        self.depth_trunk = self._create_trunk()

        self.feature = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv2d(512, 128, kernel_size=(1,1), stride=(1,1)),
            nn.Threshold(0, 1e-6),
            nn.Dropout(0.25),
            nn.Conv2d(128, 3, kernel_size=(1,1), stride=(1,1)),
            Interpolate(scale=2, mode='bilinear')
        )
        updatePadding(self.feature, nn.ReflectionPad2d)

    def forward(self, rgbd_input):
        rgb_feature = self.rgb_trunk(rgbd_input[0])
        depth_feature = self.depth_trunk(rgbd_input[1])

        # concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 1)

        out = self.feature(rgbd_parallel)
        return out

    def _create_trunk(self):
        resnet = resnet18(pretrained=True)
        if self.arch == 'resnet34':
            resnet = resnet34(pretrained=True)
        ## removing classifier layers
        m = nn.Sequential(*(list(resnet.children())[:-3]))
        updatePadding(m, nn.ReflectionPad2d)
        # m = resnet
        return m


## Suction Model with ResNet50, ResNet101, or ResNet152 backbone
class SuctionModel50(nn.Module):
    def __init__(self, options):
        super(SuctionModel50, self).__init__()
        self.arch = options.arch
        self.rgb_trunk = self._create_trunk()
        self.depth_trunk = self._create_trunk()

        self.feature = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv2d(2048, 512, kernel_size=(1,1), stride=(1,1)),
            nn.Dropout(0.25),
            nn.Conv2d(512, 128, kernel_size=(1,1), stride=(1,1)),
            nn.Threshold(0, 1e-6),
            nn.Dropout(0.25),
            nn.Conv2d(128, 3, kernel_size=(1,1), stride=(1,1)),
            Interpolate(scale=2, mode='bilinear')
        )
        updatePadding(self.feature, nn.ReflectionPad2d)

    def forward(self, rgbd_input):
        rgb_feature = self.rgb_trunk(rgbd_input[0])
        depth_feature = self.depth_trunk(rgbd_input[1])

        # concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 1)

        out = self.feature(rgbd_parallel)
        return out

    def _create_trunk(self):
        resnet = resnet101(pretrained=True)
        if self.arch == 'resnet50':
            resnet = resnet50(pretrained=True)
        elif self.arch == 'resnet152':
            resnet = resnet152(pretrained=True)
        ## removing classifier layers
        m = nn.Sequential(*(list(resnet.children())[:-3]))
        updatePadding(m, nn.ReflectionPad2d)
        # m = resnet
        return m

## Suction Model with ResNet backbone and PSPNet Segmentor
class SuctionPSPNet(nn.Module):
    def __init__(self, options):
        super(SuctionPSPNet, self).__init__()
        self.arch = options.arch
        self.psp_size = 2048
        self.rgb_trunk = self._create_trunk()
        self.depth_trunk = self._create_trunk()

        self.segment = PSPNet(n_classes=options.n_class, psp_size=self.psp_size)
        updatePadding(self.segment, nn.ReflectionPad2d)

    def forward(self, rgbd_input):
        rgb_feature = self.rgb_trunk(rgbd_input[0])
        depth_feature = self.depth_trunk(rgbd_input[1])

        # concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 1)
        
        out = self.segment(rgbd_parallel)
        return out

    def _create_trunk(self):
        resnet = resnet101(pretrained=True)
        if self.arch == 'pspnet18':
            self.psp_size = 512
            resnet = resnet18(pretrained=True)
        elif self.arch == 'pspnet34':
            self.psp_size = 512
            resnet = resnet34(pretrained=True)
        elif self.arch == 'pspnet50':
            resnet = resnet50(pretrained=True)
        elif self.arch == 'pspnet152':
            resnet = resnet152(pretrained=True)
        ## removing classifier layers
        m = nn.Sequential(*(list(resnet.children())[:-3]))
        updatePadding(m, nn.ReflectionPad2d)
        # m = resnet
        return m