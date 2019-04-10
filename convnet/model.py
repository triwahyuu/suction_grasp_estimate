## network model
import torch
import torch.nn as nn
from torchvision import models
import copy

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
    # print(typename)
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

## for reference
# https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/models/resnet_fcn.py
def get_model():
    # load pre-trained resnet model
    rgb_trunk = models.resnet50(pretrained=True)
    # remove FC layer
    rgb_trunk = nn.Sequential(*(list(rgb_trunk.children())[:-3]))
    depth_trunk = copy.deepcopy(rgb_trunk)

    # https://discuss.pytorch.org/t/multiple-input-model-architecture/19754/2
    rgbd = torch.cat((rgb_trunk, depth_trunk), 0)

    model = nn.Sequential(rgbd)
    model.add_module(nn.Conv2d(2048, 512, kernel_size=1))
    model.add_module(nn.Conv2d(512, 128, kernel_size=1))
    model.add_module(nn.Conv2d(128, 3, kernel_size=1))
    model.add_module(nn.UpsamplingBilinear2d(scale_factor=2))

    # updatePadding(model, SymmetricPad2d)
    updatePadding(model, nn.ReflectionPad2d) # is it the same?


class SuctionModel(nn.Module):
    def __init__(self, options):
        super(SuctionModel, self).__init__()
        self.rgb_trunk = self.create_trunk()
        self.depth_trunk = self.create_trunk()

        self.feature = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.Conv2d(128, 3, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        updatePadding(self.feature, nn.ReflectionPad2d)

    def forward(self, rgbd_input):
        rgb_feature = self.rgb_trunk(rgbd_input[0])
        depth_feature = self.depth_trunk(rgbd_input[1])

        # concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 0)
        out = self.feature(rgbd_parallel)
        return out

    def create_trunk(self):
        resnet50 = models.resnet50(pretrained=True)
        m = nn.Sequential(*(list(resnet50.children())[:-3]))
        updatePadding(m, nn.ReflectionPad2d)
        return m
