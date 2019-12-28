## Suction Model main definition

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

import math
import numbers

from .rfnet import rfnet101, rfnet50
from .rfnet_lw import rfnet_lw101, rfnet_lw50
from .pspnet import PSPNet
from .bisenet import BiSeNet, SpatialPath
from .bisenet import _build_contextpath_resnet, _build_contextpath_effnet
from .efficientnet import efficientnet



## https://github.com/foolwood/deepmask-pytorch/blob/master/models/DeepMask.py
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


## https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()

        self.k = kernel_size
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self._parameters.requires_grad = False

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def _load_resnet_trunk(model_name):
    avail_resnet = [('resnet' + str(n)) for n in [18, 34, 50, 101, 152]]
    if model_name not in avail_resnet:
        raise ValueError('ResNet backend should be one of: ' + ', '.join(avail_resnet))
    
    _resnet = getattr(torchvision.models, model_name)
    model = _resnet(pretrained=True)

    ## removing classifier layers
    model = nn.Sequential(*(list(model.children())[:-3]))
    updatePadding(model, nn.ReflectionPad2d)
    return model

def _load_effnet_trunk(model_name):
    avail_effnet = [('efficientnet-b' + str(n)) for n in range(6)]
    if model_name not in avail_effnet:
        raise ValueError('EfficientNet backend should be one of: ' + ', '.join(avail_effnet))

    model,_ = efficientnet(model_name, pretrained=True)

    ## removing classifier layers
    updatePadding(model, nn.ReflectionPad2d)
    return model


## original model using FCN with ResNet backbone
class SuctionModelFCN(nn.Module):
    def __init__(self, arch='resnet18', n_class=3, out_size=(480, 640)):
        super(SuctionModelFCN, self).__init__()
        self.arch = arch
        self.n_class = n_class
        self.out_size = out_size

        self.rgb_trunk = _load_resnet_trunk(arch)
        self.depth_trunk = _load_resnet_trunk(arch)

        if arch in ['resnet18', 'resnet34']:
            self.feature = nn.Sequential(
                nn.Dropout(0.25),
                nn.Conv2d(512, 128, kernel_size=(1,1), stride=(1,1)),
                # nn.Threshold(0, 1e-6),
                nn.Dropout(0.4),
                nn.Conv2d(128, n_class, kernel_size=(1,1), stride=(1,1))
            )
        elif arch in ['resnet50', 'resnet101', 'resnet152']:
            self.feature = nn.Sequential(
                nn.Dropout(0.25),
                nn.Conv2d(2048, 512, kernel_size=(1,1), stride=(1,1)),
                nn.Dropout(0.25),
                nn.Conv2d(512, 128, kernel_size=(1,1), stride=(1,1)),
                nn.Dropout(0.4),
                nn.Conv2d(128, n_class, kernel_size=(1,1), stride=(1,1))
            )
        updatePadding(self.feature, nn.ReflectionPad2d)
        self.blur = GaussianSmoothing(channels=3, kernel_size=5, sigma=7)

    def forward(self, rgb_input, ddd_input):
        rgb_feature = self.rgb_trunk(rgb_input)
        depth_feature = self.depth_trunk(ddd_input)

        # concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 1)

        out = self.feature(rgbd_parallel)
        out = F.interpolate(out, size=self.out_size, mode='bilinear')
        return self.blur(F.pad(out, (self.blur.k//2, )*4, mode='reflect'))
        # return out


## Suction Model with ResNet backbone and PSPNet as feature map
class SuctionPSPNet(nn.Module):
    def __init__(self, arch='pspnet18', n_class=3, out_size=(480, 640)):
        super(SuctionPSPNet, self).__init__()
        self.arch = arch
        self.n_class = n_class
        self.out_size = out_size

        psp_size = 2048
        if arch in ['pspnet18', 'pspnet34']:
            psp_size = 512
        self.backbone = arch.replace('pspnet', 'resnet')

        self.rgb_trunk = _load_resnet_trunk(self.backbone)
        self.depth_trunk = _load_resnet_trunk(self.backbone)

        self.segment = PSPNet(n_classes=n_class, psp_size=psp_size)
        updatePadding(self.segment, nn.ReflectionPad2d)

    def forward(self, rgb_input, ddd_input):
        rgb_feature = self.rgb_trunk(rgb_input)
        depth_feature = self.depth_trunk(ddd_input)

        # concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 1)
        
        out = self.segment(rgbd_parallel)
        return F.interpolate(out, size=self.out_size, mode='bilinear')


## Suction Model with RefineNet as feature map
class SuctionRefineNet(nn.Module):
    def __init__(self, arch='rfnet18', n_class=3, out_size=(480, 640)):
        super(SuctionRefineNet, self).__init__()
        self.arch = arch
        self.n_class = n_class
        self.out_size = out_size

        self.rgb_trunk = self._create_trunk()
        self.depth_trunk = self._create_trunk()

        self.feature = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(0.25),
            nn.Conv2d(128, n_class, kernel_size=3, stride=1, padding=1, bias=True)
        )
        updatePadding(self.feature, nn.ReflectionPad2d)

    def forward(self, rgb_input, ddd_input):
        rgb_feature = self.rgb_trunk(rgb_input)
        depth_feature = self.depth_trunk(ddd_input)

        # concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 1)

        out = self.feature(rgbd_parallel)
        return F.interpolate(out, size=self.out_size, mode='bilinear')
    
    def _create_trunk(self):
        rfnet = rfnet101(pretrained=True)
        if self.arch == 'rfnet50':
            rfnet = rfnet50(pretrained=True)
            
        updatePadding(rfnet, nn.ReflectionPad2d)
        return rfnet


## Suction Model with Light-Weight RefineNet Backbone
class SuctionRefineNetLW(nn.Module):
    def __init__(self, arch='rfnet18', n_class=3, out_size=(480, 640)):
        super(SuctionRefineNetLW, self).__init__()
        self.arch = arch
        self.n_class = n_class
        self.out_size = out_size

        self.rgb_trunk = self._create_trunk()
        self.depth_trunk = self._create_trunk()

        self.feature = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(0.25),
            nn.Conv2d(128, n_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Upsample(size=out_size, mode='bilinear')
        )
        updatePadding(self.feature, nn.ReflectionPad2d)

    def forward(self, rgb_input, ddd_input):
        rgb_feature = self.rgb_trunk(rgb_input)
        depth_feature = self.depth_trunk(ddd_input)

        ## concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 1)

        out = self.feature(rgbd_parallel)
        return F.interpolate(out, size=self.out_size, mode='bilinear')
    
    def _create_trunk(self):
        rfnet = rfnet_lw101(pretrained=True)
        if self.arch == 'rfnet50':
            rfnet = rfnet_lw50(pretrained=True)
        
        updatePadding(rfnet, nn.ReflectionPad2d)
        return rfnet


class SuctionBiSeNet(nn.Module):
    def __init__(self, arch='bisenet18', n_class=3, out_size=(480, 640)):
        super(SuctionBiSeNet, self).__init__()
        self.arch = arch
        self.n_class = n_class
        self.out_size = out_size
        self.backbone = arch.replace('bisenet', 'resnet')

        self.rgb_trunk = self._create_trunk()
        self.depth_trunk = self._create_trunk()

        self.rgb_spatial = SpatialPath()
        self.depth_spatial = SpatialPath()
        
        expansion = 1 if self.backbone in ['resnet18', 'resnet34'] else 4
        self.feature = BiSeNet('resnet', n_class, \
            512*expansion, 512*expansion, 512*3*expansion+512)
        self.upsample = nn.Upsample(size=out_size, mode='bilinear')
        updatePadding(self.feature, nn.ReflectionPad2d)
    
    def forward(self, rgb_input, ddd_input):
        ## context path
        rgb_cx1, rgb_cx2, rgb_tail = self.rgb_trunk(rgb_input)
        depth_cx1, depth_cx2, depth_tail = self.depth_trunk(ddd_input)

        ## spatial path
        rgb_sx = self.rgb_spatial(rgb_input)
        depth_sx = self.depth_spatial(ddd_input)

        ## concatenate rgb and depth
        rgbd_cx1 = torch.cat((rgb_cx1, depth_cx1), 1)
        rgbd_cx2 = torch.cat((rgb_cx2, depth_cx2), 1)
        rgbd_tail = torch.cat((rgb_tail, depth_tail), 1)
        rgbd_sx = torch.cat((rgb_sx, depth_sx), 1)
        
        out = self.feature(rgbd_sx, rgbd_cx1, rgbd_cx2, rgbd_tail)
        
        if self.training:
            return self.upsample(out[0]), self.upsample(out[1]), self.upsample(out[2])
        
        return self.upsample(out)
    
    def _create_trunk(self):
        m = _build_contextpath_resnet(self.backbone)
        updatePadding(m, nn.ReflectionPad2d)
        return m


## [TODO]: build ICNet
## F*** still don't understand it
class SuctionICNet(nn.Module):
    def __init__(self, arch='icnet18', n_class=3, out_size=(480, 640)):
        self.arch = arch
        self.n_class = n_class
        self.out_size = out_size
    
    def forward(self, rgb_input, ddd_input):
        pass
    
    def _create_trunk(self):
        pass


class SuctionEffNetFCN(nn.Module):
    def __init__(self, arch='fcneffnetb0', n_class=3, out_size=(480, 640)):
        super(SuctionEffNetFCN, self).__init__()
        self.arch = arch
        self.n_class = n_class
        self.out_size = out_size
        self.backbone = arch.replace('fcneffnet', 'efficientnet-')
        
        self.rgb_trunk = _load_effnet_trunk(self.backbone)
        self.depth_trunk = _load_effnet_trunk(self.backbone)

        ## efficientnet backend output channel size map (0-5)
        out_sz = [1280, 1280, 1408, 1536, 1792, 2048]
        effnet_idx = int(arch[-1])

        self.feature = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv2d(out_sz[effnet_idx]*2, out_sz[effnet_idx]//2, kernel_size=(1,1), stride=(1,1)),
            nn.Dropout(0.25),
            nn.Conv2d(out_sz[effnet_idx]//2, out_sz[effnet_idx]//4, kernel_size=(1,1), stride=(1,1)),
            nn.Dropout(0.4),
            nn.Conv2d(out_sz[effnet_idx]//4, self.n_class, kernel_size=(1,1), stride=(1,1))
        )
        updatePadding(self.feature, nn.ReflectionPad2d)
    
    def forward(self, rgb_input, ddd_input):
        rgb_feature = self.rgb_trunk(rgb_input)
        depth_feature = self.depth_trunk(ddd_input)

        # concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 1)
        
        out = self.feature(rgbd_parallel)
        return F.interpolate(out, size=self.out_size, mode='bilinear')


class SuctionEffNetPSP(nn.Module):
    def __init__(self, arch='pspeffnetb0', n_class=3, out_size=(480, 640)):
        super(SuctionEffNetPSP, self).__init__()
        self.arch = arch
        self.n_class = n_class
        self.out_size = out_size
        self.backbone = arch.replace('pspeffnet', 'efficientnet-')

        ## efficientnet backend output channel size map (0-5)
        out_sz = [1280, 1280, 1408, 1536, 1792, 2048]
        effnet_idx = int(arch[-1])

        self.rgb_trunk = _load_effnet_trunk(self.backbone)
        self.depth_trunk = _load_effnet_trunk(self.backbone)

        self.feature = PSPNet(n_classes=n_class, psp_size=out_sz[effnet_idx]*2)
        updatePadding(self.feature, nn.ReflectionPad2d)
    
    def forward(self, rgb_input, ddd_input):
        rgb_feature = self.rgb_trunk(rgb_input)
        depth_feature = self.depth_trunk(ddd_input)

        # concatenate rgb and depth input
        rgbd_parallel = torch.cat((rgb_feature, depth_feature), 1)
        
        out = self.feature(rgbd_parallel)
        return F.interpolate(out, size=self.out_size, mode='bilinear')

class SuctionEfficientBiSeNet(nn.Module):
    def __init__(self, arch='biseeffnetb0', n_class=3, out_size=(480, 640)):
        super(SuctionEfficientBiSeNet, self).__init__()
        self.arch = arch
        self.n_class = n_class
        self.out_size = out_size
        self.backbone = arch.replace('biseeffnet', 'efficientnet-')

        self.rgb_trunk, self.in_size = self._create_trunk()
        self.depth_trunk, self.in_size = self._create_trunk()

        self.rgb_spatial = SpatialPath()
        self.depth_spatial = SpatialPath()

        att1 = [384, 384, 416, 464, 544, 608]
        att2 = [1280, 1280, 1408, 1536, 1792, 2048]
        ffm = [3456, 3456, 3744, 4048, 4640, 5216]
        effnet_idx = int(arch[-1])
        
        self.feature = BiSeNet('efficientnet', n_class, \
            att1[effnet_idx], att2[effnet_idx], ffm[effnet_idx])
        self.upsample = nn.Upsample(size=out_size, mode='bilinear')
        updatePadding(self.feature, nn.ReflectionPad2d)
    
    def forward(self, rgb_input, ddd_input):
        ## context path
        rgb_cx1, rgb_cx2, rgb_tail = self.rgb_trunk(rgb_input)
        depth_cx1, depth_cx2, depth_tail = self.depth_trunk(ddd_input)

        ## spatial path
        rgb_input = F.interpolate(rgb_input, size=self.in_size, mode='bilinear')
        ddd_input = F.interpolate(ddd_input, size=self.in_size, mode='bilinear')
        rgb_sx = self.rgb_spatial(rgb_input)
        depth_sx = self.depth_spatial(ddd_input)

        ## concatenate rgb and depth
        rgbd_cx1 = torch.cat((rgb_cx1, depth_cx1), 1)
        rgbd_cx2 = torch.cat((rgb_cx2, depth_cx2), 1)
        rgbd_tail = torch.cat((rgb_tail, depth_tail), 1)
        rgbd_sx = torch.cat((rgb_sx, depth_sx), 1)
        
        rgbd_cx1 = torch.nn.functional.interpolate(rgbd_cx1, scale_factor=2, mode='bilinear')
        out = self.feature(rgbd_sx, rgbd_cx1, rgbd_cx2, rgbd_tail)
        
        if self.training:
            return self.upsample(out[0]), self.upsample(out[1]), self.upsample(out[2])
        
        return self.upsample(out)
    
    def _create_trunk(self):
        m, in_size = _build_contextpath_effnet(self.backbone)
        updatePadding(m, nn.ReflectionPad2d)
        return m, in_size