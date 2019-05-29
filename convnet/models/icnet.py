## ICNet model definition
## Modified from:
## https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/icnet.py
## https://arxiv.org/abs/1704.08545

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

## Loss: multi_scale_cross_entropy2d

icnet_specs = {
    "cityscapes": {"n_classes": 19, "input_size": (1025, 2049), "block_config": [3, 4, 6, 3]}
}

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride,
            padding, bias=True, dilation=1, is_batchnorm=True):

        super(conv2DBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
            padding=padding, stride=stride, bias=bias, dilation=dilation
        )

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, 
            stride, dilation=1, is_batchnorm=True):

        super(bottleNeckPSP, self).__init__()

        bias = not is_batchnorm

        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, stride=1, 
                padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, stride=stride, 
                padding=dilation, bias=bias, dilation=dilation, is_batchnorm=is_batchnorm
            )
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, stride=stride,
                padding=1, bias=bias, dilation=1, is_batchnorm=is_batchnorm
            )
        self.cb3 = conv2DBatchNorm(mid_channels, out_channels, 1, stride=1, 
                padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        self.cb4 = conv2DBatchNorm(in_channels, out_channels, 1, stride=stride,
            padding=0, bias=bias, is_batchnorm=is_batchnorm
        )

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)
        return F.relu(conv + residual, inplace=True)

class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation=1, is_batchnorm=True):
        super(bottleNeckIdentifyPSP, self).__init__()

        bias = not is_batchnorm

        self.cbr1 = conv2DBatchNormRelu(
            in_channels, mid_channels, 1, stride=1, 
            padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, stride=1,
                padding=dilation, bias=bias, dilation=dilation, is_batchnorm=is_batchnorm
            )
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, stride=1,
                padding=1, bias=bias, dilation=1, is_batchnorm=is_batchnorm
            )
        self.cb3 = conv2DBatchNorm(mid_channels, in_channels, 1, stride=1, 
                padding=0, bias=bias, is_batchnorm=is_batchnorm
        )

    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x + residual, inplace=True)


class residualBlockPSP(nn.Module):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels,
        stride, dilation=1, include_range="all", is_batchnorm=True
    ):
        super(residualBlockPSP, self).__init__()

        if dilation > 1:
            stride = 1

        # residualBlockPSP = convBlockPSP + identityBlockPSPs
        layers = []
        if include_range in ["all", "conv"]:
            layers.append(
                bottleNeckPSP(
                    in_channels,
                    mid_channels,
                    out_channels,
                    stride,
                    dilation,
                    is_batchnorm=is_batchnorm,
                )
            )
        if include_range in ["all", "identity"]:
            for i in range(n_blocks - 1):
                layers.append(
                    bottleNeckIdentifyPSP(
                        out_channels, mid_channels, stride, dilation, is_batchnorm=is_batchnorm
                    )
                )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class pyramidPooling(nn.Module):
    def __init__(
        self, in_channels, pool_sizes, model_name="pspnet", fusion_mode="cat", is_batchnorm=True
    ):
        super(pyramidPooling, self).__init__()

        bias = not is_batchnorm

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(
                conv2DBatchNormRelu(
                    in_channels,
                    int(in_channels / len(pool_sizes)),
                    1,
                    1,
                    0,
                    bias=bias,
                    is_batchnorm=is_batchnorm,
                )
            )

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    def forward(self, x):
        h, w = x.shape[2:]

        if self.training or self.model_name != "icnet":  # general settings or pspnet
            k_sizes = []
            strides = []
            for pool_size in self.pool_sizes:
                k_sizes.append((int(h / pool_size), int(w / pool_size)))
                strides.append((int(h / pool_size), int(w / pool_size)))
        else:  # eval mode and icnet: pre-trained for 1025 x 2049
            k_sizes = [(8, 15), (13, 25), (17, 33), (33, 65)]
            strides = [(5, 10), (10, 20), (16, 32), (33, 65)]

        if self.fusion_mode == "cat":  # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else:  # icnet: element-wise sum (including x)
            pp_sum = x

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
                pp_sum = pp_sum + out

            return pp_sum

class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class cascadeFeatureFusion(nn.Module):
    def __init__(self, n_classes, low_in_channels, 
            high_in_channels, out_channels, is_batchnorm=True):
        super(cascadeFeatureFusion, self).__init__()

        bias = not is_batchnorm

        self.low_dilated_conv_bn = conv2DBatchNorm(
            low_in_channels,
            out_channels,
            3,
            stride=1,
            padding=2,
            bias=bias,
            dilation=2,
            is_batchnorm=is_batchnorm,
        )
        self.low_classifier_conv = nn.Conv2d(
            int(low_in_channels),
            int(n_classes),
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
            dilation=1,
        )  # Train only
        self.high_proj_conv_bn = conv2DBatchNorm(
            high_in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

    def forward(self, x_low, x_high):
        x_low_upsampled = F.interpolate(
            x_low, size=get_interp_size(x_low, z_factor=2), mode="bilinear", align_corners=True
        )

        low_cls = self.low_classifier_conv(x_low_upsampled)

        low_fm = self.low_dilated_conv_bn(x_low_upsampled)
        high_fm = self.high_proj_conv_bn(x_high)
        high_fused_fm = F.relu(low_fm + high_fm, inplace=True)

        return high_fused_fm, low_cls

def get_interp_size(input, s_factor=1, z_factor=1):  # for caffe
    ori_h, ori_w = input.shape[2:]

    # shrink (s_factor >= 1)
    ori_h = (ori_h - 1) / s_factor + 1
    ori_w = (ori_w - 1) / s_factor + 1

    # zoom (z_factor >= 1)
    ori_h = ori_h + (ori_h - 1) * (z_factor - 1)
    ori_w = ori_w + (ori_w - 1) * (z_factor - 1)

    resize_shape = (int(ori_h), int(ori_w))
    return resize_shape

class ICNet(nn.Module):

    """
    Image Cascade Network
    URL: https://arxiv.org/abs/1704.08545
    References:
    1) Original Author's code: https://github.com/hszhao/ICNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/ICNet-tensorflow
    """

    def __init__(
        self,
        n_classes=19,
        block_config=[3, 4, 6, 3],
        input_size=(1025, 2049),
        version=None,
        is_batchnorm=True,
    ):

        super(ICNet, self).__init__()

        bias = not is_batchnorm

        self.block_config = (
            icnet_specs[version]["block_config"] if version is not None else block_config
        )
        self.n_classes = icnet_specs[version]["n_classes"] if version is not None else n_classes
        self.input_size = icnet_specs[version]["input_size"] if version is not None else input_size

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3,
            n_filters=32, padding=1, stride=2, bias=bias, is_batchnorm=is_batchnorm
        )
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=32, k_size=3,
            n_filters=32, padding=1, stride=1, bias=bias, is_batchnorm=is_batchnorm
        )
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=32, k_size=3, 
            n_filters=64, padding=1, stride=1, bias=bias, is_batchnorm=is_batchnorm
        )

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(
            self.block_config[0], 64, 32, 128, 1, 1, is_batchnorm=is_batchnorm
        )
        self.res_block3_conv = residualBlockPSP( self.block_config[1],
            128, 64, 256, 2, 1, include_range="conv", is_batchnorm=is_batchnorm
        )
        self.res_block3_identity = residualBlockPSP(self.block_config[1],
            128, 64, 256, 2, 1, include_range="identity", is_batchnorm=is_batchnorm
        )

        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(
            self.block_config[2], 256, 128, 512, 1, 2, is_batchnorm=is_batchnorm
        )
        self.res_block5 = residualBlockPSP(
            self.block_config[3], 512, 256, 1024, 1, 4, is_batchnorm=is_batchnorm
        )

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(
            1024, [6, 3, 2, 1], model_name="icnet", fusion_mode="sum", is_batchnorm=is_batchnorm
        )

        # Final conv layer with kernel 1 in sub4 branch
        self.conv5_4_k1 = conv2DBatchNormRelu(
            in_channels=1024,
            k_size=1,
            n_filters=256,
            padding=0,
            stride=1,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

        # High-resolution (sub1) branch
        self.convbnrelu1_sub1 = conv2DBatchNormRelu(
            in_channels=3,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu2_sub1 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu3_sub1 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=64,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.classification = nn.Conv2d(128, self.n_classes, 1, 1, 0)

        # Cascade Feature Fusion Units
        self.cff_sub24 = cascadeFeatureFusion(
            self.n_classes, 256, 256, 128, is_batchnorm=is_batchnorm
        )
        self.cff_sub12 = cascadeFeatureFusion(
            self.n_classes, 128, 64, 128, is_batchnorm=is_batchnorm
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # H, W -> H/2, W/2
        x_sub2 = F.interpolate(
            x, size=get_interp_size(x, s_factor=2), mode="bilinear", align_corners=True
        )

        # H/2, W/2 -> H/4, W/4
        x_sub2 = self.convbnrelu1_1(x_sub2)
        x_sub2 = self.convbnrelu1_2(x_sub2)
        x_sub2 = self.convbnrelu1_3(x_sub2)

        # H/4, W/4 -> H/8, W/8
        x_sub2 = F.max_pool2d(x_sub2, 3, 2, 1)

        # H/8, W/8 -> H/16, W/16
        x_sub2 = self.res_block2(x_sub2)
        x_sub2 = self.res_block3_conv(x_sub2)
        # H/16, W/16 -> H/32, W/32
        x_sub4 = F.interpolate(
            x_sub2, size=get_interp_size(x_sub2, s_factor=2), mode="bilinear", align_corners=True
        )
        x_sub4 = self.res_block3_identity(x_sub4)

        x_sub4 = self.res_block4(x_sub4)
        x_sub4 = self.res_block5(x_sub4)

        x_sub4 = self.pyramid_pooling(x_sub4)
        x_sub4 = self.conv5_4_k1(x_sub4)

        x_sub1 = self.convbnrelu1_sub1(x)
        x_sub1 = self.convbnrelu2_sub1(x_sub1)
        x_sub1 = self.convbnrelu3_sub1(x_sub1)

        x_sub24, sub4_cls = self.cff_sub24(x_sub4, x_sub2)
        x_sub12, sub24_cls = self.cff_sub12(x_sub24, x_sub1)

        x_sub12 = F.interpolate(
            x_sub12, size=get_interp_size(x_sub12, z_factor=2), mode="bilinear", align_corners=True
        )
        x_sub4 = self.res_block3_identity(x_sub4)
        sub124_cls = self.classification(x_sub12)

        if self.training:
            return (sub124_cls, sub24_cls, sub4_cls)
        else:
            sub124_cls = F.interpolate(
                sub124_cls,
                size=get_interp_size(sub124_cls, z_factor=4),
                mode="bilinear",
                align_corners=True,
            )
            return sub124_cls
