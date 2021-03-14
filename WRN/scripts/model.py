"""
Author: Zhou Chen
Date: 2020/4/17
Desc: WRN-50-2 PyTorch implementation
"""
import torch.nn as nn
import torch.nn.init as init


class Conv(nn.Module):
    """
    重载带relu的卷积层
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param activation: 是否带激活层
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation:
            self.f = nn.ReLU(inplace=True)
        else:
            self.f = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.f:
            x = self.f(x)
        return x


def wrn_conv1x1(in_channels, out_channels, stride, activate):
    return Conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        activation=activate)


def wrn_conv3x3(in_channels, out_channels, stride, activate):
    return Conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        activation=activate)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, widen_factor):
        super(Bottleneck, self).__init__()
        mid_channels = int(round(out_channels // 4 * widen_factor))
        self.conv1 = wrn_conv1x1(in_channels, mid_channels, stride=1, activate=True)
        self.conv2 = wrn_conv3x3(mid_channels, mid_channels, stride=stride, activate=True)
        self.conv3 = wrn_conv1x1(mid_channels, out_channels, stride=1, activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, widen_factor):
        super(Unit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        self.body = Bottleneck(in_channels, out_channels, stride, widen_factor)
        if self.resize_identity:
            self.identity_conv = wrn_conv1x1(in_channels, out_channels, stride, activate=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activation(x)
        return x


class InitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitBlock, self).__init__()
        self.conv = Conv(in_channels, out_channels, 7, 2, 3, True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class WRN(nn.Module):

    def __init__(self, channels, init_block_channels, widen_factor, in_channels=3, img_size=(224, 224), num_classes=101):
        super(WRN, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.features = nn.Sequential()
        self.features.add_module("init_block", InitBlock(in_channels=in_channels, out_channels=init_block_channels))
        in_channels = init_block_channels
        # 结构嵌套
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1),
                                 Unit(in_channels, out_channels, stride, widen_factor))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        # 平均池化层
        self.features.add_module('final_avg_pool', nn.AvgPool2d(kernel_size=7, stride=1))
        # 输出分类层
        self.output = nn.Linear(in_features=in_channels, out_features=num_classes)
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_wrn(blocks, widen_factor, **kwargs):
    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Error WRN block number: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    model = WRN(channels, init_block_channels, widen_factor, **kwargs)

    return model


def WRN50_2():
    return get_wrn(50, 2.0)


if __name__ == '__main__':
    model = WRN50_2()
    print(model)
