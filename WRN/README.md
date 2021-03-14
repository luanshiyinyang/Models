# WRN

## 简介

卷积神经网络的发展经历了 AlexNet、VGGNet、Inception 到 ResNet 的发展过程，在如今的计算机视觉中层和上层任务中使用 ResNet 是一个很常见的选择，一方面其效果较好（无论是速度还是精度），另一方面其泛化能力很强（适合于以迁移学习为基础的计算机视觉任务）。ResNet 为更深的网络设计带来了充分的先决条件，但是，随着模型的加深，参数量迅速增加而带来的精度收益却比较小，Sergey Zagoruyko 等人认为过深的残差网络并不能带来较大的性能变化，可以改用宽而浅的网络来获得更好的效果，于是提出了 Wide Residual Networks（WRN），效果显著。

- 论文标题
	
	Wide residual networks
- 论文地址

	https://arxiv.org/abs/1605.07146
- 论文源码

	https://github.com/szagoruyko/wide-residual-networks

## 网络设计

### 先前问题

网络的深浅问题是一个讨论已久的话题，很多研究表明，在复杂性相同的情况下，浅层网络比深层网络多指数倍的部件。因此何恺明等人提出了残差网络，他们希望增加网络层数来使得网络很深并且有更少的参数。本文作者认为，包含 identity mapping 的残差模块允许训练很深的网络的同时，也带来了一些问题。梯度流经网络的时候，网络不会强制要求梯度流过残差块的权重层，这可能导致训练中几乎学不到什么。最后，只有少量的残差块能够学到有用的表达（绝大多数残差块用处不大）或者很多块分享非常少的信息，这对结果影响很小。

上述问题称为 diminishing feature reuse（特征复用缺失），本论文作者希望使用一种浅而宽的模型，来有效提升模型性能。在 ResNetv2 的基础上，以正确的方式加宽残差块提供了一种更加高效地提升残差网络性能的方法。提出的 Wide Residual Networks（WRN），在深度较浅，参数同等的情况下，获得不弱于深层网络的效果，且训练更快。并且，论文提出了一种新的 Dropout 使用方法，前人将 dropout 放在 identity 连接上，性能下降，作者认为，应该将 dropout 插入卷积层之间，并借此获得了新的 sota 效果。

### 设计贡献

1. 提出了对比实验，对残差网络进行了详细的实验研究；
2. 提出了 WRN，相对于 ResNet，性能有较大的提高；
3. 提出了深度残差网络中 dropout 使用的新思路，有效进行了正则化。

## WRN

残差网络中有两种类型的模块，分别为basic（包含两个3x3卷积，每个卷积紧跟BN层和Relu激活，如下图a）和bottleneck（包含被两个1x1卷积包裹的3x3卷积，1x1卷积用来进行维度调整，如下图b）。由于BN的位置后来实验证明改为BN、Relu、Conv训练得更快，所以这里不考虑原始的设计结构，而且因为bottleneck是为了加深网络而提出的，这里也不考虑，只在basic结构上修改。

![](./assets/modules.png)

作者提出有三种方法来增加残差模块的表示能力：
- 每个残差块增加更多的卷积层
- 通过增加特征面来增加卷积层宽度
- 增大卷积核尺寸

由于VGG和Inception研究表明小卷积有着更好的效果，所以对卷积核尺寸都控制在3x3以下，主要就是研究卷积层数目和卷积层宽度的影响。为此，引入了两个系数l（深度系数）和k（宽度系数），前者表示残差模块中的卷积层数目，后者表示特征面数（后文解释）。

提出了上图c和d的两种宽残差模块，下表详细描述了结构内部。

![](./assets/block.png)

事实上，令k=1就会发现就是基础的ResNet论文中所用的结构，通道分别为16、32和64，堆叠$6*N+2$的深度。**本文作者加了个系数k从而通过增加输出的通道数来控制较小的N从而实现更宽（wide）的网络。** 调整l和k保证网络的复杂性基本不变进行了各种结构的尝试。由于网络加宽使得参数量增加，需要更有效的正则化方法，BN虽然有效但是需要配合充分的数据增强，需要尽量避免使用，通过在每个残差块的卷积层之间增加dropout并在Relu后对下一个残差块中的BN进行扰动以防过拟合。在非常深的残差网络中，上述方法有助于解决特征重用问题。

在选择合适的残差块设计后（事实上效果差别不大），与其他网络在Cifar数据集上进行对比实验，结果如下图。WRN40-4与ResNet1001相比，参数量类似，结果类似，训练速度却快了很多。这表明，增加宽度对模型的性能是有提升的，但不能武断认为哪种更好，需要寻优配合才行，不过，同等参数，宽度网络比深度网络容易训练。

![](./assets/cifar.png)

由于实验进行较多，这里不具体列举更多数据集上的实验结果了。最后，经过参数的不断尝试，获得在各个数据集上最好的结果的模型结构如下表。

![](./assets/best.png)

## 项目实战
我用Pytorch简单实现了WRN的网络结构（WRN50-2）并在Caltech101上进行了训练（原论文作者也在Github开源了代码，感兴趣可以直接访问），具体的模型代码如下。由于针对的是Caltech101数据集所以使用作者论文中对ImageNet的处理思路，基于Bottleneck结构的ResNet进行修改得到widen模型。

同时，训练过程的数据增强也采用论文里的思路：随机翻转、随机裁减以及标准化。

下面是使用PyTorch构建模型的源码，完整的训练代码可以在文末Github访问到。
```python
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
```

简单可视化训练过程（loss图）如下，训练速度确实比原始ResNet快很多（同等复杂性网络）。

![](./assets/his.png)


## 补充说明
本文其实相对是比较简单的论文，验证了宽度给模型性能带来的提升，为很多网络结构设计提供了新的思路，我在很多上层计算机视觉任务的特征提取网络中都看到了WRN的影子，是非常实用的残差网络思路。篇幅限制，并没有太过深入分析原理，感兴趣可以查看原论文，并不复杂。

## 参考论文

[Zagoruyko S, Komodakis N. Wide residual networks[J]. arXiv preprint arXiv:1605.07146, 2016.](https://arxiv.org/abs/1605.07146)
