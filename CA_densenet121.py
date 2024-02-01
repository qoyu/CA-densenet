
import torchvision.models as models

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class ModifiedDenseNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedDenseNet, self).__init__()
        # 加载预训练的DenseNet121
        densenet = models.densenet121(pretrained=False)

        # 替换分类器
        densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)

        self.features = densenet.features

        # DenseBlock的输出通道数
        num_channels = [256, 512, 1024, 1024]

        # 为每个DenseBlock添加坐标注意力模块
        self.coord_att1 = CoordAtt(num_channels[0], num_channels[0])
        self.coord_att2 = CoordAtt(num_channels[1], num_channels[1])
        self.coord_att3 = CoordAtt(num_channels[2], num_channels[2])
        self.coord_att4 = CoordAtt(num_channels[3], num_channels[3])

        self.classifier = densenet.classifier

    def forward(self, x):
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        x = self.features.denseblock1(x)
        x = self.coord_att1(x)
        x = self.features.transition1(x)

        x = self.features.denseblock2(x)
        x = self.coord_att2(x)
        x = self.features.transition2(x)

        x = self.features.denseblock3(x)
        x = self.coord_att3(x)
        x = self.features.transition3(x)

        x = self.features.denseblock4(x)
        x = self.coord_att4(x)

        x = self.features.norm5(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

# 创建模型实例

if __name__ == '__main__':  #定义一个主函数
    x = torch.rand([1,3,224,224])
    num_classes = 2
    model = ModifiedDenseNet(num_classes=num_classes)
    y = model(x)