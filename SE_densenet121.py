import torchvision.models as models
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SE_DenseNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SE_DenseNet, self).__init__()
        # 加载预训练的DenseNet121
        densenet = models.densenet121(pretrained=False)

        # 替换分类器
        densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)

        self.features = densenet.features

        # DenseBlock的输出通道数
        num_channels = [256, 512, 1024, 1024]

        # 为每个DenseBlock添加SE模块
        self.se_block1 = SEModule(num_channels[0])
        self.se_block2 = SEModule(num_channels[1])
        self.se_block3 = SEModule(num_channels[2])
        self.se_block4 = SEModule(num_channels[3])

        self.classifier = densenet.classifier

    def forward(self, x):
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        x = self.features.denseblock1(x)
        x = self.se_block1(x)
        x = self.features.transition1(x)

        x = self.features.denseblock2(x)
        x = self.se_block2(x)
        x = self.features.transition2(x)

        x = self.features.denseblock3(x)
        x = self.se_block3(x)
        x = self.features.transition3(x)

        x = self.features.denseblock4(x)
        x = self.se_block4(x)

        x = self.features.norm5(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
# 创建模型实例

if __name__ == '__main__':  #定义一个主函数
    x = torch.rand([1,3,224,224])
    num_classes = 2
    model = SE_DenseNet(num_classes=num_classes)
    y = model(x)