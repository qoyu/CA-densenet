import timm
from torch import nn
print(timm.list_models("swin*"))
import torch
import torchvision
from torch import nn
import numpy as np
from torch.optim import lr_scheduler
import os
from sklearn.metrics import roc_auc_score

from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import albumentations
import time

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


ROOT_TRAIN = r'A:/pythonProject/data/train' #导入训练集和测试集
ROOT_TEST = r'A:/pythonProject/data/val'

#将图像像素值归一化到【-1,1】之间 表示将像素值从0到1的范围标准化为-1到1的范围
normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

#print(timm.list_models())

model = timm.create_model('vit_base_patch16_224_miil', pretrained=False)
print(model.default_cfg) # 打印url！



#数据处理 resize 数据增强 变成张量 归一化
train_transform = transforms.Compose([
    transforms.Resize((224,224)), #定义尺寸
    transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
    transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

train_data_size = len(train_dataset)
val_data_size = len(val_dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


swin_transformer = timm.create_model("swin_tiny_patch4_window7_224",pretrained=True,pretrained_cfg_overlay=dict(file='C:/Users/ylf/.cache/torch/hub/checkpoints/swin_tiny_patch4_window7_224.pth'))

print(swin_transformer)
num_classes = 2

swin_transformer.fc = nn.Linear(768, num_classes)
print(swin_transformer)

model = swin_transformer.to(device)


#定义一个损失函数
loss_fn = nn.CrossEntropyLoss()

#定义一个优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 30

# 添加tensorboard
writer = SummaryWriter("./logs_train")
start_time = time.time()

best_auc = 0.0  # 初始最佳AUC值
for i in range(epoch):
    print("-------第{}轮训练开始---------".format(i+1))

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()   # 梯度清零
        loss.backward()        # 反向传播
        optimizer.step()        # 更新参数
        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:   # 训练步骤每一百打印一次loss
            print("训练次数{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    all_targets = []
    all_probs = []
    with torch.no_grad():  # with就是自带close()
        for data in val_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            # 手写求准确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    # 计算AUC值
    auc = roc_auc_score(all_targets, all_probs)

    print("整体测试集上的AUC值：{}".format(auc))
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的准确率：{}".format(total_accuracy/val_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)  # tensorboard记录日志
    writer.add_scalar("test_accuracy", total_accuracy/val_data_size, total_test_step)  # tensorboard记录日志
    writer.add_scalar("test_auc", auc, total_test_step)  # tensorboard记录日志
    total_test_step = total_test_step + 1

    # 保存最佳模型
    if auc > best_auc:
        best_auc = auc
        torch.save(model, "./model_pth/best_model.pth")
        print("保存最佳模型")

writer.close()