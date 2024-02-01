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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


ROOT_TRAIN = r'A:/pythonProject/data/train' #导入训练集和测试集
ROOT_TEST = r'A:/pythonProject/data/val'

#将图像像素值归一化到【-1,1】之间 表示将像素值从0到1的范围标准化为-1到1的范围
#normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

#数据处理 resize 数据增强 变成张量 归一化
train_transform = transforms.Compose([
    transforms.Resize((224,224)), #定义尺寸
    #transforms.Resize((299, 299)), #定义尺寸 inceptionv3专用
    transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
    # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
    # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
    # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 定义 MixUp 函数
# def mixup_data(x, y, alpha=0.3):
#     lam = np.random.beta(alpha, alpha)
#     batch_size = x.size(0)
#     index = torch.randperm(batch_size, device=x.device)  # 将 randperm 结果放在和 x 相同的设备上
#
#     mixed_x = lam * x + (1 - lam) * x[index]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam


train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)   # 用于加载位于 ROOT_TRAIN 目录下的训练图像数据，并应用了之前定义的 train_transform 数据预处理流水线。
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)


#dataloader加载数据 设置batchsize大小 shuffle=True 参数指定了数据加载器在每个训练周期（epoch）开始时是否随机打乱训练数据的顺序。
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

train_data_size = len(train_dataset)
val_data_size = len(val_dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'   # 万能语句

num_classes = 2
resnet152 = torchvision.models.mobilenet_v2(pretrained=True)  # 加载预训练模型
#resnet152.head = nn.Linear(resnet152.head.in_features, num_classes)  # 修改输出层
print(resnet152)


#resnet152.fc = nn.Linear(2048, num_classes)
resnet152.classifier[1] = nn.Linear(1280, num_classes)
print(resnet152)

model = resnet152.to(device)

# # 冻结除了最后一层之外的所有层
# for param in model.parameters():
#     param.requires_grad = False
#
# # 解冻最后两层，使其参数可训练
# for param in model.fc.parameters():
#     param.requires_grad = True
#
# for param in model.avgpool.parameters():
#     param.requires_grad = True
#
# for param in model.layer4.parameters():
#     param.requires_grad = True

#定义一个损失函数
loss_fn = nn.CrossEntropyLoss()

#定义一个优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#
# 定义冻结比例
freeze_percentage = 0.15  # 总层数的15%

# 获取模型总层数
total_layers = sum(1 for _ in model.parameters())

# 计算要冻结的层索引
freeze_index = int(total_layers * freeze_percentage)

# 冻结到计算得到的索引位置
for index, (name, param) in enumerate(model.named_parameters()):
    if index < freeze_index:
        param.requires_grad = False
    else:
        param.requires_grad = True


#定义训练函数
# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 30

# 添加tensorboard
writer = SummaryWriter("./logs_train11")
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

        # 计算准确率
        accuracy = (outputs.argmax(1) == targets).sum()
        # 在每次训练步骤之后记录损失和准确率
        writer.add_scalar("train_loss", loss.item(), total_train_step)
        writer.add_scalar("train_accuracy", accuracy.item() / len(targets), total_train_step)

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:   # 训练步骤每一百打印一次loss
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

    # 在收集到 all_probs 和 all_targets 后：
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    # 获取预测标签
    predicted_labels = (all_probs > 0.5).astype(int)  # 假设是二元分类

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_targets, predicted_labels)
    # 计算精确率、召回率、F1值和支持度
    precision, recall, f1_score, support = precision_recall_fscore_support(all_targets, predicted_labels, zero_division=1)
    # 计算特异度
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    # 计算易混淆率（EC）
    # 假设类别1是正类
    ec = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    # 计算AUC值
    auc = roc_auc_score(all_targets, all_probs)
    print("混淆矩阵:{}".format(conf_matrix))
    print("精确率:{}".format(precision))
    print("召回率:{}".format(recall))
    print("F1值:{}".format(f1_score))
    print("特异度:{}".format(specificity))
    print("易混淆率:{}".format(ec))
    print("AUC值：{}".format(auc))
    print("Loss：{}".format(total_test_loss))
    print("准确率：{}".format(total_accuracy/val_data_size))

    # writer.add_scalar("test_loss", total_test_loss, total_test_step)  # tensorboard记录日志
    # writer.add_scalar("test_accuracy", total_accuracy/val_data_size, total_test_step)  # tensorboard记录日志
    writer.add_scalar("val_loss", total_test_loss / len(val_dataloader), total_train_step)  # 使用相同的 total_train_step
    writer.add_scalar("val_accuracy", total_accuracy / val_data_size, total_train_step)  # 使用相同的 total_train_step
    writer.add_scalar("test_auc", auc, total_test_step)  # tensorboard记录日志
    total_test_step = total_test_step + 1



    # 保存最佳模型
    if auc > best_auc:
        best_auc = auc
        torch.save(model, "./model_pth/best_model11.pth")
        print("保存最佳模型")


end_time = time.time()
# 计算整个训练时长
training_duration = end_time - start_time
print(f"整个训练时长为：{training_duration} 秒")
writer.close()
