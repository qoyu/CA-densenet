import torch
from alexnet import MyAlexnet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage


ROOT_TRAIN = r'A:/pythonProject/data/train' #导入训练集和测试集
ROOT_TEST = r'A:/pythonProject/data/val'

#将图像像素值归一化到【-1,1】之间 表示将像素值从0到1的范围标准化为-1到1的范围
normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

#数据处理 resize 数据增强 变成张量 归一化
train_transform = transforms.Compose([
    transforms.Resize((224,224)), #定义尺寸
    transforms.RandomVerticalFlip(), #垂直翻转 增加一倍训练样本 但不用本地储存
    transforms.ToTensor(),
    normalize])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize])

train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform) #用于加载位于 ROOT_TRAIN 目录下的训练图像数据，并应用了之前定义的 train_transform 数据预处理流水线。
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

#dataloader加载数据 设置batchsize大小 shuffle=True 参数指定了数据加载器在每个训练周期（epoch）开始时是否随机打乱训练数据的顺序。
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu' #万能语句

model = MyAlexnet().to(device) #传到GPU中去计算

#加载模型
model.load_state_dict(torch.load("A:/pythonProject/save_model/best2_model.pth"))
classes = [
    'Brain Tumor',
    'Healthy',
]

# 把张量转化为照片格式
show = ToPILImage()

# 进入到验证阶段
model.eval()
for i in range(100):
    x, y = val_dataset[i][0], val_dataset[i][1]  #从验证数据集 val_dataset 中获取第 i 个样本的输入 x 和标签 y
    #show(x).show()
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=True).to(device)
    x = torch.tensor(x).to(device)
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted}", Actual:"{actual}"')

