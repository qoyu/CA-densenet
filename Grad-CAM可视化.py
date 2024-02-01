import torch
from torch.nn.functional import relu
from CA_densenet121 import ModifiedDenseNet
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


print(torch.cuda.is_available())

def generate_grad_cam(img, model, target_layer, target_class):
    model.eval()

    # 获取目标层
    target = [model.features.denseblock4] if target_layer == "denseblock4" else None

    if target is None:
        raise ValueError("Invalid target layer")

    # 注册钩子
    def forward_hook(module, input, output):
        global feature_maps
        feature_maps = output

    def backward_hook(module, grad_input, grad_output):
        global gradients
        gradients = grad_output[0]

    hook_forward = target[0].register_forward_hook(forward_hook)
    hook_backward = target[0].register_backward_hook(backward_hook)

    # 前向传播
    output = model(img)
    model.zero_grad()

    # 反向传播
    one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float32).to(device)
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output, retain_graph=True)

    # 生成 Grad-CAM 热图
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(feature_maps.size()[1]):
        feature_maps[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(feature_maps, dim=1).squeeze()
    heatmap = relu(heatmap)
    heatmap /= torch.max(heatmap)

    # 清除钩子
    hook_forward.remove()
    hook_backward.remove()

    return heatmap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('A:/pythonProject/model_pth/best_model16.pth')

model.eval()


# 加载图像
image = Image.open("data/train/Brain Tumor/Cancer (34).tif")


# 定义与训练时相同的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 假设您的模型是用224x224图像训练的
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet的均值和标准差
])

# 应用转换
img = transform(image)

# 添加批次维度
img = img.unsqueeze(0)
img = img.to(device)


heatmap = generate_grad_cam(img, model, "denseblock4", 0)

# 将Tensor转换为NumPy数组
img_np = img.squeeze().cpu().numpy()
img_np = np.transpose(img_np, (1, 2, 0))
img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # 反归一化
img_np = np.clip(img_np, 0, 1)

# 将热图转换为NumPy数组并放大到原始图像大小
heatmap_np = heatmap.cpu().detach().numpy()  # 使用 detach()
heatmap_np = cv2.resize(heatmap_np, (img_np.shape[1], img_np.shape[0]))
heatmap_np = np.uint8(255 * heatmap_np)  # 将热图转换为0-255的范围
heatmap_np = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)  # 应用颜色映射


# # 将热图叠加到原始图像上的强度参数
# heatmap_intensity = 0.3  # 可以调整这个值来改变热图的强度
#
# # 使用不同的颜色映射
# color_map = cv2.COLORMAP_HOT  # 可以尝试不同的颜色映射，例如 COLORMAP_HOT, COLORMAP_COOL, 等
#
# # 应用高斯模糊来平滑热图（可选）
# heatmap_np = cv2.GaussianBlur(heatmap_np, (11, 11), 0)
#
# # 将热图叠加到原始图像上
# superimposed_img = heatmap_np * heatmap_intensity + img_np * 255
# superimposed_img = np.uint8(superimposed_img)
# superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
#
# # 显示图像
# plt.figure(figsize=(4, 4))
# plt.imshow(superimposed_img)
# plt.axis('off')
# plt.show()


# 使用不同的颜色映射
color_map = cv2.COLORMAP_VIRIDIS

# 调整透明度
alpha = 0.3  # 可以进一步调整以改变热图的透明度

# 应用颜色映射并调整大小
heatmap_np = cv2.applyColorMap(cv2.resize(heatmap_np, (img_np.shape[1], img_np.shape[0])), color_map)

# 应用高斯模糊（可选）
heatmap_np = cv2.GaussianBlur(heatmap_np, (3, 3), 0)

# 叠加热图到原始图像
superimposed_img = heatmap_np * alpha + img_np * 255 * (1 - alpha)
superimposed_img = superimposed_img / np.max(superimposed_img)  # 归一化以增强效果
superimposed_img = np.uint8(255 * superimposed_img)  # 转换回0-255范围
superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# 显示图像
plt.figure(figsize=(4, 4))
plt.imshow(superimposed_img)
plt.axis('off')
plt.show()


