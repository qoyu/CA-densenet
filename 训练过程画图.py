import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

# 路径到你的TensorBoard日志文件
log_path = 'A:/pythonProject/logs_train4/'

# 创建一个事件累加器实例来加载日志数据
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()  # 加载日志数据

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # 第一个值保持不变
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed




# 假设我们关注的是 'AUC' 和 'loss' 标量
if 'train_accuracy' in ea.scalars.Keys() and 'train_loss' in ea.scalars.Keys():
    auc_values = ea.scalars.Items('train_accuracy')
    loss_values = ea.scalars.Items('train_loss')

    # 提取步骤和AUC值
    steps_auc = [x.step for x in auc_values]
    aucs = [x.value for x in auc_values]

    # 提取步骤和loss值
    steps_loss = [x.step for x in loss_values]
    losses = [x.value for x in loss_values]

    # 假设aucs和losses是我们从TensorBoard数据中提取的原始数值
    smoothed_aucs = smooth(aucs, 0.855)
    smoothed_losses = smooth(losses, 0.855)
    # 创建图形和轴对象
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制AUC曲线
    ax1.plot(steps_auc, aucs, 'b', label='train_accuracy')
    ax1.set_title('AUC Over Time')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('AUC')
    ax1.legend()

    # 绘制Loss曲线
    ax2.plot(steps_loss, losses, 'r', label='train_loss')
    ax2.set_title('Training Loss Over Time')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # 移除周围的框线
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', colors='gray')


        # 设置背景颜色
        ax1.set_facecolor('#f0f0f0')
        ax2.set_facecolor('#f0f0f0')
        fig.set_facecolor('#f0f0f0')

# 调整布局
plt.tight_layout()

# 保存图像或显示
plt.show()
