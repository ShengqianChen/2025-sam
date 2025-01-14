import matplotlib.pyplot as plt
import re

# 初始化列表，用于存储每个模型每个epoch的loss和accuracy
model1_losses = []
model1_accuracies = []
model2_losses = []
model2_accuracies = []
model3_losses = []
model3_accuracies = []

# 读取文件中的数据
def read_data(file_path, losses, accuracies):
    with open(file_path, 'r') as file:
        for line in file:
            # 使用正则表达式提取loss和accuracy
            match = re.search(r'Loss = (\d+\.\d+), Accuracy = (\d+\.\d+)%', line)
            if match:
                # 提取loss和accuracy的值
                loss = float(match.group(1))
                accuracy = float(match.group(2))
                # 将loss和accuracy添加到列表中
                losses.append(loss)
                accuracies.append(accuracy)
            else:
                print(f"Skipping line due to format issue: {line}")

# 读取三个模型的数据
read_data('save/Cifar10_sam_without_momentum_without_Steplr.txt', model1_losses, model1_accuracies)
read_data('save/Cifar10_sgd_without_momentum_without_Steplr.txt', model2_losses, model2_accuracies)
read_data('save/Cifar10_adam_without_Steplr.txt', model3_losses, model3_accuracies)

# 生成epoch的序列
epochs1 = range(1, len(model1_losses) + 1)
epochs2 = range(1, len(model2_losses) + 1)
epochs3 = range(1, len(model3_losses) + 1)

# 绘制loss曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs1, model1_losses, 'b-', label='Cifar10_sam_without_momentum_without_Steplr Loss')
plt.plot(epochs2, model2_losses, 'g-', label='Cifar10_sgd_without_momentum_without_Steplr Loss')
plt.plot(epochs3, model3_losses, 'r-', label='Cifar10_adam_without_Steplr Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(epochs1, model1_accuracies, 'b-', label='Cifar10_sam_without_momentum_without_Steplr Accuracy')
plt.plot(epochs2, model2_accuracies, 'g-', label='Cifar10_sgd_without_momentum_without_Steplr Accuracy')
plt.plot(epochs3, model3_accuracies, 'r-', label='Cifar10_adam_without_Steplr Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.savefig('save/cifar10_sam_sgd_adam_without_momentum_without_StepLR.png')
# 显示图表
plt.tight_layout()
plt.show()