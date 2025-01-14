# 实验环境

主要工作包及版本如下：

PyTorch  2.5.1

Python  3.12

Ubuntu22.04

Cuda  12.4

Torchvision 0.11.1

其余各工作包版本可见于2025-sam/requirements.txt

------

# 数据集下载

根据选择的网络架构WRN主要在图像任务上进行实验，所以选择经典图像任务数据集Cifar-10和CIFAR-100作为实验运行的数据集，保存在work/data

Cifar-10数据集由60000张32×32的RGB彩色图片构成，共10个类别；50000张训练，10000张测试，下载网址 http://www.cs.toronto.edu/~kriz/cifar.html。

CIFAR-100数据集由60000张32×32的RGB彩色图片构成，共100个类别；50000张训练，10000张测试,数据库下载网址：http://www.cs.toronto.edu/~kriz/cifar.html。

------

# 运行方式

超参数的交叉搜索由work文件夹下的grid_search_rho.py文件完成。

使用SGD优化器的WRN训练由work文件夹下的train_sgd.py文件完成。

使用Adam优化器的WRN训练由work文件夹下的train_adam.py文件完成。

使用SAM优化器的WRN训练由work文件夹下的train_sam.py文件夹完成。

使用ASAM优化器的WRN训练由work文件夹下的train_asam.py文件夹完成。

训练文件皆可以使用通过终端传递参数的方式调整训练细节，具体如下：

| 参数名称 | 默认值 | 类型 | 描述 |
| -- | -- | -- | -- |
| adaptive | False | bool | 是否使用自适应 SAM (Adaptive SAM)。默认不使用。 |
| batch_size | 256 | int | 训练和验证循环中使用的批量大小。默认为 256。 |
| depth | 16 | int | 网络层数。 |
| dropout | 0.0 | float | Dropout 比率。默认无 Dropout。 |
| epochs | 10 | int | 总共的训练周期数。默认为 10。 |
| label_smoothing | 0.1 | float | 标签平滑参数。使用 0.0 表示不进行标签平滑。 |
| learning_rate | 0.1 | float | 训练开始时的基础学习率。默认为 0.1。 |
| momentum | 0.9 | float | SGD 优化器的动量参数。 |
| threads | 2 | int | 数据加载器使用的 CPU 线程数。 |
| rho | 0.5 | int | SAM 的 Rho 参数。 |
| weight_decay | 0.0005 | float | L2 权重衰减参数。 |
| width_factor | 8 | int | 相比普通 ResNet，模型宽度的倍数。 |
| model_save_path | model.pth | str | 保存训练模型的路径。默认为 "model.pth"。 |
| log_save_path | train_log.txt | str | 保存训练日志的路径。默认为 "train_log.txt"。 |

是否使用学习率调整策略可以在各文件代码中的相关内容通过注释完成，如在work文件夹中的train_sam.py第75行：   
> #scheduler(epoch) #是否采用学习率迭代更新规则

数据集的选择需要调整data/cifar.py里的数据集选择部分代码：

> train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

> test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

同时需要调整训练文件中的模型预测输出通道数，如在work文件夹中的train_sam.py第39行

> model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

------

# 实验结果

![](https://raw.githubusercontent.com/ShengqianChen/2025-sam/refs/heads/main/work/save/cifar10_sam_sgd_adam_without_momentum_without_StepLR.png)

------

# 项目内容

SAM.py代码文件中实现了所有有关SAM优化器的代码，并添加详细注释。

work文件夹中存放了所有实验相关的代码，下面介绍其中各文件夹：

    data文件夹中存放了实验使用的数据集，cifar.py文件用于对数据实现预处理。

    model文件夹中存放了loss损失函数的代码和WRN模型架构的代码，默认架构为WRN16-8。

    utility文件夹中存放了输出信息的辅助代码、用于数据预处理的CutOut代码、用于动态调整学习率的代码

    save文件夹中存放了所有实验保存的模型和实验过程，里面的所有图片可以通过work文件夹下的plot.py代码文件生成。