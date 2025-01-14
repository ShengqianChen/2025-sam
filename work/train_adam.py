import argparse
import torch
import matplotlib.pyplot as plt  # 用于绘制图表

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys; sys.path.append("..")
from sam import SAM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.") #默认不适用ASAM
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size used in the training and validation loop.") #默认为256batch_size
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.") 
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.") #默认无dropout
    parser.add_argument("--epochs", default=10, type=int, help="Total number of epochs.") #默认10epoch
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.") #默认学习率初始为0.1
    parser.add_argument("--momentum", default=0.0, type=float, help="SGD Momentum.") #SGD优化器是否使用动量
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.") 
    parser.add_argument("--rho", default=0.5, type=int, help="Rho parameter for SAM.") 
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--model_save_path", default="model.pth", type=str, help="Path to save the trained model.") #保存模型的路径
    parser.add_argument("--log_save_path", default="train_log.txt", type=str, help="Path to save the training log.") #保存日志的路径
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0005)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    # 用于存储每个epoch的loss和accuracy
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                #scheduler(epoch) #是否采用学习率迭代更新规则

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
                
                correct_predictions += correct.sum().item()
                total_samples += targets.size(0)
                epoch_loss += loss.sum().item()

         # 计算每个epoch的平均loss和准确率
        epoch_accuracy = correct_predictions / total_samples
        epoch_loss /= total_samples

        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

    # 保存训练好的模型
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

    # 将训练日志保存到文件
    with open(args.log_save_path, "w") as log_file:
        for epoch in range(args.epochs):
            log_file.write(f"Epoch {epoch + 1}: Loss = {epoch_losses[epoch]:.4f}, Accuracy = {epoch_accuracies[epoch] * 100:.2f}%\n")

    # 刷新日志
    log.flush()
