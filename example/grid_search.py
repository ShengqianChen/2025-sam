import argparse
import torch
import os
import matplotlib.pyplot as plt
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import sys; sys.path.append("..")
from sam import SAM

def hyperparameter_search(rho_values, args):
    accuracies = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_len = int(0.9 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    print(train_len, val_len)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

    # Initialize the model and optimizer outside the loop
    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

    for rho in rho_values:
        print(f"Running with rho={rho}")

        
        # Initialize model, optimizer, and scheduler for each hyperparameter combination
        model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=rho, adaptive=False, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
        
        log = Log(log_each=10)
        # Train the model for 50 epochs
        for epoch in range(1):
            model.train()
            log.train(len_dataset=len(train_loader))

            for batch in train_loader:
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
                    scheduler(epoch)

            # Evaluate the model
            model.eval()
            log.eval(len_dataset=len(val_loader))

            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = (b.to(device) for b in batch)

                    predictions = model(inputs)
                    correct_predictions += (torch.argmax(predictions, 1) == targets).sum().item()
                    total_samples += targets.size(0)

                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())

            accuracy = correct_predictions / total_samples

            accuracies.append((rho, accuracy * 100))

    return accuracies


def plot_accuracy_bar(accuracies, rho_values):
    print("Accuracy table:")
    for rho, accuracy in accuracies:
        print(f"RHO: {rho}, Accuracy: {accuracy}")

    # Prepare accuracy values
    accuracy_values = [accuracy for _, accuracy in accuracies]

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(rho_values)), accuracy_values, color='skyblue')
    plt.xlabel('SAM Rho')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Different Rho Values')
    plt.xticks(range(len(rho_values)), [f"{rho:.2f}" for rho in rho_values])  # 设置x轴刻度标签为rho_values
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha='center', va='bottom')

    # Ensure the img directory exists in the parent directory
    current_dir = os.path.dirname(__file__)
    img_dir = os.path.join(current_dir, '..', 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Save the figure to the img directory in the parent directory
    plt.savefig(os.path.join(img_dir, 'accuracy_bar.png'))

    # Optionally, display the plot
    plt.show()

def main():
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train.")
    parser.add_argument("--depth", default=16, type=int, help="Depth of the network.")
    parser.add_argument("--width_factor", default=8, type=int, help="Width factor for the network.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing factor.")
    parser.add_argument("--adaptive", default=True, type=bool, help="Whether to use Adaptive SAM.")
    parser.add_argument("--threads", default=4, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="Weight decay for optimizer.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho value for SAM.")
    args = parser.parse_args()

    # Define hyperparameter search ranges for rho
    rho_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    # Perform hyperparameter search
    accuracies = hyperparameter_search(rho_values, args)

    # Plot the accuracy bar chart
    plot_accuracy_bar(accuracies, rho_values)


if __name__ == "__main__":
    main()