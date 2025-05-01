import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from functions.lr_lambda import lr_lambda
import wandb
from dataloader.data_loader import data_loader
from model.DANN import DANN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    args = parser.parse_args()
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(entity="hails",
               project="Efficient Model",
               config=args.__dict__,
               name="[CNN]S:SVHN/T:MNIST_" + str(args.hidden_size)
                    + "_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)

    print("Data load complete, start training")

    model = DANN(hidden_size=args.hidden_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    for epoch in range(num_epochs):
        model.train()
        total_svhn_correct, total_loss, total_samples = 0, 0, 0

        for svhn_images, svhn_labels in svhn_loader:
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                svhn_out = model.cnn(svhn_images)
                loss = criterion(svhn_out, svhn_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_svhn_correct += (torch.argmax(svhn_out, dim=1) == svhn_labels).sum().item()
            total_samples += svhn_labels.size(0)

        # scheduler.step()

        total_avg_loss = total_loss / total_samples
        svhn_acc_epoch = total_svhn_correct / total_samples * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'Total Loss: {total_avg_loss:.4f} | '
              f'SVHN Acc: {svhn_acc_epoch:.3f}% | '
        )

        wandb.log({
            'Train/Total Loss': total_avg_loss,
            'Train/SVHN Label Accuracy': svhn_acc_epoch,
        }, step=epoch + 1)

        model.eval()

        def tester(loader, group):
            label_correct, total = 0, 0

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                class_output = model.cnn(images)
                label_correct += (torch.argmax(class_output, dim=1) == labels).sum().item()
                total += images.size(0)

            label_acc = label_correct / total * 100

            wandb.log({
                f'Test/Label {group} Accuracy': label_acc,
            }, step=epoch + 1)

            print(f'Test {group} | Label Acc: {label_acc:.3f}%')

        with torch.no_grad():
            tester(mnist_loader_test, 'MNIST')
            tester(svhn_loader_test, 'SVHN')

if __name__ == '__main__':
    main()
