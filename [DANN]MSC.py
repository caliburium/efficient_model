import argparse
import numpy as np
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
    parser.add_argument('--epoch', type=int, default=100)
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
               name="[DANN]MSC_" + str(args.hidden_size) + "_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    cifar_loader, cifar_loader_test = data_loader('CIFAR10', args.batch_size)

    print("Data load complete, start training")

    model = DANN(hidden_size=args.hidden_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    for epoch in range(num_epochs):
        model.train()
        total_mnist_domain_loss, total_svhn_domain_loss, total_cifar_domain_loss, total_domain_loss = 0, 0, 0, 0
        total_mnist_domain_correct, total_svhn_domain_correct, total_cifar_domain_correct = 0, 0, 0
        total_mnist_loss, total_svhn_loss, total_cifar_loss, total_label_loss = 0, 0, 0, 0
        total_mnist_correct, total_svhn_correct, total_cifar_correct = 0, 0, 0
        total_samples = 0

        for i, (mnist_data, svhn_data, cifar_data) in enumerate(zip(mnist_loader, svhn_loader, cifar_loader)):
            p = epoch / num_epochs
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            mnist_images, mnist_labels = mnist_data
            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            svhn_images, svhn_labels = svhn_data
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            cifar_images, cifar_labels = cifar_data
            cifar_images, cifar_labels = cifar_images.to(device), cifar_labels.to(device)
            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            svhn_dlabels = torch.full((svhn_images.size(0),), 0, dtype=torch.long, device=device)
            cifar_dlabels = torch.full((cifar_images.size(0),), 1, dtype=torch.long, device=device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                mnist_out, mnist_domain_out = model(mnist_images, alpha=lambda_p)
                svhn_out, svhn_domain_out = model(svhn_images, alpha=lambda_p)
                cifar_out, cifar_domain_out = model(cifar_images, alpha=lambda_p)

                mnist_label_loss = criterion(mnist_out, mnist_labels)
                svhn_label_loss = criterion(svhn_out, svhn_labels)
                cifar_label_loss = criterion(cifar_out, cifar_labels)
                label_loss = (mnist_label_loss + svhn_label_loss) * 0.5 + cifar_label_loss

                mnist_domain_loss = criterion(mnist_domain_out, mnist_dlabels)
                svhn_domain_loss = criterion(svhn_domain_out, svhn_dlabels)
                cifar_domain_loss = criterion(cifar_domain_out, cifar_dlabels)
                domain_loss = (mnist_domain_loss + svhn_domain_loss) * 0.5 + cifar_domain_loss

                loss = label_loss + domain_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_label_loss += label_loss.item()
            total_mnist_loss += mnist_label_loss.item()
            total_svhn_loss += svhn_label_loss.item()
            total_cifar_loss += cifar_label_loss.item()

            total_domain_loss += domain_loss.item()
            total_mnist_domain_loss += mnist_domain_loss.item()
            total_svhn_domain_loss += svhn_domain_loss.item()
            total_cifar_domain_loss += cifar_domain_loss.item()

            total_mnist_correct += (torch.argmax(mnist_out, dim=1) == mnist_labels).sum().item()
            total_svhn_correct += (torch.argmax(svhn_out, dim=1) == svhn_labels).sum().item()
            total_cifar_correct += (torch.argmax(cifar_out, dim=1) == cifar_labels).sum().item()

            total_mnist_domain_correct += (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
            total_svhn_domain_correct += (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()
            total_cifar_domain_correct += (torch.argmax(cifar_domain_out, dim=1) == cifar_dlabels).sum().item()

            total_samples += mnist_labels.size(0)

        scheduler.step()

        mnist_domain_avg_loss = total_mnist_domain_loss / total_samples
        svhn_domain_avg_loss = total_svhn_domain_loss / total_samples
        cifar_domain_avg_loss = total_cifar_domain_loss / total_samples
        domain_avg_loss = total_domain_loss / (total_samples * 3)

        mnist_avg_loss = total_mnist_loss / total_samples
        svhn_avg_loss = total_svhn_loss / total_samples
        cifar_avg_loss = total_cifar_loss / total_samples
        label_avg_loss = total_label_loss / (total_samples * 3)

        mnist_acc_epoch = total_mnist_correct / total_samples * 100
        svhn_acc_epoch = total_svhn_correct / total_samples * 100
        cifar_acc_epoch = total_cifar_correct / total_samples * 100

        mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples * 100
        svhn_domain_acc_epoch = total_svhn_domain_correct / total_samples * 100
        cifar_domain_acc_epoch = total_cifar_domain_correct / total_samples * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'Label Loss: {label_avg_loss:.4f} | '
              f'Domain Loss: {domain_avg_loss:.4f} | '
              f'Total Loss: {label_avg_loss + domain_avg_loss:.4f} | '
        )
        print(f'MNIST Loss: {mnist_avg_loss:.4f} | '
              f'SVHN Loss: {svhn_avg_loss:.4f} | '
              f'CIFAR Loss: {cifar_avg_loss:.4f} | '
              f'MNIST Domain Loss: {mnist_domain_avg_loss:.4f} | '
              f'SVHN Domain Loss: {svhn_domain_avg_loss:.4f} | '
              f'CIFAR Domain Loss: {cifar_domain_avg_loss:.4f} | '
        )
        print(f'MNIST Acc: {mnist_acc_epoch:.3f}% | '
              f'SVHN Acc: {svhn_acc_epoch:.3f}% | '
              f'CIFAR Acc: {cifar_acc_epoch:.3f}% | '
              f'MNIST Domain Acc: {mnist_domain_acc_epoch:.3f}% | '
              f'SVHN Domain Acc: {svhn_domain_acc_epoch:.3f}% | '
              f'CIFAR Domain Acc: {cifar_domain_acc_epoch:.3f}% |')

        wandb.log({
            'Train/MNIST Label Loss': mnist_avg_loss,
            'Train/SVHN Label Loss': svhn_avg_loss,
            'Train/CIFAR Label Loss': cifar_avg_loss,
            'Train/Label Loss': label_avg_loss,
            'Train/Domain MNIST Loss': mnist_domain_avg_loss,
            'Train/Domain SVHN Loss': svhn_domain_avg_loss,
            'Train/Domain CIFAR Loss': cifar_domain_avg_loss,
            'Train/Domain Loss': domain_avg_loss,
            'Train/Total Loss': (label_avg_loss + domain_avg_loss),
            'Train/MNIST Label Accuracy': mnist_acc_epoch,
            'Train/SVHN Label Accuracy': svhn_acc_epoch,
            'Train/CIFAR Label Accuracy': cifar_acc_epoch,
            'Train/MNIST Domain Accuracy': mnist_domain_acc_epoch,
            'Train/SVHN Domain Accuracy': svhn_domain_acc_epoch,
            'Train/CIFAR Domain Accuracy': cifar_domain_acc_epoch,
        }, step=epoch + 1)

        model.eval()

        def tester(loader, group, domain_label):
            label_correct, domain_correct, total = 0, 0, 0

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                class_output, domain_output = model(images, alpha=0.0)
                label_correct += (torch.argmax(class_output, dim=1) == labels).sum().item()
                domain_correct += (torch.argmax(domain_output, dim=1) == domain_label).sum().item()
                total += images.size(0)

            label_acc = label_correct / total * 100
            domain_acc = domain_correct / total * 100

            wandb.log({
                f'Test/Label {group} Accuracy': label_acc,
                f'Test/Domain {group} Accuracy': domain_acc,
            }, step=epoch + 1)

            print(f'Test {group} | Label Acc: {label_acc:.3f}% | Domain Acc: {domain_acc:.3f}%')

        with torch.no_grad():
            tester(mnist_loader_test, 'MNIST', 0)
            tester(svhn_loader_test, 'SVHN', 0)
            tester(cifar_loader_test, 'CIFAR', 1)

if __name__ == '__main__':
    main()
