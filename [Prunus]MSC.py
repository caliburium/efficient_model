import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functions.lr_lambda import lr_lambda
from model.Prunus import Prunus, prunus_weights
from dataloader.data_loader import data_loader
import numpy as np
import wandb
import time
from tqdm import tqdm

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--pretrain_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=200)

    # Optimizer
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    # parameter weight amplifier
    parser.add_argument('--fc_weight', type=float, default=1.0)
    parser.add_argument('--disc_weight', type=float, default=1.0)


    args = parser.parse_args()

    pre_epochs = args.pretrain_epoch
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="Efficient Model",
               entity="hails",
               config=args.__dict__,
               name="Prunus_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    cifar_loader, cifar_loader_test = data_loader('CIFAR10', args.batch_size)

    print("Data load complete, start training")

    model = Prunus().to(device)
    pre_opt = optim.Adam(model.parameters(), lr=1e-5)
    param = prunus_weights(model, args.lr, args.fc_weight, args.disc_weight)
    optimizer = optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(pre_epochs):
        model.train()
        i = 0

        total_mnist_loss, total_svhn_loss, total_cifar_loss = 0, 0, 0
        total_mnist_acc, total_svhn_acc, total_cifar_acc = 0, 0, 0
        num_batches = min(len(mnist_loader), len(svhn_loader), len(cifar_loader))

        for mnist_data, svhn_data, cifar_data in zip(mnist_loader, svhn_loader, cifar_loader):
            lambda_p = 1.0

            # Training with source data
            mnist_images, mnist_labels = mnist_data.to(device)
            svhn_images, svhn_labels = svhn_data.to(device)
            cifar_images, cifar_labels = cifar_data.to(device)

            mnist_out_part, mnist_out, _ = model(mnist_images, alpha=lambda_p)
            mnist_loss = criterion(mnist_out_part, mnist_labels)
            svhn_out_part, svhn_out, _ = model(svhn_images, alpha=lambda_p)
            svhn_loss = criterion(svhn_out_part, svhn_labels)
            cifar_out_part, cifar_out, _ = model(cifar_images, alpha=lambda_p)
            cifar_loss = criterion(svhn_out, cifar_labels)

            loss = mnist_loss + svhn_loss + cifar_loss

            pre_opt.zero_grad()
            loss.backward()
            pre_opt.step()

            # Pretrain하는 부분이라 Partitioned 안된 상태로 봐야하는거아닌가?
            mnist_acc = (torch.argmax(mnist_out_part, dim=1) == mnist_labels).sum().item() / mnist_labels.size(0)
            svhn_acc = (torch.argmax(svhn_out_part, dim=1) == svhn_labels).sum().item() / svhn_labels.size(0)
            cifar_acc = (torch.argmax(cifar_out_part, dim=1) == cifar_labels).sum().item() / cifar_labels.size(0)

            total_mnist_loss += mnist_loss.item()
            total_svhn_loss += svhn_loss.item()
            total_cifar_loss += cifar_loss.item()
            total_mnist_acc += mnist_acc
            total_svhn_acc += svhn_acc
            total_cifar_acc += cifar_acc

            print(f'Batches [{i + 1}/{min(len(mnist_loader), len(svhn_loader), len(cifar_loader))}] | '
                  f'MNIST Loss: {mnist_loss.item():.4f} | '
                  f'SVHN Loss: {svhn_loss.item():.4f} | '
                  f'CIFAR10 Loss: {cifar_loss.item():.4f} | '
                  f'MNIST Accuracy: {mnist_acc * 100:.3f}% | '
                  f'SVHN Accuracy: {svhn_acc * 100:.3f}% | '
                  f'CIFAR10 Accuracy: {cifar_acc * 100:.3f}%')

            i += 1

        wandb.log({
            'PreTrain/MNIST Loss': total_mnist_loss / num_batches,
            'PreTrain/SVHN Loss': total_svhn_loss / num_batches,
            'PreTrain/CIFAR10 Loss': total_cifar_loss / num_batches,
            'PreTrain/MNIST Accuracy': total_mnist_acc / num_batches,
            'PreTrain/SVHN Accuracy': total_svhn_acc / num_batches,
            'PreTrain/CIFAR10 Accuracy': total_cifar_acc / num_batches,
        }, step=pre_epochs + 1)


    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()

        loss_ani_domain_epoch, loss_num_domain_epoch, loss_label_epoch = 0, 0, 0
        total_mnist_correct, total_svhn_correct, total_cifar_correct = 0, 0, 0
        total_mnist_domain_correct, total_svhn_domain_correct, total_cifar_domain_correct = 0, 0, 0
        total_mnist_samples, total_svhn_samples, total_cifar_samples = 0, 0, 0

        for i, (mnist_data, svhn_data, cifar_data) in enumerate(zip(mnist_loader, svhn_loader, cifar_loader)):
            p = (float(i + epoch * min(len(mnist_loader), len(svhn_loader), len(cifar_loader))) /
                 num_epochs / min(len(mnist_loader), len(svhn_loader), len(cifar_loader)))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            # Training with source data
            mnist_images, mnist_labels = mnist_data.to(device)
            svhn_images, svhn_labels = svhn_data.to(device)
            cifar_images, cifar_labels = cifar_data.to(device)

            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            svhn_dlabels = torch.full((svhn_images.size(0),), 0, dtype=torch.long, device=device)
            cifar_dlabels = torch.full((cifar_images.size(0),), 1, dtype=torch.long, device=device)

            mnist_out_part, mnist_class_out, mnist_domain_out = model(mnist_images, alpha=lambda_p)
            svhn_out_part, svhn_class_out, svhn_domain_out = model(svhn_images, alpha=lambda_p)
            cifar_out_part, cifar_class_out, cifar_domain_out = model(cifar_images, alpha=lambda_p)

            mnist_label_loss = criterion(mnist_out_part, mnist_labels)
            svhn_label_loss = criterion(svhn_out_part, svhn_labels)
            cifar_label_loss = criterion(cifar_out_part, cifar_labels)

            label_loss = mnist_label_loss + svhn_label_loss + cifar_label_loss
            loss_label_epoch += label_loss.item()

            mnist_domain_loss = criterion(mnist_domain_out, mnist_dlabels)
            svhn_domain_loss = criterion(svhn_domain_out, svhn_dlabels)
            cifar_domain_loss = criterion(cifar_domain_out, cifar_dlabels)

            domain_num_loss = mnist_domain_loss + svhn_domain_loss
            loss_num_domain_epoch += domain_num_loss.item()
            domain_ani_loss = cifar_domain_loss
            loss_ani_domain_epoch += domain_ani_loss.item()

            loss = domain_num_loss + domain_ani_loss + label_loss
            loss.backward()
            optimizer.step()

            mnist_correct = (torch.argmax(mnist_out_part, dim=1) == mnist_labels).sum().item()
            svhn_correct = (torch.argmax(svhn_out_part, dim=1) == svhn_labels).sum().item()
            cifar_correct = (torch.argmax(cifar_out_part, dim=1) == cifar_labels).sum().item()

            mnist_domain_correct = (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
            svhn_domain_correct = (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()
            cifar_domain_correct = (torch.argmax(cifar_domain_out, dim=1) == cifar_dlabels).sum().item()

            total_mnist_correct += mnist_correct
            total_svhn_correct += svhn_correct
            total_cifar_correct += cifar_correct

            total_mnist_domain_correct += mnist_domain_correct
            total_svhn_domain_correct += svhn_domain_correct
            total_cifar_domain_correct += cifar_domain_correct

            total_mnist_samples += mnist_labels.size(0)
            total_svhn_samples += svhn_labels.size(0)
            total_cifar_samples += cifar_labels.size(0)


            print(f'Batches [{i + 1}/{min(len(mnist_loader), len(svhn_loader), len(cifar_loader))}] | '
                  f'MNIST Loss: {mnist_label_loss.item():.4f} | '
                  f'SVHN Loss: {svhn_label_loss.item():.4f} | '
                  f'CIFAR Loss: {cifar_label_loss.item():.4f} | '
                  f'Label Loss: {label_loss.item():.4f}')

            print(f'MNIST Acc: {mnist_correct / mnist_labels.size(0) * 100:.3f}% | '
                  f'SVHN Acc: {svhn_correct / svhn_labels.size(0) * 100:.3f}% | '
                  f'CIFAR Acc: {cifar_correct / cifar_labels.size(0) * 100:.3f}% | '
                  f'MNIST Domain Acc: {mnist_domain_correct / mnist_dlabels.size(0) * 100:.3f}% | '
                  f'SVHN Domain Acc: {svhn_domain_correct / svhn_dlabels.size(0) * 100:.3f}% | '
                  f'CIFAR Domain Acc: {cifar_domain_correct / cifar_dlabels.size(0) * 100:.3f}%')

        scheduler.step()

        mnist_acc_epoch = total_mnist_correct / total_mnist_samples * 100
        svhn_acc_epoch = total_svhn_correct / total_svhn_samples * 100
        cifar_acc_epoch = total_cifar_correct / total_cifar_samples * 100

        mnist_domain_acc_epoch = total_mnist_domain_correct / total_mnist_samples * 100
        svhn_domain_acc_epoch = total_svhn_domain_correct / total_svhn_samples * 100
        cifar_domain_acc_epoch = total_cifar_domain_correct / total_cifar_samples * 100

        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'Numbers Domain Loss: {loss_num_domain_epoch:.4f} | '
              f'Animals Domain Loss: {loss_ani_domain_epoch:.4f} | '
              f'Label Loss: {loss_label_epoch:.4f} | '
              f'Total Loss: {loss_num_domain_epoch + loss_ani_domain_epoch + loss_label_epoch:.4f} | '
              f'Time: {end_time - start_time:.2f} sec')

        print(f'MNIST Acc: {mnist_acc_epoch:.3f}% | '
              f'SVHN Acc: {svhn_acc_epoch:.3f}% | '
              f'CIFAR Acc: {cifar_acc_epoch:.3f}% | '
              f'MNIST Domain Acc: {mnist_domain_acc_epoch:.3f}% | '
              f'SVHN Domain Acc: {svhn_domain_acc_epoch:.3f}% | '
              f'CIFAR Domain Acc: {cifar_domain_acc_epoch:.3f}%')

        wandb.log({
            'Train/Domain Numbers Loss': loss_num_domain_epoch,
            'Train/Domain Animals Loss': loss_ani_domain_epoch,
            'Train/Label Loss': loss_label_epoch,
            'Train/Total Loss': loss_num_domain_epoch + loss_ani_domain_epoch + loss_label_epoch,
            'Train/MNIST Label Accuracy': mnist_acc_epoch,
            'Train/SVHN Label Accuracy': svhn_acc_epoch,
            'Train/CIFAR Label Accuracy': cifar_acc_epoch,
            'Train/MNIST Domain Accuracy': mnist_domain_acc_epoch,
            'Train/SVHN Domain Accuracy': svhn_domain_acc_epoch,
            'Train/CIFAR Domain Accuracy': cifar_domain_acc_epoch,
            'Train/Training Time': end_time - start_time
        }, step=num_epochs+1)

        model.eval()

        def tester(loader, group, domain_label):
            label_correct, domain_correct, total = 0, 0, 0
            for images, labels in loader:
                images = images.to(device)

                class_output_partitioned, _, domain_output = model(images, alpha=0.0)
                label_preds = F.log_softmax(class_output_partitioned, dim=1)
                domain_preds = F.log_softmax(domain_output, dim=1)

                _, label_predicted = torch.max(label_preds.data, 1)
                _, domain_predicted = torch.max(domain_preds.data, 1)
                total += images.size(0)
                label_correct += (label_predicted == labels).sum().item()
                domain_correct += (domain_predicted == domain_label).sum().item()

            label_acc = label_correct / total * 100
            domain_acc = domain_correct / total * 100

            log_data = {
                f'Test/Label {group} Accuracy': label_acc,
                f'Test/Domain {group} Accuracy': domain_acc
            }
            wandb.log(log_data, step=epoch + 1)

            print(f'Test {group} | Label Acc: {label_acc:.3f}% | Domain Acc: {domain_acc:.3f}%')

        with torch.no_grad():
            tester(mnist_loader_test, 'MNIST', 0)
            tester(svhn_loader_test, 'SVHN', 0)
            tester(cifar_loader_test, 'CIFAR', 1)


if __name__ == '__main__':
    main()