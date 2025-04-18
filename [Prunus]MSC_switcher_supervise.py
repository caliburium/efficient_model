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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--pretrain_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--feature_extractor', type=str, default='SimpleCNN')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--num_partition', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--pre_classifier_out', type=int, default=1024)
    parser.add_argument('--part_layer', type=int, default=384)

    # Optimizer
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    # parameter weight amplifier
    parser.add_argument('--pre_weight', type=float, default=1.0)
    parser.add_argument('--fc_weight', type=float, default=100.0)
    parser.add_argument('--disc_weight', type=float, default=1.0)
    parser.add_argument('--switcher_weight', type=float, default=1.0)

    args = parser.parse_args()

    pre_epochs = args.pretrain_epoch
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(entity="hails",
               project="Efficient Model",
               config=args.__dict__,
               name="[Prunus" + str(args.num_partition)
                    + "]MSC_DPtrain_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    cifar_loader, cifar_loader_test = data_loader('CIFAR10', args.batch_size)

    print("Data load complete, start training")

    model = Prunus(feature_extractor=args.feature_extractor,
                   pretrained=args.pretrained,
                   num_classes=args.num_classes,
                   pre_classifier_out=args.pre_classifier_out,
                   n_partition=args.num_partition,
                   part_layer=args.part_layer,
                   device=device
                   )

    pre_opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)

    param = prunus_weights(model, args.lr, args.pre_weight, args.fc_weight, args.disc_weight, args.switcher_weight)
    optimizer = optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    # pretrain with using non-partitioned classifier
    for epoch in range(pre_epochs):
        model.train()
        i = 0

        total_mnist_loss, total_svhn_loss, total_cifar_loss = 0, 0, 0
        total_mnist_correct, total_svhn_correct, total_cifar_correct = 0, 0, 0
        total_mnist_domain_correct, total_svhn_domain_correct, total_cifar_domain_correct = 0, 0, 0
        total_samples = 0

        for mnist_data, svhn_data, cifar_data in zip(mnist_loader, svhn_loader, cifar_loader):

            lambda_p = 0.0

            # Training with source data
            mnist_images, mnist_labels = mnist_data
            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            svhn_images, svhn_labels = svhn_data
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            cifar_images, cifar_labels = cifar_data
            cifar_images, cifar_labels = cifar_images.to(device), cifar_labels.to(device)

            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            svhn_dlabels = torch.full((svhn_images.size(0),), 0, dtype=torch.long, device=device)
            cifar_dlabels = torch.full((cifar_images.size(0),), 1, dtype=torch.long, device=device)

            pre_opt.zero_grad()

            mnist_out_partition, mnist_domain_out, mnist_switcher = model.pretrain_fwd(0, mnist_images, alpha=lambda_p)
            svhn_out_partition, svhn_domain_out, svhn_switcher = model.pretrain_fwd(0, svhn_images, alpha=lambda_p)
            cifar_out_partition, cifar_domain_out, cifar_switcher = model.pretrain_fwd(1, cifar_images, alpha=lambda_p)

            # mnist_loss = criterion(mnist_out_partition, mnist_labels)
            # svhn_loss = criterion(svhn_out_partition, svhn_labels)
            # cifar_loss = criterion(cifar_out_partition, cifar_labels)
            # label_loss = mnist_loss + svhn_loss + cifar_loss

            mnist_domain_loss = criterion(mnist_domain_out, mnist_dlabels)
            svhn_domain_loss = criterion(svhn_domain_out, svhn_dlabels)
            cifar_domain_loss = criterion(cifar_domain_out, cifar_dlabels)
            domain_loss = mnist_domain_loss + svhn_domain_loss + cifar_domain_loss

            switcher_mnist_loss = criterion(mnist_switcher, mnist_dlabels)
            switcher_svhn_loss = criterion(svhn_switcher, svhn_dlabels)
            switcher_cifar_loss = criterion(cifar_switcher, cifar_dlabels)
            switcher_loss = switcher_mnist_loss + switcher_svhn_loss + switcher_cifar_loss

            loss = domain_loss + switcher_loss
            loss.backward()

            pre_opt.step()

            # total_mnist_loss += mnist_loss.item()
            # total_svhn_loss += svhn_loss.item()
            # total_cifar_loss += cifar_loss.item()

            # mnist_correct = (torch.argmax(mnist_out_partition, dim=1) == mnist_labels).sum().item()
            # svhn_correct = (torch.argmax(svhn_out_partition, dim=1) == svhn_labels).sum().item()
            # cifar_correct = (torch.argmax(cifar_out_partition, dim=1) == cifar_labels).sum().item()
            # total_mnist_correct += mnist_correct
            # total_svhn_correct += svhn_correct
            # total_cifar_correct += cifar_correct

            mnist_domain_correct = (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
            svhn_domain_correct = (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()
            cifar_domain_correct = (torch.argmax(cifar_domain_out, dim=1) == cifar_dlabels).sum().item()
            total_mnist_domain_correct += mnist_domain_correct
            total_svhn_domain_correct += svhn_domain_correct
            total_cifar_domain_correct += cifar_domain_correct

            total_samples += mnist_labels.size(0)
            i += 1

        # mnist_acc_epoch = (total_mnist_correct / total_samples) * 100
        # svhn_acc_epoch = (total_svhn_correct / total_samples) * 100
        # cifar_acc_epoch = (total_cifar_correct / total_samples) * 100

        # mnist_loss_epoch = total_mnist_loss / total_samples
        # svhn_loss_epoch = total_svhn_loss / total_samples
        # cifar_loss_epoch = total_cifar_loss / total_samples

        # print(f"Pre Epoch {epoch + 1} | "
        #       f"MNIST Acc: {mnist_acc_epoch:.2f}%, Loss: {mnist_loss_epoch:.6f} | "
        #       f"SVHN Acc: {svhn_acc_epoch:.2f}%, Loss: {svhn_loss_epoch:.6f} | "
        #       f"CIFAR Acc: {cifar_acc_epoch:.2f}%, Loss: {cifar_loss_epoch:.6f}")

        mnist_domain_acc_epoch = (total_mnist_domain_correct / total_samples) * 100
        svhn_domain_acc_epoch = (total_svhn_domain_correct / total_samples) * 100
        cifar_domain_acc_epoch = (total_cifar_domain_correct / total_samples) * 100

        print(f"Pre Epoch {epoch + 1} | "
              f"MNIST domain Acc: {mnist_domain_acc_epoch:.2f}% | "
              f"SVHN Acc: {svhn_domain_acc_epoch:.2f}% | "
              f"CIFAR Acc: {cifar_domain_acc_epoch:.2f}%")

    print("Pretraining done")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()

        loss_ani_domain_epoch, loss_num_domain_epoch, loss_label_epoch = 0, 0, 0
        total_mnist_loss, total_svhn_loss, total_cifar_loss, total_label_loss = 0, 0, 0, 0
        total_mnist_correct, total_svhn_correct, total_cifar_correct = 0, 0, 0
        total_mnist_domain_correct, total_svhn_domain_correct, total_cifar_domain_correct = 0, 0, 0
        mnist_partition_counts = torch.zeros(args.num_partition, device=device)
        svhn_partition_counts = torch.zeros(args.num_partition, device=device)
        cifar_partition_counts = torch.zeros(args.num_partition, device=device)
        total_samples = 0

        for i, (mnist_data, svhn_data, cifar_data) in enumerate(zip(mnist_loader, svhn_loader, cifar_loader)):
            #p = (float(i + epoch * min(len(mnist_loader), len(svhn_loader), len(cifar_loader))) /
            #     num_epochs / min(len(mnist_loader), len(svhn_loader), len(cifar_loader)))
            #lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
            p = epoch / num_epochs
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            mnist_images, mnist_labels = mnist_data
            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            svhn_images, svhn_labels = svhn_data
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            cifar_images, cifar_labels = cifar_data
            cifar_images, cifar_labels = cifar_images.to(device), cifar_labels.to(device)

            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            svhn_dlabels = torch.full((svhn_images.size(0),), 0, dtype=torch.long, device=device)
            cifar_dlabels = torch.full((cifar_images.size(0),), 1, dtype=torch.long, device=device)

            mnist_out_part, mnist_domain_out, mnist_part_idx = model(mnist_images, alpha=lambda_p)
            svhn_out_part, svhn_domain_out, svhn_part_idx = model(svhn_images, alpha=lambda_p)
            cifar_out_part, cifar_domain_out, cifar_part_idx = model(cifar_images, alpha=lambda_p)

            # count partition ratio
            mnist_partition_counts += torch.bincount(mnist_part_idx, minlength=args.num_partition).to(device)
            svhn_partition_counts += torch.bincount(svhn_part_idx, minlength=args.num_partition).to(device)
            cifar_partition_counts += torch.bincount(cifar_part_idx, minlength=args.num_partition).to(device)

            mnist_label_loss = criterion(mnist_out_part, mnist_labels)
            svhn_label_loss = criterion(svhn_out_part, svhn_labels)
            cifar_label_loss = criterion(cifar_out_part, cifar_labels)
            label_loss = mnist_label_loss + svhn_label_loss + cifar_label_loss

            mnist_domain_loss = criterion(mnist_domain_out, mnist_dlabels)
            svhn_domain_loss = criterion(svhn_domain_out, svhn_dlabels)
            cifar_domain_loss = criterion(cifar_domain_out, cifar_dlabels)
            domain_num_loss = mnist_domain_loss + svhn_domain_loss
            domain_ani_loss = cifar_domain_loss

            loss_label_epoch += label_loss.item()
            loss_num_domain_epoch += domain_num_loss.item()
            loss_ani_domain_epoch += domain_ani_loss.item()

            loss = domain_num_loss + domain_ani_loss + label_loss
            loss.backward()
            optimizer.step()

            total_mnist_loss += mnist_label_loss.item()
            total_svhn_loss += svhn_label_loss.item()
            total_cifar_loss += cifar_label_loss.item()

            total_mnist_correct += (torch.argmax(mnist_out_part, dim=1) == mnist_labels).sum().item()
            total_svhn_correct += (torch.argmax(svhn_out_part, dim=1) == svhn_labels).sum().item()
            total_cifar_correct += (torch.argmax(cifar_out_part, dim=1) == cifar_labels).sum().item()

            total_mnist_domain_correct += (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
            total_svhn_domain_correct += (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()
            total_cifar_domain_correct += (torch.argmax(cifar_domain_out, dim=1) == cifar_dlabels).sum().item()

            total_samples += mnist_labels.size(0)

            """
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
            """
        scheduler.step()

        mnist_partition_ratios = mnist_partition_counts / total_samples * 100
        svhn_partition_ratios = svhn_partition_counts / total_samples * 100
        cifar_partition_ratios = cifar_partition_counts / total_samples * 100

        mnist_partition_ratio_str = " | ".join(
            [f"Partition {p}: {mnist_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
        svhn_partition_ratio_str = " | ".join(
            [f"Partition {p}: {svhn_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
        cifar_partition_ratio_str = " | ".join(
            [f"Partition {p}: {cifar_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])

        mnist_avg_loss = total_mnist_loss / total_samples
        svhn_avg_loss = total_svhn_loss / total_samples
        cifar_avg_loss = total_cifar_loss / total_samples
        label_avg_loss = (total_mnist_loss + total_svhn_loss + total_cifar_loss) / (total_samples * 3)

        mnist_acc_epoch = total_mnist_correct / total_samples * 100
        svhn_acc_epoch = total_svhn_correct / total_samples * 100
        cifar_acc_epoch = total_cifar_correct / total_samples * 100

        mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples * 100
        svhn_domain_acc_epoch = total_svhn_domain_correct / total_samples * 100
        cifar_domain_acc_epoch = total_cifar_domain_correct / total_samples * 100

        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'MNIST Partition Ratios: {mnist_partition_ratio_str} | '
              f'SVHN Partition Ratios: {svhn_partition_ratio_str} | '
              f'CIFAR Partition Ratios: {cifar_partition_ratio_str} | '
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

        }, step=epoch + 1)

        wandb.log({
            **{f"Train/MNIST Partition {p} Ratio": mnist_partition_ratios[p].item() for p in range(args.num_partition)},
            **{f"Train/SVHN Partition {p} Ratio": svhn_partition_ratios[p].item() for p in range(args.num_partition)},
            **{f"Train/CIFAR Partition {p} Ratio": cifar_partition_ratios[p].item() for p in range(args.num_partition)},
            'Train/Domain Numbers Loss': loss_num_domain_epoch,
            'Train/Domain Animals Loss': loss_ani_domain_epoch,
            'Train/Label Loss': label_avg_loss,
            'Train/MNIST Loss': mnist_avg_loss,
            'Train/SVHN Loss': svhn_avg_loss,
            'Train/CIFAR Loss': cifar_avg_loss,
            'Train/Total Loss': (loss_num_domain_epoch + loss_ani_domain_epoch + label_avg_loss),
            'Train/MNIST Label Accuracy': mnist_acc_epoch,
            'Train/SVHN Label Accuracy': svhn_acc_epoch,
            'Train/CIFAR Label Accuracy': cifar_acc_epoch,
            'Train/MNIST Domain Accuracy': mnist_domain_acc_epoch,
            'Train/SVHN Domain Accuracy': svhn_domain_acc_epoch,
            'Train/CIFAR Domain Accuracy': cifar_domain_acc_epoch,
            'Train/Training Time': end_time - start_time
        }, step=epoch + 1)

        model.eval()

        def tester(loader, group, domain_label):
            label_correct, domain_correct, total = 0, 0, 0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                class_output_partitioned, domain_output, partition_idx = model(images, alpha=0.0)
                label_preds = F.log_softmax(class_output_partitioned, dim=1)

                _, label_predicted = torch.max(label_preds.data, 1)
                _, domain_predicted = torch.max(domain_output.data, 1)
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
