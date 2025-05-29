import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
from functions.lr_lambda import lr_lambda
from functions.GumbelTauScheduler import GumbelTauScheduler
from model.Prunus import Prunus, prunus_weights
from dataloader.data_loader import data_loader
import numpy as np
import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--pretrain_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_partition', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--pre_classifier_out', type=int, default=4096)
    parser.add_argument('--part_layer', type=int, default=4096)
    parser.add_argument('--tau', type=float, default=2.0)

    # tau scheduler
    parser.add_argument('--init_tau', type=float, default=2.0)
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--tau_decay', type=float, default=0.97)

    # Optimizer
    parser.add_argument('--pre_lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--ll_amp', type=int, default=1e9)
    parser.add_argument('--dl_amp', type=int, default=1e9)

    # parameter lr amplifier
    parser.add_argument('--prefc_lr', type=float, default=1.0)
    parser.add_argument('--fc_lr', type=float, default=1.0)
    parser.add_argument('--disc_lr', type=float, default=10.0)
    parser.add_argument('--switcher_lr', type=float, default=10.0)

    # load pretrained model
    # parser.add_argument('--pretrained_model', type=str, default='pretrained_model/Prunus4096_pretrained_epoch_20.pth')
    parser.add_argument('--pretrained_model', type=str, default=None)

    args = parser.parse_args()

    pre_epochs = args.pretrain_epoch
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(entity="hails",
               project="Efficient Model",
               config=args.__dict__,
               name="[Prunus]MSC_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
                    + "_tau" + str(args.tau)
                    + "_PLayer:4096_NoScheduler"
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    cifar_loader1, cifar_loader_test = data_loader('CIFAR10', args.batch_size)
    cifar_loader2, _ = data_loader('CIFAR10', args.batch_size)

    print("Data load complete, start training")

    model = Prunus(num_classes=args.num_classes,
                   pre_classifier_out=args.pre_classifier_out,
                   n_partition=args.num_partition,
                   part_layer=args.part_layer,
                   device=device
                   )

    # tau_scheduler = GumbelTauScheduler(initial_tau=args.init_tau, min_tau=args.min_tau, decay_rate=args.tau_decay)
    param1 = prunus_weights(model, args.pre_lr, args.prefc_lr, args.fc_lr, args.disc_lr, args.switcher_lr)
    pre_opt = optim.SGD(param1, lr=args.pre_lr, momentum=args.momentum, weight_decay=args.opt_decay)
    # param2 = prunus_weights(model, args.pre_lr, args.prefc_lr, args.fc_lr, args.disc_lr, args.switcher_lr)
    optimizer = optim.Adam(model.partition_switcher.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    w_src, w_tgt = 1.0, 2.0
    domain_criterion = nn.CrossEntropyLoss(weight=torch.tensor([w_src, w_tgt], device=device))
    criterion = nn.CrossEntropyLoss()

    if args.pretrained_model is not None:
        print("Loading pretrained model")
        checkpoint = torch.load(args.pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        pre_opt.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Done")
    else:
        for epoch in range(pre_epochs):
            pretrained_model_dir = f"pretrained_model/Prunus4096_pretrained_epoch_{epoch + 1}.pth"
            os.makedirs(os.path.dirname(pretrained_model_dir), exist_ok=True)
            model.train()
            i = 0

            total_mnist_loss, total_svhn_loss, total_cifar_loss, total_label_loss = 0, 0, 0, 0
            total_mnist_correct, total_svhn_correct, total_cifar_correct = 0, 0, 0
            total_mnist_domain_loss, total_svhn_domain_loss, total_cifar_domain_loss, total_domain_loss = 0, 0, 0, 0
            total_mnist_domain_correct, total_svhn_domain_correct, total_cifar_domain_correct = 0, 0, 0
            total_samples = 0

            for mnist_data, svhn_data, cifar_data1, cifar_data2 in zip(mnist_loader, svhn_loader, cifar_loader1, cifar_loader2):
                p = epoch / num_epochs
                lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

                # Training with source data
                mnist_images, mnist_labels = mnist_data
                mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
                svhn_images, svhn_labels = svhn_data
                svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
                cifar_images1, cifar_labels1 = cifar_data1
                cifar_images1, cifar_labels1 = cifar_images1.to(device), cifar_labels1.to(device)
                cifar_images2, cifar_labels2 = cifar_data2
                cifar_images2, cifar_labels2 = cifar_images2.to(device), cifar_labels2.to(device)
                mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
                svhn_dlabels = torch.full((svhn_images.size(0),), 0, dtype=torch.long, device=device)
                cifar_dlabels = torch.full((cifar_images1.size(0),), 1, dtype=torch.long, device=device)

                pre_opt.zero_grad()
                mnist_out_partition, mnist_domain_out, _ = model.pretrain(0, mnist_images, alpha=lambda_p)
                svhn_out_partition, svhn_domain_out, _ = model.pretrain(0, svhn_images, alpha=lambda_p)
                cifar_out_partition1, cifar_domain_out1, _ = model.pretrain(1, cifar_images1, alpha=lambda_p)
                cifar_out_partition2, cifar_domain_out2, _ = model.pretrain(1, cifar_images2, alpha=lambda_p)

                mnist_loss = criterion(mnist_out_partition, mnist_labels)
                svhn_loss = criterion(svhn_out_partition, svhn_labels)
                cifar_loss1 = criterion(cifar_out_partition1, cifar_labels1)
                cifar_loss2 = criterion(cifar_out_partition2, cifar_labels2)
                label_loss = mnist_loss + svhn_loss + cifar_loss1 + cifar_loss2

                mnist_domain_loss = domain_criterion(mnist_domain_out, mnist_dlabels)
                svhn_domain_loss = domain_criterion(svhn_domain_out, svhn_dlabels)
                cifar_domain_loss1 = domain_criterion(cifar_domain_out1, cifar_dlabels)
                cifar_domain_loss2 = domain_criterion(cifar_domain_out2, cifar_dlabels)
                domain_loss = mnist_domain_loss + svhn_domain_loss + cifar_domain_loss1 + cifar_domain_loss2

                loss = label_loss + domain_loss

                loss.backward()
                pre_opt.step()

                total_mnist_loss += mnist_loss.item()
                total_svhn_loss += svhn_loss.item()
                total_cifar_loss += (cifar_loss1.item() + cifar_loss2.item())
                total_label_loss += label_loss.item()

                total_mnist_domain_loss += mnist_domain_loss.item()
                total_svhn_domain_loss += svhn_domain_loss.item()
                total_cifar_domain_loss += (cifar_domain_loss1.item() + cifar_domain_loss2.item())
                total_domain_loss += domain_loss.item()

                total_mnist_correct += (torch.argmax(mnist_out_partition, dim=1) == mnist_labels).sum().item()
                total_svhn_correct += (torch.argmax(svhn_out_partition, dim=1) == svhn_labels).sum().item()
                total_cifar_correct += ((torch.argmax(cifar_out_partition1, dim=1) == cifar_labels1).sum().item()
                                        + (torch.argmax(cifar_out_partition2, dim=1) == cifar_labels2).sum().item())

                total_mnist_domain_correct += (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
                total_svhn_domain_correct += (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()
                total_cifar_domain_correct += ((torch.argmax(cifar_domain_out1, dim=1) == cifar_dlabels).sum().item()
                                               + (torch.argmax(cifar_domain_out2, dim=1) == cifar_dlabels).sum().item())

                total_samples += mnist_labels.size(0)

                i += 1

            mnist_loss_epoch = total_mnist_loss / total_samples
            svhn_loss_epoch = total_svhn_loss / total_samples
            cifar_loss_epoch = total_cifar_loss / (total_samples * 2)
            label_avg_loss = total_label_loss / (total_samples * 4)

            mnist_domain_avg_loss = total_mnist_domain_loss / total_samples
            svhn_domain_avg_loss = total_svhn_domain_loss / total_samples
            cifar_domain_avg_loss = total_cifar_domain_loss / (total_samples * 2)
            domain_avg_loss = total_domain_loss / (total_samples * 4)

            mnist_acc_epoch = (total_mnist_correct / total_samples) * 100
            svhn_acc_epoch = (total_svhn_correct / total_samples) * 100
            cifar_acc_epoch = (total_cifar_correct / (total_samples * 2) ) * 100

            mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples * 100
            svhn_domain_acc_epoch = total_svhn_domain_correct / total_samples * 100
            cifar_domain_acc_epoch = total_cifar_domain_correct / (total_samples * 2) * 100

            print(f"Pre Epoch {epoch + 1} | "
                  f"Label Loss: {label_avg_loss:.6f} | "
                  f"Domain Loss: {domain_avg_loss:.6f}"
                  )
            print(f"Label MNIST Acc: {mnist_acc_epoch:.2f}%, Loss: {mnist_loss_epoch:.6f} | "
                  f"SVHN Acc: {svhn_acc_epoch:.2f}%, Loss: {svhn_loss_epoch:.6f} | "
                  f"CIFAR Acc: {cifar_acc_epoch:.2f}%, Loss: {cifar_loss_epoch:.6f}"
                  )
            print(f"Domain MNIST Acc: {mnist_domain_acc_epoch:.2f}%, Loss: {mnist_domain_avg_loss:.6f} | "
                  f"SVHN Acc: {svhn_domain_acc_epoch:.2f}%, Loss: {svhn_domain_avg_loss:.6f} | "
                  f"CIFAR Acc: {cifar_domain_acc_epoch:.2f}%, Loss: {cifar_domain_avg_loss:.6f}"
                  )

            # save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': pre_opt.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
            }, pretrained_model_dir)
            print(f"Pretrained model saved to {pretrained_model_dir}")
            print("Pretraining done")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        # tau = tau_scheduler.get_tau()

        total_mnist_domain_loss, total_svhn_domain_loss, total_cifar_domain_loss, total_domain_loss = 0, 0, 0, 0
        total_mnist_domain_correct, total_svhn_domain_correct, total_cifar_domain_correct = 0, 0, 0
        total_mnist_loss, total_svhn_loss, total_cifar_loss, total_label_loss = 0, 0, 0, 0
        total_mnist_correct, total_svhn_correct, total_cifar_correct = 0, 0, 0

        mnist_partition_counts = torch.zeros(args.num_partition, device=device)
        svhn_partition_counts = torch.zeros(args.num_partition, device=device)
        cifar_partition_counts = torch.zeros(args.num_partition, device=device)
        total_samples = 0

        for i, (mnist_data, svhn_data, cifar_data1, cifar_data2) in enumerate(zip(mnist_loader, svhn_loader, cifar_loader1, cifar_loader2)):
            p = epoch / num_epochs
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            mnist_images, mnist_labels = mnist_data
            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            svhn_images, svhn_labels = svhn_data
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            cifar_images1, cifar_labels1 = cifar_data1
            cifar_images1, cifar_labels1 = cifar_images1.to(device), cifar_labels1.to(device)
            cifar_images2, cifar_labels2 = cifar_data2
            cifar_images2, cifar_labels2 = cifar_images2.to(device), cifar_labels2.to(device)
            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            svhn_dlabels = torch.full((svhn_images.size(0),), 0, dtype=torch.long, device=device)
            cifar_dlabels = torch.full((cifar_images1.size(0),), 1, dtype=torch.long, device=device)

            optimizer.zero_grad()

            mnist_out_part, mnist_domain_out, mnist_part_idx = model(mnist_images, alpha=lambda_p, tau=args.tau)
            svhn_out_part, svhn_domain_out, svhn_part_idx = model(svhn_images, alpha=lambda_p, tau=args.tau)
            cifar_out_part1, cifar_domain_out1, cifar_part_idx1 = model(cifar_images1, alpha=lambda_p, tau=args.tau)
            cifar_out_part2, cifar_domain_out2, cifar_part_idx2 = model(cifar_images2, alpha=lambda_p, tau=args.tau)
            # mnist_out_part, mnist_domain_out, mnist_part_idx = model(mnist_images, alpha=lambda_p, tau=tau)
            # svhn_out_part, svhn_domain_out, svhn_part_idx = model(svhn_images, alpha=lambda_p, tau=tau)
            # cifar_out_part, cifar_domain_out, cifar_part_idx = model(cifar_images, alpha=lambda_p, tau=tau)


            mnist_label_loss = criterion(mnist_out_part, mnist_labels)
            svhn_label_loss = criterion(svhn_out_part, svhn_labels)
            cifar_label_loss1 = criterion(cifar_out_part1, cifar_labels1)
            cifar_label_loss2 = criterion(cifar_out_part2, cifar_labels2)
            label_loss = mnist_label_loss + svhn_label_loss + cifar_label_loss1 + cifar_label_loss2

            mnist_domain_loss = domain_criterion(mnist_domain_out, mnist_dlabels)
            svhn_domain_loss = domain_criterion(svhn_domain_out, svhn_dlabels)
            cifar_domain_loss1 = domain_criterion(cifar_domain_out1, cifar_dlabels)
            cifar_domain_loss2 = domain_criterion(cifar_domain_out2, cifar_dlabels)
            domain_loss = mnist_domain_loss + svhn_domain_loss + cifar_domain_loss1 + cifar_domain_loss2

            # loss = label_loss + domain_loss
            loss = (label_loss * args.ll_amp) + (domain_loss * args.dl_amp)

            loss.backward()
            entries = []
            for name, param in model.partition_switcher.named_parameters():
                if param.grad is None:
                    grad_str = 'None'
                else:
                    grad_str = f"{torch.mean(torch.abs(param.grad)).item():.4f}"
                data_str = f"{torch.mean(torch.abs(param.data)).item():.4f}"
                entries.append(f"{name}: {grad_str}, {data_str}")

            print(" | ".join(entries) + f" | loss: {loss.item():.4f}")

            optimizer.step()

            # count partition ratio
            mnist_partition_counts += torch.bincount(mnist_part_idx, minlength=args.num_partition).to(device)
            svhn_partition_counts += torch.bincount(svhn_part_idx, minlength=args.num_partition).to(device)
            cifar_partition_counts += ((torch.bincount(cifar_part_idx1, minlength=args.num_partition).to(device)
                                       + torch.bincount(cifar_part_idx2, minlength=args.num_partition).to(device)))

            total_label_loss += label_loss.item()
            total_mnist_loss += mnist_label_loss.item()
            total_svhn_loss += svhn_label_loss.item()
            total_cifar_loss += (cifar_label_loss1.item() + cifar_label_loss2.item())

            total_domain_loss += domain_loss.item()
            total_mnist_domain_loss += mnist_domain_loss.item()
            total_svhn_domain_loss += svhn_domain_loss.item()
            total_cifar_domain_loss += (cifar_domain_loss1.item() + cifar_domain_loss2.item())

            total_mnist_correct += (torch.argmax(mnist_out_part, dim=1) == mnist_labels).sum().item()
            total_svhn_correct += (torch.argmax(svhn_out_part, dim=1) == svhn_labels).sum().item()
            total_cifar_correct += ((torch.argmax(cifar_out_part1, dim=1) == cifar_labels1).sum().item()
                                    + (torch.argmax(cifar_out_part2, dim=1) == cifar_labels2).sum().item())

            total_mnist_domain_correct += (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
            total_svhn_domain_correct += (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()
            total_cifar_domain_correct += ((torch.argmax(cifar_domain_out1, dim=1) == cifar_dlabels).sum().item()
                                           + (torch.argmax(cifar_domain_out2, dim=1) == cifar_dlabels).sum().item())

            total_samples += mnist_labels.size(0)

        # tau_scheduler.step()
        # scheduler.step()

        mnist_partition_ratios = mnist_partition_counts / total_samples * 100
        svhn_partition_ratios = svhn_partition_counts / total_samples * 100
        cifar_partition_ratios = cifar_partition_counts / (total_samples * 2) * 100

        mnist_partition_ratio_str = " | ".join(
            [f"Partition {p}: {mnist_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
        svhn_partition_ratio_str = " | ".join(
            [f"Partition {p}: {svhn_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
        cifar_partition_ratio_str = " | ".join(
            [f"Partition {p}: {cifar_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])

        mnist_domain_avg_loss = total_mnist_domain_loss / total_samples
        svhn_domain_avg_loss = total_svhn_domain_loss / total_samples
        cifar_domain_avg_loss = total_cifar_domain_loss / (total_samples * 2)
        domain_avg_loss = total_domain_loss / (total_samples * 4)

        mnist_avg_loss = total_mnist_loss / total_samples
        svhn_avg_loss = total_svhn_loss / total_samples
        cifar_avg_loss = total_cifar_loss / (total_samples * 2)
        label_avg_loss = total_label_loss / (total_samples * 4)

        mnist_acc_epoch = total_mnist_correct / total_samples * 100
        svhn_acc_epoch = total_svhn_correct / total_samples * 100
        cifar_acc_epoch = total_cifar_correct / (total_samples * 2) * 100

        mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples * 100
        svhn_domain_acc_epoch = total_svhn_domain_correct / total_samples * 100
        cifar_domain_acc_epoch = total_cifar_domain_correct / (total_samples * 2) * 100

        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'MNIST Ratios {mnist_partition_ratio_str} | '
              f'SVHN Ratios {svhn_partition_ratio_str} | '
              f'CIFAR Ratios {cifar_partition_ratio_str} | '
              f'Label Loss: {label_avg_loss:.4f} | '
              f'Domain Loss: {domain_avg_loss:.4f} | '
              f'Total Loss: {label_avg_loss + domain_avg_loss:.4f} | '
              f'Time: {end_time - start_time:.2f} sec | '
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
              f'CIFAR Domain Acc: {cifar_domain_acc_epoch:.3f}% |'
              )

        wandb.log({
            **{f"Train/MNIST Partition {p} Ratio": mnist_partition_ratios[p].item() for p in range(args.num_partition)},
            **{f"Train/SVHN Partition {p} Ratio": svhn_partition_ratios[p].item() for p in range(args.num_partition)},
            **{f"Train/CIFAR Partition {p} Ratio": cifar_partition_ratios[p].item() for p in range(args.num_partition)},
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
            'Train/Training Time': end_time - start_time
        }, step=epoch + 1)

        model.eval()

        def tester(loader, group, domain_label):
            label_correct, domain_correct, total = 0, 0, 0
            partition_counts = torch.zeros(args.num_partition, device=device)
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output_partitioned, domain_output, partition_idx = model.test(images)

                total += images.size(0)
                label_correct += (torch.argmax(class_output_partitioned, dim=1) == labels).sum().item()
                domain_correct += (torch.argmax(domain_output, dim=1) == domain_label).sum().item()
                partition_counts += torch.bincount(partition_idx, minlength=args.num_partition)

            label_acc = label_correct / total * 100
            domain_acc = domain_correct / total * 100
            partition_ratios = partition_counts / total * 100
            partition_ratio_str = " | ".join(
                [f"Partition {p}: {partition_ratios[p]:.2f}%" for p in range(args.num_partition)])

            wandb.log({
                f'Test/Label {group} Accuracy': label_acc,
                f'Test/Domain {group} Accuracy': domain_acc,
                **{f"Test/{group} Partition {p} Ratio": partition_ratios[p].item() for p in range(args.num_partition)}
            }, step=epoch + 1)

            print(f'Test {group} | Label Acc: {label_acc:.3f}% | Domain Acc: {domain_acc:.3f}% | Partition Ratio: {partition_ratio_str}')

        with torch.no_grad():
            tester(mnist_loader_test, 'MNIST', 0)
            tester(svhn_loader_test, 'SVHN', 0)
            tester(cifar_loader_test, 'CIFAR', 1)


if __name__ == '__main__':
    main()
