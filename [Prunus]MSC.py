import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from functions.GumbelTauScheduler import GumbelTauScheduler
from model.Prunus import Prunus, prunus_weights
from dataloader.data_loader import data_loader
import numpy as np
import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_partition', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_domains', type=int, default=2)
    parser.add_argument('--pre_classifier_out', type=int, default=4096)
    parser.add_argument('--part_layer', type=int, default=4096)

    # tau scheduler
    parser.add_argument('--init_tau', type=float, default=2.0)
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--tau_decay', type=float, default=0.98)

    # Optimizer
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    parser.add_argument('--lr_alpha', type=float, default=0.1)
    parser.add_argument('--lr_beta', type=float, default=0.25)

    # parameter lr amplifier
    parser.add_argument('--prefc_lr', type=float, default=1.0)
    parser.add_argument('--disc_lr', type=float, default=1.0)
    parser.add_argument('--fc_lr', type=float, default=1.0)
    parser.add_argument('--switcher_lr', type=float, default=2.0)

    # regularization
    parser.add_argument('--reg_alpha', type=float, default=0.5)
    parser.add_argument('--reg_beta', type=float, default=5.0)

    args = parser.parse_args()
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(entity="hails",
               project="Efficient Model - Partition",
               config=args.__dict__,
               name="[Prunus]MSC_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
                    + "_PLayer:" + str(args.part_layer)
                    + "_spe:" + str(args.reg_alpha)
                    + "_div:" + str(args.reg_beta)
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    cifar_loader, cifar_loader_test = data_loader('CIFAR10', args.batch_size)

    print("Data load complete, start training")

    model = Prunus(num_classes=args.num_classes,
                   pre_classifier_out=args.pre_classifier_out,
                   n_partition=args.num_partition,
                   part_layer=args.part_layer,
                   num_domains=args.num_domains,
                   device=device
                   )

    def lr_lambda(progress):
        alpha = args.lr_alpha
        beta = args.lr_beta
        return (1 + alpha * progress) ** (-beta)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(prunus_weights(
        model,
        lr=args.lr,
        pre_weight=args.prefc_lr,
        disc_weight=args.disc_lr,
        fc_weight=args.fc_lr,
        switcher_weight=args.switcher_lr
    ), lr=args.lr)

    tau_scheduler = GumbelTauScheduler(initial_tau=args.init_tau, min_tau=args.min_tau, decay_rate=args.tau_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # w_src, w_tgt = 2.0, 1.0
    # domain_criterion = nn.CrossEntropyLoss(weight=torch.tensor([w_src, w_tgt], device=device))
    domain_criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()

        tau = tau_scheduler.get_tau()

        total_mnist_domain_loss, total_mnist_domain_correct, total_mnist_loss, total_mnist_correct = 0, 0, 0, 0
        total_svhn_domain_loss, total_svhn_domain_correct, total_svhn_loss, total_svhn_correct = 0, 0, 0, 0
        total_cifar_domain_loss, total_cifar_domain_correct, total_cifar_loss, total_cifar_correct = 0, 0, 0, 0
        total_domain_loss, total_label_loss = 0, 0
        total_specialization_loss, total_diversity_loss = 0, 0

        mnist_partition_counts = torch.zeros(args.num_partition, device=device)
        svhn_partition_counts = torch.zeros(args.num_partition, device=device)
        cifar_partition_counts = torch.zeros(args.num_partition, device=device)
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

            mnist_out_part, mnist_domain_out, mnist_part_idx, mnist_part_gumbel = model(mnist_images, alpha=lambda_p, tau=tau, inference=False)
            svhn_out_part, svhn_domain_out, svhn_part_idx, svhn_part_gumbel = model(svhn_images, alpha=lambda_p, tau=tau, inference=False)
            cifar_out_part, cifar_domain_out, cifar_part_idx, cifar_part_gumbel = model(cifar_images, alpha=lambda_p, tau=tau, inference=False)

            if i % 1 == 0:
                print(f"--- [Epoch {epoch + 1}, Batch {i}] Partition Stats ---")
                mnist_counts = torch.bincount(mnist_part_idx, minlength=args.num_partition)
                svhn_counts = torch.bincount(svhn_part_idx, minlength=args.num_partition)
                cifar_counts = torch.bincount(cifar_part_idx, minlength=args.num_partition)
                print(f"MNIST : {mnist_counts.cpu().numpy()} / SVHN  : {svhn_counts.cpu().numpy()} / CIFAR : {cifar_counts.cpu().numpy()}")
                print(f"Switcher Weight Mean: {model.partition_switcher.weight.data.mean():.8f}, Bias Mean: {model.partition_switcher.bias.data.mean():.8f}")

            mnist_label_loss = criterion(mnist_out_part, mnist_labels)
            svhn_label_loss = criterion(svhn_out_part, svhn_labels)
            cifar_label_loss = criterion(cifar_out_part, cifar_labels)

            numbers_part_gumbel = torch.cat((mnist_part_gumbel, svhn_part_gumbel))
            avg_prob_numbers = torch.mean(numbers_part_gumbel, dim=0)
            avg_prob_cifar = torch.mean(cifar_part_gumbel, dim=0)

            loss_specialization_numbers = -torch.sum(avg_prob_numbers * torch.log(avg_prob_numbers + 1e-8))
            loss_specialization_cifar = -torch.sum(avg_prob_cifar * torch.log(avg_prob_cifar + 1e-8))
            loss_specialization = loss_specialization_numbers + loss_specialization_cifar

            all_probs = torch.cat((numbers_part_gumbel, cifar_part_gumbel), dim=0)
            avg_prob_global = torch.mean(all_probs, dim=0)

            loss_diversity = torch.sum(avg_prob_global * torch.log(avg_prob_global + 1e-8))


            label_loss = (mnist_label_loss + svhn_label_loss + cifar_label_loss
                          + args.reg_alpha * loss_specialization + args.reg_beta * loss_diversity)

            mnist_domain_loss = domain_criterion(mnist_domain_out, mnist_dlabels)
            svhn_domain_loss = domain_criterion(svhn_domain_out, svhn_dlabels)
            cifar_domain_loss = domain_criterion(cifar_domain_out, cifar_dlabels)
            domain_loss = (mnist_domain_loss + svhn_domain_loss) / 2 + cifar_domain_loss

            loss = label_loss + domain_loss

            loss.backward()

            entries = []
            for name, param in model.partition_switcher.named_parameters():
                if param.grad is None:
                    grad_str = 'None'
                else:
                    grad_str = f"{torch.mean(torch.abs(param.grad)).item():.6f}"
                data_str = f"{torch.mean(torch.abs(param.data)).item():.6f}"
                entries.append(f"{name}: {grad_str}, {data_str}")

            print(" | ".join(entries) + f" | loss: {loss.item():.6f} ")

            optimizer.step()

            # count partition ratio
            mnist_partition_counts += torch.bincount(mnist_part_idx, minlength=args.num_partition).to(device)
            svhn_partition_counts += torch.bincount(svhn_part_idx, minlength=args.num_partition).to(device)
            cifar_partition_counts += torch.bincount(cifar_part_idx, minlength=args.num_partition).to(device)

            total_label_loss += label_loss.item()
            total_mnist_loss += mnist_label_loss.item()
            total_svhn_loss += svhn_label_loss.item()
            total_cifar_loss += cifar_label_loss.item()

            total_specialization_loss += loss_specialization.item()
            total_diversity_loss += (-loss_diversity.item())

            total_domain_loss += domain_loss.item()
            total_mnist_domain_loss += mnist_domain_loss.item()
            total_svhn_domain_loss += svhn_domain_loss.item()
            total_cifar_domain_loss += cifar_domain_loss.item()

            total_mnist_correct += (torch.argmax(mnist_out_part, dim=1) == mnist_labels).sum().item()
            total_svhn_correct += (torch.argmax(svhn_out_part, dim=1) == svhn_labels).sum().item()
            total_cifar_correct += ((torch.argmax(cifar_out_part, dim=1) == cifar_labels).sum().item())

            total_mnist_domain_correct += (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
            total_svhn_domain_correct += (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()
            total_cifar_domain_correct += ((torch.argmax(cifar_domain_out, dim=1) == cifar_dlabels).sum().item())

            total_samples += mnist_labels.size(0)

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

        mnist_domain_avg_loss = total_mnist_domain_loss / total_samples
        svhn_domain_avg_loss = total_svhn_domain_loss / total_samples
        cifar_domain_avg_loss = total_cifar_domain_loss / total_samples
        domain_avg_loss = total_domain_loss / (total_samples * 3)

        mnist_avg_loss = total_mnist_loss / total_samples
        svhn_avg_loss = total_svhn_loss / total_samples
        cifar_avg_loss = total_cifar_loss / total_samples
        label_avg_loss = total_label_loss / (total_samples * 3)

        specialization_loss = total_specialization_loss / (total_samples * 3)
        diversity_loss = total_diversity_loss / (total_samples * 3)

        mnist_acc_epoch = total_mnist_correct / total_samples * 100
        svhn_acc_epoch = total_svhn_correct / total_samples * 100
        cifar_acc_epoch = total_cifar_correct / total_samples * 100

        mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples * 100
        svhn_domain_acc_epoch = total_svhn_domain_correct / total_samples * 100
        cifar_domain_acc_epoch = total_cifar_domain_correct / total_samples * 100

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
        print(f'Specialization Loss {specialization_loss} | '
              f'Diversity Loss {diversity_loss} | '
              f'Current Tau {tau} | '
        )
        print(
              f'MNIST Loss: {mnist_avg_loss:.4f} | '
              f'SVHN Loss: {svhn_avg_loss:.4f} | '
              f'CIFAR Loss: {cifar_avg_loss:.4f} | '
              f'MNIST Domain Loss: {mnist_domain_avg_loss:.4f} | '
              f'SVHN Domain Loss: {svhn_domain_avg_loss:.4f} | '
              f'CIFAR Domain Loss: {cifar_domain_avg_loss:.4f} | '
        )
        print(
              f'MNIST Acc: {mnist_acc_epoch:.3f}% | '
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
            'Train/Specialization Loss': specialization_loss,
            'Train/Diversity Loss' : diversity_loss,
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
            'Train/Training Time': end_time - start_time,
            'Train/Tau': tau,
            'Train/Learning Rate': optimizer.param_groups[0]['lr']
        }, step=epoch + 1)

        model.eval()

        def evaluate(loader, group, domain_label_int, criterion, domain_criterion, tau):
            label_correct, domain_correct, total = 0, 0, 0
            total_label_loss, total_domain_loss = 0.0, 0.0
            total_specialization_loss, total_diversity_loss = 0.0, 0.0
            partition_counts = torch.zeros(args.num_partition, device=device)
            label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)

            all_part_gumbel = []

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                dlabels = torch.full_like(labels, fill_value=domain_label_int)

                class_output_partitioned, domain_output, partition_idx, part_gumbel = model(images, alpha=0, tau=tau, inference=True)

                all_part_gumbel.append(part_gumbel)

                label_loss = criterion(class_output_partitioned, labels)
                domain_loss = domain_criterion(domain_output, dlabels)
                total_label_loss += label_loss.item() * images.size(0)
                total_domain_loss += domain_loss.item() * images.size(0)

                total += images.size(0)
                label_correct += (torch.argmax(class_output_partitioned, dim=1) == labels).sum().item()
                domain_correct += (torch.argmax(domain_output, dim=1) == dlabels).sum().item()
                partition_counts += torch.bincount(partition_idx, minlength=args.num_partition)

                for i in range(labels.size(0)):
                    label_idx = labels[i].item()
                    part_idx = partition_idx[i].item()
                    label_partition_counts[label_idx, part_idx] += 1

            if all_part_gumbel:
                all_probs = torch.cat(all_part_gumbel, dim=0)

                avg_prob = torch.mean(all_probs, dim=0)
                loss_specialization = -torch.sum(avg_prob * torch.log(avg_prob + 1e-8))

                total_specialization_loss = loss_specialization.item()

            avg_label_loss = total_label_loss / total
            avg_domain_loss = total_domain_loss / total
            label_acc = label_correct / total * 100
            domain_acc = domain_correct / total * 100
            partition_ratios = partition_counts / total * 100
            partition_ratio_str = " | ".join(
                [f"P{p}: {r:.2f}%" for p, r in enumerate(partition_ratios)])

            log_data = {
                f'Val/Label {group} Accuracy': label_acc,
                f'Val/Domain {group} Accuracy': domain_acc,
                f'Val/Label {group} Loss': avg_label_loss,
                f'Val/Domain {group} Loss': avg_domain_loss,
                **{f"Test/{group} Partition {p} Ratio": r.item() for p, r in enumerate(partition_ratios)},
                f'Val/{group} Specialization Loss': total_specialization_loss,
            }

            partition_log_data = {}
            total_counts_per_label = label_partition_counts.sum(dim=1)

            for label_idx in range(args.num_classes):
                total_for_label = total_counts_per_label[label_idx].item()
                for part_idx in range(args.num_partition):
                    count = label_partition_counts[label_idx, part_idx].item()

                    if total_for_label > 0:
                        percentage = (count / total_for_label) * 100
                    else:
                        percentage = 0.0

                    log_key = f"Partition {group}/Partition:{part_idx}/Label:{label_idx}"
                    partition_log_data[log_key] = percentage

            wandb.log({**log_data, **partition_log_data}, step=epoch + 1)

            print(f'Test {group} | Label Acc: {label_acc:.3f}% | Domain Acc: {domain_acc:.3f}% | '
                  f'Label Loss: {avg_label_loss:.4f} | Domain Loss: {avg_domain_loss:.4f} | '
                  f'Specialization Loss: {total_specialization_loss:.4f} | '  # Print the new loss
                  f'Partition Ratio: {partition_ratio_str} | '
                  )

        def tester(loader, group, domain_label_int, criterion, domain_criterion):
            label_correct, domain_correct, total = 0, 0, 0
            total_label_loss, total_domain_loss = 0.0, 0.0
            partition_counts = torch.zeros(args.num_partition, device=device)
            label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                dlabels = torch.full_like(labels, fill_value=domain_label_int)

                class_output_partitioned, domain_output, partition_idx, _ = model(images, alpha=0, inference=True)

                # Loss 계산
                label_loss = criterion(class_output_partitioned, labels)
                domain_loss = domain_criterion(domain_output, dlabels)
                total_label_loss += label_loss.item() * images.size(0)
                total_domain_loss += domain_loss.item() * images.size(0)

                total += images.size(0)
                label_correct += (torch.argmax(class_output_partitioned, dim=1) == labels).sum().item()
                domain_correct += (torch.argmax(domain_output, dim=1) == dlabels).sum().item()
                partition_counts += torch.bincount(partition_idx, minlength=args.num_partition)

                for i in range(labels.size(0)):
                    label_idx = labels[i].item()
                    part_idx = partition_idx[i].item()
                    label_partition_counts[label_idx, part_idx] += 1

            avg_label_loss = total_label_loss / total
            avg_domain_loss = total_domain_loss / total
            label_acc = label_correct / total * 100
            domain_acc = domain_correct / total * 100
            partition_ratios = partition_counts / total * 100
            partition_ratio_str = " | ".join(
                [f"P{p}: {r:.2f}%" for p, r in enumerate(partition_ratios)])

            log_data = {
                f'Test/Label {group} Accuracy': label_acc,
                f'Test/Domain {group} Accuracy': domain_acc,
                f'Test/Label {group} Loss': avg_label_loss,
                f'Test/Domain {group} Loss': avg_domain_loss,
                **{f"Test/{group} Partition {p} Ratio": r.item() for p, r in enumerate(partition_ratios)},
            }

            partition_log_data = {}
            total_counts_per_label = label_partition_counts.sum(dim=1)

            for label_idx in range(args.num_classes):
                total_for_label = total_counts_per_label[label_idx].item()
                for part_idx in range(args.num_partition):
                    count = label_partition_counts[label_idx, part_idx].item()

                    if total_for_label > 0:
                        percentage = (count / total_for_label) * 100
                    else:
                        percentage = 0.0

                    log_key = f"Partition {group}/Partition:{part_idx}/Label:{label_idx}"
                    partition_log_data[log_key] = percentage

            wandb.log({**log_data, **partition_log_data}, step=epoch + 1)

            print(f'Test {group} | Label Acc: {label_acc:.3f}% | Domain Acc: {domain_acc:.3f}% | '
                  f'Label Loss: {avg_label_loss:.4f} | Domain Loss: {avg_domain_loss:.4f} | '
                  f'Partition Ratio: {partition_ratio_str} | '
                  )

        with torch.no_grad():
            # evaluate(mnist_loader, 'MNIST', 0, criterion, domain_criterion, tau)
            # evaluate(svhn_loader, 'SVHN', 0, criterion, domain_criterion, tau)
            # evaluate(cifar_loader, 'CIFAR', 1, criterion, domain_criterion, tau)
            tester(mnist_loader_test, 'MNIST', 0, criterion, domain_criterion)
            tester(svhn_loader_test, 'SVHN', 0, criterion, domain_criterion)
            tester(cifar_loader_test, 'CIFAR', 1, criterion, domain_criterion)

        tau_scheduler.step()

if __name__ == '__main__':
    main()
