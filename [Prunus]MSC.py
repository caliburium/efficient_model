import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
# from functions.lr_lambda import lr_lambda
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
    parser.add_argument('--pretrain_epoch', type=int, default=1)
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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--ll_amp', type=float, default=1)
    parser.add_argument('--dl_amp', type=float, default=1)

    # parameter lr amplifier
    parser.add_argument('--prefc_lr', type=float, default=1.0)
    parser.add_argument('--fc_lr', type=float, default=1.0)
    parser.add_argument('--disc_lr', type=float, default=1.0)
    parser.add_argument('--switcher_lr', type=float, default=1.0)

    # regularization
    parser.add_argument('--reg_alpha', type=float, default=0.01)
    parser.add_argument('--reg_beta', type=float, default=0.1)

    # load pretrained model
    # parser.add_argument('--pretrained_model', type=str, default='pretrained_model/Prunus4096_pretrained_epoch_10.pth')
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
                    + "_PLayer:4096"
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    cifar_loader, cifar_loader_test = data_loader('CIFAR10', args.batch_size)

    print("Data load complete, start training")

    model = Prunus(num_classes=args.num_classes,
                   pre_classifier_out=args.pre_classifier_out,
                   n_partition=args.num_partition,
                   part_layer=args.part_layer,
                   device=device
                   )

    # tau_scheduler = GumbelTauScheduler(initial_tau=args.init_tau, min_tau=args.min_tau, decay_rate=args.tau_decay)
    param = prunus_weights(model, args.pre_lr, args.prefc_lr, args.fc_lr, args.disc_lr, args.switcher_lr)
    pre_opt = optim.SGD(param, lr=args.pre_lr, momentum=args.momentum, weight_decay=args.opt_decay)
    optimizer = pre_opt
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
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

            total_mnist_loss, total_mnist_correct, total_mnist_domain_loss, total_mnist_domain_correct = 0, 0, 0, 0
            total_svhn_loss, total_svhn_correct, total_svhn_domain_loss, total_svhn_domain_correct = 0, 0, 0, 0
            total_cifar_loss, total_cifar_correct, total_cifar_domain_loss, total_cifar_domain_correct = 0, 0, 0, 0
            total_label_loss, total_domain_loss, total_samples = 0, 0, 0

            # for mnist_data, svhn_data, cifar_data1, cifar_data2 in zip(mnist_loader, svhn_loader, cifar_loader1, cifar_loader2):
            for mnist_data, svhn_data, cifar_data in zip(mnist_loader, svhn_loader, cifar_loader):
                p = epoch / num_epochs
                lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

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
                mnist_out_partition, mnist_domain_out, _ = model.pretrain(0, mnist_images, alpha=lambda_p)
                svhn_out_partition, svhn_domain_out, _ = model.pretrain(0, svhn_images, alpha=lambda_p)
                cifar_out_partition, cifar_domain_out, _ = model.pretrain(1, cifar_images, alpha=lambda_p)

                mnist_loss = criterion(mnist_out_partition, mnist_labels)
                svhn_loss = criterion(svhn_out_partition, svhn_labels)
                cifar_loss = criterion(cifar_out_partition, cifar_labels)
                label_loss = mnist_loss + svhn_loss + cifar_loss

                mnist_domain_loss = domain_criterion(mnist_domain_out, mnist_dlabels)
                svhn_domain_loss = domain_criterion(svhn_domain_out, svhn_dlabels)
                cifar_domain_loss = domain_criterion(cifar_domain_out, cifar_dlabels)
                domain_loss = mnist_domain_loss + svhn_domain_loss + cifar_domain_loss

                loss = label_loss + domain_loss

                loss.backward()
                pre_opt.step()

                total_mnist_loss += mnist_loss.item()
                total_svhn_loss += svhn_loss.item()
                total_cifar_loss += cifar_loss.item()
                total_label_loss += label_loss.item()

                total_mnist_domain_loss += mnist_domain_loss.item()
                total_svhn_domain_loss += svhn_domain_loss.item()
                total_cifar_domain_loss += cifar_domain_loss.item()
                total_domain_loss += domain_loss.item()

                total_mnist_correct += (torch.argmax(mnist_out_partition, dim=1) == mnist_labels).sum().item()
                total_svhn_correct += (torch.argmax(svhn_out_partition, dim=1) == svhn_labels).sum().item()
                total_cifar_correct += (torch.argmax(cifar_out_partition, dim=1) == cifar_labels).sum().item()

                total_mnist_domain_correct += (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
                total_svhn_domain_correct += (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()
                total_cifar_domain_correct += ((torch.argmax(cifar_domain_out, dim=1) == cifar_dlabels).sum().item())

                total_samples += svhn_labels.size(0)

                i += 1

            mnist_loss_epoch = total_mnist_loss / total_samples
            svhn_loss_epoch = total_svhn_loss / total_samples
            cifar_loss_epoch = total_cifar_loss / total_samples
            label_avg_loss = total_label_loss / (total_samples * 3)

            mnist_domain_avg_loss = total_mnist_domain_loss / total_samples
            svhn_domain_avg_loss = total_svhn_domain_loss / total_samples
            cifar_domain_avg_loss = total_cifar_domain_loss / total_samples
            domain_avg_loss = total_domain_loss / (total_samples * 3)

            mnist_acc_epoch = (total_mnist_correct / total_samples) * 100
            svhn_acc_epoch = (total_svhn_correct / total_samples) * 100
            cifar_acc_epoch = (total_cifar_correct / total_samples) * 100

            mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples * 100
            svhn_domain_acc_epoch = total_svhn_domain_correct / total_samples * 100
            cifar_domain_acc_epoch = total_cifar_domain_correct / total_samples * 100

            print(f"Pre Epoch {epoch + 1} | "
                  f"Label Loss: {label_avg_loss:.6f} | "
                  f"Domain Loss: {domain_avg_loss:.6f}"
                  )
            print(
                  f"Label MNIST Acc: {mnist_acc_epoch:.2f}%, Loss: {mnist_loss_epoch:.6f} | "
                  f"SVHN Acc: {svhn_acc_epoch:.2f}%, Loss: {svhn_loss_epoch:.6f} | "
                  f"CIFAR Acc: {cifar_acc_epoch:.2f}%, Loss: {cifar_loss_epoch:.6f}"
                  )
            print(
                  f"Domain MNIST Acc: {mnist_domain_acc_epoch:.2f}%, Loss: {mnist_domain_avg_loss:.6f} | "
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

        total_mnist_domain_loss, total_mnist_domain_correct, total_mnist_loss, total_mnist_correct = 0, 0, 0, 0
        total_svhn_domain_loss, total_svhn_domain_correct, total_svhn_loss, total_svhn_correct = 0, 0, 0, 0
        total_cifar_domain_loss, total_cifar_domain_correct, total_cifar_loss, total_cifar_correct = 0, 0, 0, 0
        total_domain_loss, total_label_loss = 0, 0
        total_specialization_loss, total_diversity_loss, total_reg_loss = 0, 0, 0

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

            mnist_out_part, mnist_domain_out, mnist_part_idx, mnist_part_gumbel = model(mnist_images, alpha=lambda_p, tau=args.tau, return_partition_prob=True)
            svhn_out_part, svhn_domain_out, svhn_part_idx, svhn_part_gumbel = model(svhn_images, alpha=lambda_p, tau=args.tau, return_partition_prob=True)
            cifar_out_part, cifar_domain_out, cifar_part_idx, cifar_part_gumbel = model(cifar_images, alpha=lambda_p, tau=args.tau, return_partition_prob=True)
            # mnist_out_part, mnist_domain_out, mnist_part_idx, mnist_part_gumbel = model(mnist_images, alpha=lambda_p, tau=tau, return_partition_prob=True)
            # svhn_out_part, svhn_domain_out, svhn_part_idx, svhn_part_gumbel = model(svhn_images, alpha=lambda_p, tau=tau, return_partition_prob=True)
            # cifar_out_part, cifar_domain_out, cifar_part_idx, cifar_part_gumbel = model(cifar_images, alpha=lambda_p, tau=tau, return_partition_prob=True)

            if i % 1 == 0:
                print(f"--- [Epoch {epoch + 1}, Batch {i}] Partition Stats ---")
                mnist_counts = torch.bincount(mnist_part_idx, minlength=args.num_partition)
                svhn_counts = torch.bincount(svhn_part_idx, minlength=args.num_partition)
                cifar_counts = torch.bincount(cifar_part_idx, minlength=args.num_partition)
                print(f"MNIST : {mnist_counts.cpu().numpy()} / SVHN  : {svhn_counts.cpu().numpy()} / CIFAR : {cifar_counts.cpu().numpy()}")
                print(f"Switcher Weight Mean: {model.partition_switcher.weight.data.mean():.4f}, Bias Mean: {model.partition_switcher.bias.data.mean():.4f}")

            mnist_label_loss = criterion(mnist_out_part, mnist_labels)
            svhn_label_loss = criterion(svhn_out_part, svhn_labels)
            cifar_label_loss = criterion(cifar_out_part, cifar_labels)

            # 두 평균값 사이의 거리(차이의 제곱)를 최대화하기 위해 음수를 취함
            # numbers_idx = torch.cat((mnist_part_idx, cifar_part_idx))
            # mean_numbers_idx = torch.mean(numbers_idx.float())
            # mean_cifar_idx = torch.mean(cifar_part_idx.float())
            # separation_loss = -torch.pow(mean_numbers_idx - mean_cifar_idx, 2)

            # dat_cat = torch.cat((numbers_idx, cifar_part_idx))
            # dat_cat = dat_cat.float()

            # reg_numbers = torch.var(numbers_idx.float())
            # reg_cifar = torch.var(cifar_part_idx.float())
            # reg_loss = torch.var(dat_cat)

            # 최종 Loss에 반영
            # 기존 reg_loss 대신 separation_loss 사용

            # 1. 각 데이터셋 내의 평균 확률 분포 계산
            numbers_part_gumbel = torch.cat((mnist_part_gumbel, svhn_part_gumbel))
            avg_prob_numbers = torch.mean(numbers_part_gumbel, dim=0)  # [M] 크기
            # avg_prob_mnist = torch.mean(mnist_part_gumbel, dim=0) # [M] 크기
            # avg_prob_svhn = torch.mean(svhn_part_gumbel, dim=0) # [M] 크기
            avg_prob_cifar = torch.mean(cifar_part_gumbel, dim=0) # [M] 크기

            # 2. 각 데이터셋의 조건부 엔트로피 계산 (작을수록 좋음)
            # log(0)을 피하기 위해 작은 값(epsilon)을 더함
            epsilon = 1e-8
            # loss_specialization_mnist = -torch.sum(avg_prob_mnist * torch.log(avg_prob_mnist + epsilon))
            # loss_specialization_svhn = -torch.sum(avg_prob_svhn * torch.log(avg_prob_svhn + epsilon))
            loss_specialization_numbers = -torch.sum(avg_prob_numbers * torch.log(avg_prob_numbers + epsilon))
            loss_specialization_cifar = -torch.sum(avg_prob_cifar * torch.log(avg_prob_cifar + epsilon))

            # 1. 전체 데이터셋의 평균 확률 분포 계산
            all_probs = torch.cat((numbers_part_gumbel, cifar_part_gumbel), dim=0)
            avg_prob_global = torch.mean(all_probs, dim=0)
            # avg_prob_global = (loss_specialization_mnist + loss_specialization_svhn + avg_prob_cifar) / 3

            # 2. 전체 엔트로피 계산 (클수록 좋음)
            loss_diversity = -torch.sum(avg_prob_global * torch.log(avg_prob_global + epsilon))

            # Loss로 사용하기 위해 음수를 취함 (즉, 엔트로피를 최대화)
            loss_diversity = -loss_diversity

            # 최종 전문성 Loss
            loss_specialization =  loss_specialization_numbers + loss_specialization_cifar
            # loss_specialization = loss_specialization_mnist + loss_specialization_svhn + loss_specialization_cifar

            label_loss = (mnist_label_loss + svhn_label_loss + cifar_label_loss
                          + args.reg_alpha * loss_specialization + args.reg_beta * loss_diversity)

            mnist_domain_loss = domain_criterion(mnist_domain_out, mnist_dlabels)
            svhn_domain_loss = domain_criterion(svhn_domain_out, svhn_dlabels)
            cifar_domain_loss = domain_criterion(cifar_domain_out, cifar_dlabels)
            domain_loss = (mnist_domain_loss + svhn_domain_loss) / 2 + cifar_domain_loss

            # loss = label_loss + domain_loss
            # loss = (label_loss * args.ll_amp) + (domain_loss * args.dl_amp)
            loss = label_loss
            loss.backward()

            entries = []
            for name, param in model.partition_switcher.named_parameters():
                if param.grad is None:
                    grad_str = 'None'
                else:
                    grad_str = f"{torch.mean(torch.abs(param.grad)).item():.4f}"
                data_str = f"{torch.mean(torch.abs(param.data)).item():.4f}"
                entries.append(f"{name}: {grad_str}, {data_str}")

            # print(" | ".join(entries) + f" | loss: {loss.item():.4f} | reg_loss: {reg_loss.item():.4f}")
            print(" | ".join(entries) + f" | loss: {loss.item():.4f} ")

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
            total_diversity_loss += loss_diversity.item()

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
            # total_reg_loss += reg_loss.item()

            total_samples += svhn_labels.size(0)

        # tau_scheduler.step()
        # scheduler.step()

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
            # 'Train/Reg Loss': total_reg_loss / total_samples,
        }, step=epoch + 1)

        model.eval()

        def tester(loader, group, domain_label):
            label_correct, domain_correct, total = 0, 0, 0
            total_reg_loss = 0
            partition_counts = torch.zeros(args.num_partition, device=device)
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output_partitioned, domain_output, partition_idx = model.test(images)

                total += images.size(0)
                label_correct += (torch.argmax(class_output_partitioned, dim=1) == labels).sum().item()
                domain_correct += (torch.argmax(domain_output, dim=1) == domain_label).sum().item()
                partition_counts += torch.bincount(partition_idx, minlength=args.num_partition)
                # reg_loss = torch.var(partition_idx.float())
                # total_reg_loss += reg_loss.item()

            label_acc = label_correct / total * 100
            domain_acc = domain_correct / total * 100
            partition_ratios = partition_counts / total * 100
            partition_ratio_str = " | ".join(
                [f"Partition {p}: {partition_ratios[p]:.2f}%" for p in range(args.num_partition)])

            wandb.log({
                f'Test/Label {group} Accuracy': label_acc,
                f'Test/Domain {group} Accuracy': domain_acc,
                **{f"Test/{group} Partition {p} Ratio": partition_ratios[p].item() for p in range(args.num_partition)},
                # f'Test/Reg Loss {group}': total_reg_loss / total,
            }, step=epoch + 1)

            print(f'Test {group} | Label Acc: {label_acc:.3f}% | Domain Acc: {domain_acc:.3f}% | Partition Ratio: {partition_ratio_str}')

        with torch.no_grad():
            tester(mnist_loader_test, 'MNIST', 0)
            tester(svhn_loader_test, 'SVHN', 0)
            tester(cifar_loader_test, 'CIFAR', 1)


if __name__ == '__main__':
    main()
