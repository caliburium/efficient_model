import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functions.lr_lambda import *
import wandb
import numpy as np
from tqdm import tqdm
from dataloader.data_loader import data_loader
from model.AlexNet import DANN_Alex32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--pre_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="Efficient Model - MetaLearning & Domain Adaptation",
               entity="hails",
               config=args.__dict__,
               name="[DANN]SingleTask_CIFAR10/STL10_PEpoch:" + str(args.pre_epoch)
                    + "_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    SVHN_loader, SVHN_loader_test = data_loader('SVHN', args.batch_size)
    CIFAR10_loader, CIFAR10_loader_test = data_loader('CIFAR10', args.batch_size)
    MNIST_loader, MNIST_loader_test = data_loader('MNIST', args.batch_size)
    STL10_loader, STL10_loader_test = data_loader('STL10', args.batch_size)

    print("Data load complete, start training")

    model = DANN_Alex32(pretrained=True, num_class=10, num_domain=2).to(device)
    pre_opt = optim.SGD(list(model.parameters()), lr=args.lr)
    optimizer = optim.SGD(list(model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.pre_epoch):
        model.train()
        total_loss, loss_svhn, loss_cifar10 = 0, 0, 0
        svhn_correct, cifar10_correct = 0, 0
        svhn_samples, cifar10_samples = 0, 0

        for svhn_data, cifar10_data in zip(SVHN_loader, tqdm(CIFAR10_loader)):
            svhn_images, svhn_labels = svhn_data
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            cifar10_images, cifar10_labels = cifar10_data
            cifar10_images, cifar10_labels = cifar10_images.to(device), cifar10_labels.to(device)

            svhn_output, _ = model(svhn_images, 0.0)
            svhn_loss = criterion(svhn_output, svhn_labels)
            cifar10_output, _ = model(cifar10_images, 0.0)
            cifar10_loss = criterion(cifar10_output, cifar10_labels)
            loss = svhn_loss + cifar10_loss

            pre_opt.zero_grad()
            loss.backward()
            pre_opt.step()

            total_loss += loss.item()
            loss_svhn += svhn_loss.item()
            loss_cifar10 += cifar10_loss.item()

            svhn_preds = torch.argmax(svhn_output, dim=1)
            svhn_correct += (svhn_preds == svhn_labels).sum().item()
            svhn_samples += svhn_labels.size(0)

            cifar10_preds = torch.argmax(cifar10_output, dim=1)
            cifar10_correct += (cifar10_preds == cifar10_labels).sum().item()
            cifar10_samples += cifar10_labels.size(0)

        svhn_accuracy = svhn_correct / svhn_samples
        cifar10_accuracy = cifar10_correct / cifar10_samples

        print(
            f'Epoch [{epoch + 1}/{args.pre_epoch}], SVHN Loss: {loss_svhn:.4f}, SVHN Accuracy: {svhn_accuracy * 100:.3f}%'
            f'CIFAR10 Loss: {loss_cifar10:.4f}, CIFAR10 Accuracy: {cifar10_accuracy * 100:.3f}%')

        model.eval()

    print("Pretrain Finished")

    for epoch in range(num_epochs):
        model.train()
        i = 0

        loss_tgt_domain_epoch = 0
        loss_src_domain_epoch = 0
        loss_label_epoch = 0

        for source_data, target_data in zip(SVHN_loader, tqdm(CIFAR10_loader)):
            p = (float(i + epoch * min(len(SVHN_loader), len(CIFAR10_loader))) /
                 num_epochs / min(len(SVHN_loader), len(CIFAR10_loader)))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            source_images, source_labels = source_data
            source_images, source_labels = source_images.to(device), source_labels.to(device)

            target_images, _ = target_data
            target_images = target_images.to(device)
            source_domain = torch.full((source_images.size(0),), 1, dtype=torch.long).to(device)
            target_domain = torch.full((target_images.size(0),), 0, dtype=torch.long).to(device)

            optimizer.zero_grad()

            source_class_out, source_domain_out = model(source_images, 0.0)
            _, target_domain_out = model(target_images, lambda_p)

            label_src_loss = criterion(source_class_out, source_labels)
            domain_src_loss = criterion(source_domain_out, source_domain)
            domain_tgt_loss = criterion(target_domain_out, target_domain)

            loss_label_epoch += label_src_loss.item()
            loss_src_domain_epoch += domain_src_loss.item()
            loss_tgt_domain_epoch += domain_tgt_loss.item()

            loss = label_src_loss + domain_src_loss + domain_tgt_loss

            loss.backward()
            optimizer.step()

            i += 1

        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Domain source Loss: {loss_src_domain_epoch:.4f}, '
              f'Domain target Loss: {loss_tgt_domain_epoch:.4f}, '
              f'Label Loss: {loss_label_epoch:.4f}, '
              f'Total Loss: {loss_src_domain_epoch + loss_tgt_domain_epoch + loss_label_epoch:.4f}, '
              )

        wandb.log({
            'Domain source Loss': loss_src_domain_epoch,
            'Domain target Loss': loss_tgt_domain_epoch,
            'Label Loss': loss_label_epoch,
            'Total Loss': loss_src_domain_epoch + loss_tgt_domain_epoch + loss_label_epoch,
        })

        model.eval()

        def lc_tester(loader, group):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output, _ = model(images, 0.0)
                preds = F.log_softmax(class_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            wandb.log({group + ' Accuracy': accuracy}, step=epoch + 1)
            print(group + f' Accuracy: {accuracy * 100:.3f}%')

        def dc_tester(loader, group, domain):
            correct, total = 0, 0
            domains = torch.full((len(loader.dataset),), domain, dtype=torch.long).to(device)
            for images, _ in loader:
                images = images.to(device)

                _, domain_output = model(images, 0.0)
                preds = F.log_softmax(domain_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += images.size(0)
                correct += (predicted == domains[:images.size(0)]).sum().item()

            accuracy = correct / total
            wandb.log({'[Domain] ' + group + ' Accuracy': accuracy}, step=epoch + 1)
            print('[Domain] ' + group + f' Accuracy: {accuracy * 100:.3f}%')

        with torch.no_grad():
            lc_tester(source_loader_test, 'CIFAR10')
            lc_tester(target_loader_test, 'STL10')
            dc_tester(source_loader_test, 'CIFAR10', 1)
            dc_tester(target_loader_test, 'STL10', 0)


if __name__ == '__main__':
    main()
