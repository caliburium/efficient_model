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
    parser.add_argument('--pre_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="Efficient Model - MetaLearning & Domain Adaptation",
               entity="hails",
               config=args.__dict__,
               name="[DANN]SingleTask_SVHN/MNIST_PEpoch:" + str(args.pre_epoch)
                    + "_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    # Photo, Art_Painting, Cartoon, Sketch
    source_loader, source_loader_test = data_loader('SVHN', args.batch_size)
    target_loader, target_loader_test = data_loader('MNIST', args.batch_size)

    print("Data load complete, start training")

    model = DANN_Alex32(pretrained=True, num_class=10, num_domain=2).to(device)
    pre_opt = optim.SGD(list(model.parameters()), lr=0.01)
    optimizer = optim.SGD(list(model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.pre_epoch):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for source_images, source_labels in source_loader:
            source_images, source_labels = source_images.to(device), source_labels.to(device)

            class_output, _ = model(source_images, 0.0)
            loss = criterion(class_output, source_labels)

            pre_opt.zero_grad()
            loss.backward()
            pre_opt.step()

            total_loss += loss.item()
            total_correct += (torch.argmax(class_output, dim=1) == source_labels).sum().item()
            total_samples += source_labels.size(0)

        epoch_accuracy = total_correct / total_samples
        print(
            f'Epoch [{epoch + 1}/{args.pre_epoch}], Pretrain Loss: {total_loss:.4f}, Pretrain Accuracy: {epoch_accuracy * 100:.3f}%')

        model.eval()

    print("Pretrain Finished")

    for epoch in range(num_epochs):
        model.train()
        i = 0

        loss_tgt_domain_epoch = 0
        loss_src_domain_epoch = 0
        loss_label_epoch = 0

        for source_data, target_data in zip(source_loader, tqdm(target_loader)):
            p = (float(i + epoch * min(len(source_loader), len(target_loader))) /
                 num_epochs / min(len(source_loader), len(target_loader)))
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
            lc_tester(source_loader_test, 'SVHN')
            lc_tester(target_loader_test, 'MNIST')
            dc_tester(source_loader_test, 'SVHN', 1)
            dc_tester(target_loader_test, 'MNIST', 0)


if __name__ == '__main__':
    main()
