import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
from functions.lr_lambda import lr_lambda
from dataloader.data_loader import data_loader
from model.SimpleCNN import CNN32
from model.Discriminator import Discriminator32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--pretrain_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--source', type=str, default='SVHN')
    parser.add_argument('--target', type=str, default='MNIST')
    parser.add_argument('--lr_cls', type=float, default=0.001)
    parser.add_argument('--lr_dom', type=float, default=0.01)
    args = parser.parse_args()

    pre_epochs = args.pretrain_epoch
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="Efficient Model - MetaLearning & Domain Adaptation",
               entity="hails",
               config=args.__dict__,
               name="[DANN]_S:" + args.source + "_T:" + args.target
                    + "_clr:" + str(args.lr_cls) + "_dlr:" + str(args.lr_dom)
                    + "_Batch:" + str(args.batch_size)
               )

    source_loader, source_loader_test = data_loader(args.source, args.batch_size)
    target_loader, target_loader_test = data_loader(args.target, args.batch_size)

    print("Data load complete, start training")

    model = CNN32(num_classes=10).to(device)
    discriminator = Discriminator32(num_domains=2).to(device)

    pre_opt = optim.Adam(model.parameters(), lr=1e-5)
    optimizer_cls = optim.SGD(list(model.feature_extractor.parameters())
                              + list(model.classifier.parameters()), lr=args.lr_cls, momentum=0.9, weight_decay=1e-6)
    optimizer_dom = optim.SGD(list(model.feature_extractor.parameters())
                              + list(discriminator.discriminator.parameters()), lr=args.lr_dom, momentum=0.9, weight_decay=1e-6)
    scheduler_cls = optim.lr_scheduler.LambdaLR(optimizer_cls, lr_lambda)
    scheduler_dom = optim.lr_scheduler.LambdaLR(optimizer_dom, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(pre_epochs):
        model.train()
        i = 0

        for source_data in source_loader:
            p = (float(i + epoch * len(source_loader)) / num_epochs / len(source_loader))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            # Training with source data
            source_images, source_labels = source_data
            source_images, source_labels = source_images.to(device), source_labels.to(device)

            _, class_output = model(source_images)
            loss = criterion(class_output, source_labels)

            pre_opt.zero_grad()
            loss.backward()
            pre_opt.step()

            label_acc = (torch.argmax(class_output, dim=1) == source_labels).sum().item() / source_labels.size(0)

            print(f'Batches [{i + 1}/{len(source_loader)}], '
                  f'Pretrain Loss: {loss.item():.4f}, '
                  f'Pretrain Accuracy: {label_acc * 100:.3f}%, '
                  )

            i += 1

        model.eval()

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
            source_dlabel = torch.full((source_images.size(0),), 1, dtype=torch.long, device=device)

            target_images, _ = target_data
            target_images = target_images.to(device)
            target_dlabel = torch.full((target_images.size(0),), 0, dtype=torch.long, device=device)

            optimizer_cls.zero_grad()
            optimizer_dom.zero_grad()

            source_feature, source_class_out = model(source_images)
            source_domain_out = discriminator(source_feature, lambda_p=lambda_p)
            target_feature, _ = model(target_images)
            target_domain_out = discriminator(target_feature, lambda_p=lambda_p)

            label_src_loss = criterion(source_class_out, source_labels)
            domain_src_loss = criterion(source_domain_out, source_dlabel)
            domain_tgt_loss = criterion(target_domain_out, target_dlabel)

            loss_label_epoch += label_src_loss.item()
            loss_src_domain_epoch += domain_src_loss.item()
            loss_tgt_domain_epoch += domain_tgt_loss.item()

            loss = label_src_loss + domain_src_loss + domain_tgt_loss

            loss.backward()
            optimizer_cls.step()
            optimizer_dom.step()

            i += 1

        scheduler_cls.step()
        scheduler_dom.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Domain source Loss: {loss_src_domain_epoch:.4f}, '
              f'Domain target Loss: {loss_tgt_domain_epoch:.4f}, '
              f'Label Loss: {loss_label_epoch:.4f}, '
              f'Total Loss: {loss_src_domain_epoch + loss_tgt_domain_epoch  + loss_label_epoch:.4f}, '
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

                _, class_output = model(images)
                preds = F.log_softmax(class_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            wandb.log({group + ' Accuracy': accuracy}, step=epoch + 1)
            print(group + f' Accuracy: {accuracy * 100:.3f}%')

        def dc_tester(loader, group, d_label):
            correct, total = 0, 0
            for images, _ in loader:
                images = images.to(device)

                features, _ = model(images)
                domain_output = discriminator(features, lambda_p=0.0)
                preds = F.log_softmax(domain_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += images.size(0)
                correct += (predicted == d_label).sum().item()

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
