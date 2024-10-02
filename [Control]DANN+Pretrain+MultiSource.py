import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functions import ReverseLayerF, lr_lambda
from data_loader import data_loader
import numpy as np
import wandb
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.restored = False

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),  # 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 16
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),  # 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 8
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),  # 8
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 3072),
            nn.ReLU(),
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input_data, alpha=1.0, discriminator=1):
        input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(feature.size(0),-1)
        class_output = self.classifier(feature)
        if discriminator == 1 :
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.discriminator(reverse_feature)
        elif discriminator == 0:
            domain_output = 0
        return class_output, domain_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--pretrain_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--source', type=str, default='SVHN')
    parser.add_argument('--target', type=str, default='MNIST')
    parser.add_argument('--pre_lr', type=float, default=0.025)
    parser.add_argument('--lr', type=float, default=0.05)
    args = parser.parse_args()

    pre_epochs = args.pretrain_epoch
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="EM_Domain",
               entity="hails",
               config=args.__dict__,
               name="DANN_target+_S:" + args.source + "_T:" + args.target
               )

    source_loader, source_loader_test = data_loader(args.source, args.batch_size)
    target_loader, target_loader_test = data_loader(args.target, args.batch_size)

    print("Data load complete, start training")

    model = DANN().to(device)

    pre_opt = optim.SGD(model.parameters(), lr=args.pre_lr, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(pre_epochs):
        model.train()
        i = 0
        loss_epoch = 0.0
        source_correct = 0.0
        target_correct = 0.0
        source_label_size = 0
        target_label_size = 0

        for source_data, target_data in zip(source_loader, target_loader):
            p = (float(i + epoch * len(source_loader)) / num_epochs / len(source_loader))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            source_images, source_labels = source_data
            source_images, source_labels = source_images.to(device), source_labels.to(device)
            target_images, target_labels = target_data
            target_images, target_labels = target_images.to(device), target_labels.to(device)

            source_output, _ = model(source_images, alpha=lambda_p, discriminator=1)
            target_output, _ = model(target_images, alpha=lambda_p, discriminator=1)
            source_loss = criterion(source_output, source_labels)
            target_loss = criterion(target_output, target_labels)
            loss = source_loss + target_loss

            pre_opt.zero_grad()
            loss.backward()
            pre_opt.step()

            loss_epoch += loss.item()
            source_correct += (torch.argmax(source_output, dim=1) == source_labels).sum().item()
            target_correct += (torch.argmax(target_output, dim=1) == target_labels).sum().item()
            source_label_size += source_labels.size(0)
            target_label_size += target_labels.size(0)

            i += 1

        print(f'epoch [{epoch}/{pre_epochs}], '
              f'Pretrain Loss: {loss_epoch:.4f}, '
              f'Pretrain Source Accuracy: {source_correct / source_label_size * 100:.3f}%, '
              f'Pretrain Source Accuracy: {target_correct / target_label_size * 100:.3f}% '
              )

        scheduler.step()
        model.eval()

    for epoch in range(num_epochs):
        model.train()
        i = 0
        total_loss_epoch = 0
        source_label_loss_epoch = 0
        source_domain_loss_epoch = 0
        target_label_loss_epoch = 0
        target_domain_loss_epoch = 0

        for source_data, target_data in zip(source_loader, target_loader):
            p = (float(i + epoch * min(len(source_loader), len(target_loader))) /
                 num_epochs / min(len(source_loader), len(target_loader)))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            source_images, source_labels = source_data
            source_images, source_labels = source_images.to(device), source_labels.to(device)
            target_images, target_labels = target_data
            target_images, target_labels = target_images.to(device), target_labels.to(device)

            source_dlabel = torch.full((source_images.size(0),), 1, dtype=torch.long, device=device)
            target_dlabel = torch.full((target_images.size(0),), 0, dtype=torch.long, device=device)

            optimizer.zero_grad()
            source_class_output, source_domain_output = model(source_images, alpha=lambda_p)
            target_class_output, target_domain_output = model(target_images, alpha=lambda_p)

            source_label_loss = criterion(source_class_output, source_labels)
            target_label_loss = criterion(target_class_output, source_labels)
            source_domain_loss = criterion(source_domain_output, source_dlabel)
            target_domain_loss = criterion(target_domain_output, target_dlabel)

            loss = target_domain_loss + target_label_loss + source_domain_loss + source_label_loss
            loss.backward()
            optimizer.step()

            source_label_loss_epoch += source_label_loss.item()
            target_label_loss_epoch += source_label_loss.item()
            source_domain_loss_epoch += source_domain_loss.item()
            target_domain_loss_epoch += target_domain_loss.item()

            """
            label_acc = (torch.argmax(class_output, dim=1) == source_labels).sum().item() / source_labels.size(0)
            domain_acc = (torch.argmax(domain_output, dim=1) == target_dlabel).sum().item() / target_dlabel.size(0)

            print(f'Batches [{i + 1}/{min(len(source_loader), len(target_loader))}], '
                  f'Domain source Loss: {domain_src_loss.item():.4f}, '
                  f'Domain target Loss: {domain_tgt_loss.item():.4f}, '
                  f'Label Loss: {label_loss.item():.4f}, '
                  f'Label Accuracy: {label_acc * 100:.3f}%, '
                  f'Domain Accuracy: {domain_acc * 100:.3f}%')
            """
            i += 1

        scheduler.step()

        total_loss_epoch = (source_domain_loss_epoch + target_domain_loss_epoch
                            + source_label_loss_epoch + target_label_loss_epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Domain source Loss: {source_domain_loss_epoch:.4f}, '
              f'Domain target Loss: {target_domain_loss_epoch:.4f}, '
              f'Source Label Loss: {source_label_loss_epoch:.4f}, '
              f'Target Label Loss: {target_label_loss_epoch:.4f}, '
              f'Total Loss: {total_loss_epoch:.4f}, '
              )

        wandb.log({
            'Domain source Loss': source_domain_loss_epoch,
            'Domain target Loss': target_domain_loss_epoch,
            'Source Label Loss': source_label_loss_epoch,
            'Target Label Loss': target_label_loss_epoch,
            'Total Loss': total_loss_epoch
        })

        model.eval()

        def lc_tester(loader, group):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output, _ = model(images, alpha=0.0)
                _, predicted = torch.max(class_output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            wandb.log({'[Label] ' + group + ' Accuracy': accuracy}, step=epoch + 1)
            print('[Label] ' + group + f' Accuracy: {accuracy * 100:.3f}%')

        def dc_tester(loader, group, d_label):
            correct, total = 0, 0
            for images, _ in loader:
                images = images.to(device)

                _, domain_output = model(images, alpha=0.0)
                _, predicted = torch.max(domain_output, 1)
                total += images.size(0)
                correct += (predicted == d_label).sum().item()

            accuracy = correct / total
            wandb.log({'[Domain] ' + group + ' Accuracy': accuracy}, step=epoch + 1)
            print('[Domain] ' + group + f' Accuracy: {accuracy * 100:.3f}%')

        with torch.no_grad():
            lc_tester(source_loader_test, 'Source')
            lc_tester(target_loader_test, 'Target')
            dc_tester(source_loader_test, 'Source', 1)
            dc_tester(target_loader_test, 'Target', 0)


if __name__ == '__main__':
    main()
