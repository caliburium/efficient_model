import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functions import ReverseLayerF, lr_lambda
from data_loader import data_loader
import numpy as np
import wandb
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.restored = False

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),  # 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),  # 9
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4),  # 1
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, input_data, alpha=1.0):
        input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 128 * 1 * 1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)
        return class_output, domain_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--pretrain_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--source1', type=str, default='SVHN')
    parser.add_argument('--source2', type=str, default='CIFAR10')
    parser.add_argument('--target', type=str, default='MNIST')
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    pre_epochs = args.pretrain_epoch
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="EM_Domain",
               entity="hails",
               config=args.__dict__,
               name="DANN_MultiTarget_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
                    + "_S1:" + args.source1 + "/S2:" + args.source2 + "/T:" + args.target
               )

    source1_loader, source1_loader_test = data_loader(args.source1, args.batch_size)
    source2_loader, source2_loader_test = data_loader(args.source2, args.batch_size)
    target_loader, target_loader_test = data_loader(args.target, args.batch_size)

    print("Data load complete, start training")

    model = DANN().to(device)

    pre_opt = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(pre_epochs):
        model.train()
        i = 0

        for source1_data, source2_data in zip(source1_loader, source2_loader):
            p = (float(i + epoch * min(len(source1_loader), len(source2_loader))) /
                 num_epochs / min(len(source1_loader), len(source2_loader)))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            # Training with source data
            source1_images, source1_labels = source1_data
            source2_images, source2_labels = source2_data
            source1_images, source1_labels = source1_images.to(device), source1_labels.to(device)
            source2_images, source2_labels = source2_images.to(device), source2_labels.to(device)

            class1_output, _ = model(source1_images, alpha=lambda_p)
            source1_loss = criterion(class1_output, source1_labels)
            class2_output, _ = model(source2_images, alpha=lambda_p)
            source2_loss = criterion(class2_output, source2_labels)
            loss = source1_loss + source2_loss

            pre_opt.zero_grad()
            loss.backward()
            pre_opt.step()
            """
            label_acc = (torch.argmax(class_output, dim=1) == source_labels).sum().item() / source_labels.size(0)
            print(f'Batches [{i + 1}/{len(source_loader)}], '
                  f'Pretrain Loss: {loss.item():.4f}, '
                  f'Pretrain Accuracy: {label_acc * 100:.3f}%, '
                  )
            """
            i += 1

        scheduler.step()
        model.eval()

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        i = 0

        loss_tgt_domain_epoch = 0
        loss_src_domain_epoch = 0
        loss_label_epoch = 0

        for source1_data, source2_data, target_data in zip(source1_loader, source2_loader, tqdm(target_loader)):
            p = (float(i + epoch * min(len(source1_loader), len(source2_loader), len(target_loader))) /
                 num_epochs / min(len(source1_loader), len(source2_loader), len(target_loader)))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            # Training with source data
            source1_images, source1_labels = source1_data
            source1_images, source1_labels = source1_images.to(device), source1_labels.to(device)
            source1_dlabel = torch.full((source1_images.size(0),), 1, dtype=torch.long, device=device)
            source2_images, source2_labels = source2_data
            source2_images, source2_labels = source2_images.to(device), source2_labels.to(device)
            source2_dlabel = torch.full((source1_images.size(0),), 1, dtype=torch.long, device=device)

            source1_label_output, source1_domain_output = model(source1_images, alpha=lambda_p)
            source2_label_output, source2_domain_output = model(source2_images, alpha=lambda_p)
            source1_label_loss = criterion(source1_label_output, source1_labels)
            source2_label_loss = criterion(source2_label_output, source2_labels)

            label_loss = source1_label_loss + source2_label_loss
            loss_label_epoch += label_loss.item()

            domain_src1_loss = criterion(source1_domain_output, source1_dlabel)
            domain_src2_loss = criterion(source2_domain_output, source2_dlabel)

            domain_src_loss = domain_src1_loss + domain_src2_loss
            loss_src_domain_epoch += domain_src_loss.item()

            # Training with target data
            target_images, _ = target_data
            target_images = target_images.to(device)
            target_dlabel = torch.full((target_images.size(0),), 0, dtype=torch.long, device=device)

            _, domain_output = model(target_images, alpha=lambda_p)
            domain_tgt_loss = criterion(domain_output, target_dlabel)

            loss = domain_tgt_loss + domain_src_loss + label_loss
            loss_tgt_domain_epoch += domain_tgt_loss.item()

            loss.backward()
            optimizer.step()

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

        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Domain source Loss: {loss_src_domain_epoch:.4f}, '
              f'Domain target Loss: {loss_tgt_domain_epoch:.4f}, '
              f'Label Loss: {loss_label_epoch:.4f}, '
              f'Total Loss: {loss_src_domain_epoch + loss_tgt_domain_epoch  + loss_label_epoch:.4f}, '
              f'Time: {end_time - start_time:.2f} seconds'
              )

        wandb.log({
            'Domain source Loss': loss_src_domain_epoch,
            'Domain target Loss': loss_tgt_domain_epoch,
            'Label Loss': loss_label_epoch,
            'Total Loss': loss_src_domain_epoch + loss_tgt_domain_epoch + loss_label_epoch,
            'Training Time': end_time - start_time
        })

        model.eval()

        def lc_tester(loader, group):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output, _ = model(images, alpha=0.0)
                preds = F.log_softmax(class_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
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
                preds = F.log_softmax(domain_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += images.size(0)
                correct += (predicted == d_label).sum().item()

            accuracy = correct / total
            wandb.log({'[Domain] ' + group + ' Accuracy': accuracy}, step=epoch + 1)
            print('[Domain] ' + group + f' Accuracy: {accuracy * 100:.3f}%')

        with torch.no_grad():
            lc_tester(source1_loader_test, 'Source1')
            lc_tester(source2_loader_test, 'Source2')
            lc_tester(target_loader_test, 'Target')
            dc_tester(source1_loader_test, 'Source1', 1)
            dc_tester(source2_loader_test, 'Source2', 1)
            dc_tester(target_loader_test, 'Target', 0)


if __name__ == '__main__':
    main()
