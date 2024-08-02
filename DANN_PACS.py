import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functions.ReverseLayerF import *
from functions.lr_lambda import *
import numpy as np
import wandb
import time
from tqdm import tqdm
from dataloader.pacs_loader import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.restored = False

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 5, 3),  # 228 -> 75
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # 75 -> 37
            nn.Conv2d(64, 64, 5, 3),  # 37 -> 11
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # 11 -> 5
            nn.Conv2d(64, 128, 5),  # 5 -> 1
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
            nn.Linear(256, 4)
        )

    def forward(self, input_data, reverse=True, alpha=1.0):
        input_data = input_data.expand(input_data.data.shape[0], 3, 228, 228)
        feature = self.feature(input_data)
        feature = feature.view(-1, 128 * 1 * 1)
        class_output = self.classifier(feature)
        if reverse:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.discriminator(reverse_feature)
        else:
            domain_output = self.discriminator(feature)

        return class_output, domain_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--pretrain_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    pre_epochs = args.pretrain_epoch
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="EM_Domain",
               entity="hails",
               config=args.__dict__,
               name="DANN_PACS_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    _, photo_loader_test = pacs_loader('photo', args.batch_size)
    _, art_loader_test = pacs_loader('art_painting', args.batch_size)
    _, cartoon_loader_test = pacs_loader('cartoon', args.batch_size)
    _, sketch_loader_test = pacs_loader('sketch', args.batch_size)
    train_loader, _ = pacs_loader('all', args.batch_size)

    print("Data load complete, start training")

    model = DANN().to(device)

    pre_opt = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(pre_epochs):
        model.train()
        i = 0

        for train_data in train_loader:
            p = (float(i + epoch * len(train_data)) / num_epochs / len(train_data))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            images, labels, domain = train_data
            images, labels, domain = images.to(device), labels.to(device), domain.to(device)

            class_output, domain_output = model(images, False, alpha=lambda_p)
            loss_l = criterion(class_output, labels)
            loss_d = criterion(domain_output, domain)

            loss = loss_l + loss_d

            pre_opt.zero_grad()
            loss.backward()
            pre_opt.step()

            i += 1

        scheduler.step()
        model.eval()

    for epoch in range(num_epochs):
        model.train()
        i = 0

        loss_domain_epoch = 0
        loss_label_epoch = 0

        for train_data in tqdm(train_loader):
            p = (float(i + epoch * len(train_data)) / num_epochs / len(train_data))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            images, labels, domain = train_data
            images, labels, domain = images.to(device), labels.to(device), domain.to(device)

            class_output, domain_output = model(images, True, alpha=lambda_p)
            loss_l = criterion(class_output, labels)
            loss_d = criterion(domain_output, domain)
            loss = loss_l + loss_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_label_epoch += loss_l.item()
            loss_domain_epoch += loss_d.item()

            i += 1

        scheduler.step()

        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Label Loss: {loss_label_epoch:.4f}, '
              f'Domain Loss: {loss_domain_epoch:.4f}, '
              f'Total Loss: {loss_label_epoch + loss_domain_epoch:.4f}, '
              )

        wandb.log({
            'Label Loss': loss_label_epoch,
            'Domain Loss': loss_domain_epoch,
            'Total Loss': loss_label_epoch + loss_domain_epoch,
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
            lc_tester(photo_loader_test, 'Photo')
            lc_tester(art_loader_test, 'Art')
            lc_tester(cartoon_loader_test, 'Cartoon')
            lc_tester(sketch_loader_test, 'Sketch')
            dc_tester(photo_loader_test, 'Photo', 0)
            dc_tester(art_loader_test, 'Art', 1)
            dc_tester(cartoon_loader_test, 'Cartoon', 2)
            dc_tester(sketch_loader_test, 'Sketch', 3)


if __name__ == '__main__':
    main()
