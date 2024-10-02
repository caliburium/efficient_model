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
# device = 'cpu'


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1)
        self.BatchNorm2d_64 = nn.BatchNorm2d(64)
        self.BatchNorm2d_128 = nn.BatchNorm2d(128)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.MaxPool2d(self.relu(self.BatchNorm2d_64(self.conv1(x))))
        x = self.MaxPool2d(self.relu(self.BatchNorm2d_64(self.conv2(x))))
        x = self.relu(self.BatchNorm2d_128(self.conv3(x)))
        return x


class LabelClassifier(nn.Module):
    def __init__(self):
        super(LabelClassifier, self).__init__()
        self.fc1 = nn.Linear(128 * 8 * 8, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn1 = nn.BatchNorm1d(3072)
        self.bn2 = nn.BatchNorm1d(2048)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = torch.softmax(self.fc3(x), dim=1)

        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, lambda_p):
        x = x.view(-1, 128 * 8 * 8)
        x = ReverseLayerF.apply(x, lambda_p)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


def main():
    # MNIST, SVHN, CIFAR10, STL10
    args = argparse.ArgumentParser()
    args.add_argument('--epoch', type=int, default=1000)
    args.add_argument('--pre_epoch', type=int, default=30)
    args.add_argument('--batch_size', type=int, default=100)
    args.add_argument('--source', type=str, default='SVHN')
    args.add_argument('--target', type=str, default='MNIST')
    args.add_argument('--lr_pre', type=float, default=0.05)
    args.add_argument('--lr_train', type=float, default=0.05)

    args = args.parse_args()
    pre_epochs = args.pre_epoch
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="EM_Domain",
               entity="hails",
               config=args.__dict__,
               name="DANN(S)_S:" + args.source + "_T:" + args.target + "_Batch:" + str(args.batch_size)
               )

    source_loader, source_loader_test = data_loader(args.source, args.batch_size)
    target_loader, target_loader_test = data_loader(args.target, args.batch_size)

    print("data load complete, start training")

    feature_extractor = FeatureExtractor().to(device)
    domain_classifier = DomainClassifier().to(device)
    label_classifier = LabelClassifier().to(device)

    optimizer_pre = optim.SGD(list(feature_extractor.parameters()) +
                               list(label_classifier.parameters()), lr=args.lr_pre)
    optimizer_train = optim.SGD(list(feature_extractor.parameters()) +
                                list(domain_classifier.parameters()) +
                                list(label_classifier.parameters()), lr=args.lr_train, momentum=0.9)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer_train, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    # pre train
    for epoch in range(pre_epochs):
        feature_extractor.train()
        label_classifier.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for source_data in tqdm(source_loader):
            source_images, source_labels = source_data
            source_images, source_labels = source_images.to(device), source_labels.to(device)

            features = feature_extractor(source_images)
            preds = label_classifier(features)
            loss = criterion(preds, source_labels)

            optimizer_pre.zero_grad()
            loss.backward()
            optimizer_pre.step()

            running_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            total += source_labels.size(0)
            correct += (predicted == source_labels).sum().item()

        epoch_loss = running_loss / len(source_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{pre_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    for epoch in range(num_epochs):
        feature_extractor.train()
        domain_classifier.train()
        label_classifier.train()
        i = 0

        loss_domain_epoch = 0
        loss_label_epoch = 0
        total_loss_epoch = 0

        for source_data, target_data in zip(source_loader, tqdm(target_loader)):
            p = (float(i + epoch * min(len(source_loader), len(target_loader))) /
                 num_epochs / min(len(source_loader), len(target_loader)))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            source_images, source_labels = source_data
            source_images, source_labels = source_images.to(device), source_labels.to(device)
            target_images, _ = target_data
            target_images = target_images.to(device)

            # alternating domain data
            source_dlabel = torch.full((source_images.size(0),), 1, dtype=torch.float, device=device)
            target_dlabel = torch.full((target_images.size(0),), 0, dtype=torch.float, device=device)

            # training of domain classifier
            source_features = feature_extractor(source_images)
            target_features = feature_extractor(target_images)

            combined_features = torch.cat((source_features, target_features), dim=0)
            combined_dlabel = torch.cat((source_dlabel, target_dlabel), dim=0)

            domain_preds = domain_classifier(combined_features, lambda_p).view(-1)
            domain_loss = criterion(domain_preds, combined_dlabel)

            # training of label classifier
            features = feature_extractor(source_images)
            label_preds = label_classifier(features)
            label_loss = criterion(label_preds, source_labels)

            total_loss = label_loss + domain_loss
            optimizer_train.zero_grad()
            total_loss.backward()
            optimizer_train.step()

            loss_label_epoch += label_loss.item()
            loss_domain_epoch += domain_loss.item()
            total_loss_epoch += total_loss.item()

            label_acc = (torch.argmax(label_preds, dim=1) == source_labels).sum().item() / source_labels.size(0)
            domain_acc = (torch.round(domain_preds) == combined_dlabel).sum().item() / combined_dlabel.size(0)
            """
            print(f'Batches [{i + 1}/{min(len(source_loader), len(target_loader))}], '
                  f'Domain Loss: {domain_loss.item():.4f}, '
                  f'Label Loss: {label_loss.item():.4f}, '
                  f'Total Loss: {total_loss.item():.4f}, '
                  f'Label Accuracy: {label_acc * 100:.3f}%, '
                  f'Domain Accuracy: {domain_acc * 100:.3f}%'
                  )
            """
            i += 1

        scheduler.step()

        avg_loss_domain = loss_domain_epoch / min(len(source_loader), len(target_loader))
        avg_loss_label = loss_label_epoch / min(len(source_loader), len(target_loader))
        avg_total_loss = total_loss_epoch / min(len(source_loader), len(target_loader))

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Domain Loss: {avg_loss_domain:.4f}, '
              f'Label Loss: {avg_loss_label:.4f}, '
              f'Total Loss: {avg_total_loss:.4f}')

        wandb.log({
            'Domain Loss': avg_loss_domain,
            'Label Loss': avg_loss_label,
            'Total Loss': avg_total_loss,
        })

        # 테스트
        feature_extractor.eval()
        label_classifier.eval()
        domain_classifier.eval()

        def lc_tester(loader, group):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                features = feature_extractor(images)
                preds = label_classifier(features)
                preds = F.log_softmax(preds, dim=1)

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

                features = feature_extractor(images)
                preds = domain_classifier(features, lambda_p=0.0)
                preds = F.log_softmax(preds, dim=1)

                _, predicted = torch.max(preds.data, 1)
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
