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
# device = 'cpu'


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 2, 1)
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, 3, 2, 1)
        x = self.relu(self.conv3(x))
        return x


class LabelClassifier(nn.Module):
    def __init__(self):
        super(LabelClassifier, self).__init__()
        self.fc1 = nn.Linear(128 * 8 * 8, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.fc3 = nn.Linear(2048, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        # self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(1024, 2)

        # they used 100-2 for mnist dataset
        self.fc1 = nn.Linear(128 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 2)


    def forward(self, x, lambda_p):
        x = x.view(-1, 128 * 8 * 8)
        x = ReverseLayerF.apply(x, lambda_p)
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))

        # they used 100-2 for mnist dataset
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def main():
    # MNIST, SVHN, CIFAR10, STL10
    args = argparse.ArgumentParser()
    args.add_argument('--epoch', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=50)
    args.add_argument('--source', type=str, default='SVHN')
    args.add_argument('--target', type=str, default='MNIST')
    args.add_argument('--lr_domain', type=float, default=0.02)
    args.add_argument('--lr_class', type=float, default=0.2)

    args = args.parse_args()

    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="Efficient_Model_Research",
               entity="hails",
               config=args.__dict__,
               name="DAbB_S:" + args.source + "_T:" + args.target
                    + "_lr(C)" + str(args.lr_class) + "_lr(D)" + str(args.lr_domain)
                    + "_Batch:" + str(args.batch_size)
               )

    source_loader, source_loader_test = data_loader(args.source, args.batch_size)
    target_loader, target_loader_test = data_loader(args.target, args.batch_size)

    print("data load complete, start training")

    feature_extractor = FeatureExtractor().to(device)
    domain_classifier = DomainClassifier().to(device)
    label_classifier = LabelClassifier().to(device)

    optimizer_domain_classifier = optim.SGD(list(feature_extractor.parameters())
                                            + list(domain_classifier.parameters()), lr=args.lr_domain, momentum=0.9)
    optimizer_label_classifier = optim.SGD(list(feature_extractor.parameters())
                                           + list(label_classifier.parameters()), lr=args.lr_class, momentum=0.9)

    scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_domain_classifier, lr_lambda)
    scheduler_l = optim.lr_scheduler.LambdaLR(optimizer_label_classifier, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        start_time = time.time()

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

            # combined source and target data
            source_dlabel = torch.full((source_images.size(0), 1), 1, dtype=torch.int, device=device)
            target_dlabel = torch.full((target_images.size(0), 1), 0, dtype=torch.int, device=device)
            combined_data = torch.cat((source_images, target_images), dim=0)
            combined_dlabel = torch.cat((source_dlabel, target_dlabel), dim=0)

            # training of domain classifier
            combined_features = feature_extractor(combined_data)
            combined_dlabel = combined_dlabel.view(-1,).long()

            domain_preds = domain_classifier(combined_features, lambda_p)
            domain_preds = F.log_softmax(domain_preds, dim=1)
            domain_loss = criterion(domain_preds, combined_dlabel)

            optimizer_domain_classifier.zero_grad()
            domain_loss.backward()
            optimizer_domain_classifier.step()

            # training of label classifier
            features = feature_extractor(source_images)
            label_preds = label_classifier(features)
            label_preds = F.log_softmax(label_preds, dim=1)
            label_loss = criterion(label_preds, source_labels)

            optimizer_label_classifier.zero_grad()
            label_loss.backward()
            optimizer_label_classifier.step()

            # source_features = feature_extractor(source_images)
            # target_features = feature_extractor(target_images)
            # source_dlabel = torch.full((source_features.size(0), 1), 1, dtype=torch.int, device=device)
            # target_dlabel = torch.full((target_features.size(0), 1), 0, dtype=torch.int, device=device)
            #
            # combined_features = torch.cat((source_features, target_features), dim=0)
            # combined_dlabel = torch.cat((source_dlabel, target_dlabel), dim=0)
            #
            # indices = torch.randperm(combined_features.size(0))
            # combined_features_shuffled = combined_features[indices]
            # combined_dlabel_shuffled = combined_dlabel[indices]
            #
            # domain_preds = domain_classifier(combined_features_shuffled, lambda_p)
            # domain_loss = criterion(domain_preds, combined_dlabel_shuffled.view(-1,).long())
            #
            # optimizer_domain_classifier.zero_grad()
            # domain_loss.backward()
            # optimizer_domain_classifier.step()
            #
            # label_preds = label_classifier(source_features)
            # label_loss = criterion(label_preds, source_labels)
            #
            # optimizer_label_classifier.zero_grad()
            # label_loss.backward()
            # optimizer_label_classifier.step()

            total_loss = label_loss + domain_loss

            loss_label_epoch += label_loss.item()
            loss_domain_epoch += domain_loss.item()
            total_loss_epoch += total_loss.item()

            label_acc = (torch.argmax(label_preds, dim=1) == source_labels).sum().item() / source_labels.size(0)
            domain_acc = (torch.argmax(domain_preds, dim=1) == combined_dlabel).sum().item() / combined_dlabel.size(0)

            print(f'Batches [{i + 1}/{min(len(source_loader), len(target_loader))}], '
                    f'Domain Loss: {domain_loss.item():.4f}, '
                    f'Label Loss: {label_loss.item():.4f}, '
                    f'Total Loss: {total_loss.item():.4f}, '
                    f'Label Accuracy: {label_acc * 100:.3f}%, '
                    f'Domain Accuracy: {domain_acc * 100:.3f}%'
                    )

            i += 1

        scheduler_d.step()
        scheduler_l.step()

        end_time = time.time()
        print()
        # 결과 출력
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Domain Loss: {loss_domain_epoch:.4f}, '
              f'Label Loss: {loss_label_epoch:.4f}, '
              f'Total Loss: {total_loss_epoch:.4f}, '
              f'Time: {end_time - start_time:.2f} seconds'
              )

        wandb.log({
            'Domain Loss': loss_domain_epoch,
            'Label Loss': loss_label_epoch,
            'Total Loss': total_loss_epoch,
            'Training Time': end_time - start_time
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
