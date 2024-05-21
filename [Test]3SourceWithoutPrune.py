import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from data_loader import data_loader
import wandb
import time

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
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        x = x.view(-1, 128 * 8 * 8)
        # x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def main():
    # MNIST, SVHN, CIFAR10, STL10
    args = argparse.ArgumentParser()
    args.add_argument('--epoch', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=50)
    args.add_argument('--source1', type=str, default='CIFAR10')
    args.add_argument('--source2', type=str, default='SVHN')
    args.add_argument('--source3', type=str, default='STL10')
    args.add_argument('--lr_domain', type=float, default=0.02)
    args.add_argument('--lr_class', type=float, default=0.2)

    args = args.parse_args()

    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="Efficient_Model_Research",
               entity="hails",
               config=args.__dict__,
               name="[Test]NoPrune_S1:" + args.source1 + "_S2:" + args.source2 + "_S3:" + args.source3
               )

    source1_loader, source1_loader_test = data_loader(args.source1, args.batch_size)
    source2_loader, source2_loader_test = data_loader(args.source2, args.batch_size)
    source3_loader, source3_loader_test = data_loader(args.source3, args.batch_size)

    print("data load complete, start training")

    feature_extractor = FeatureExtractor().to(device)
    domain_classifier = DomainClassifier().to(device)
    label_classifier = LabelClassifier().to(device)

    # optimizer_domain_classifier = optim.SGD(list(feature_extractor.parameters())
    #                                        + list(domain_classifier.parameters()), lr=args.lr_domain, momentum=0.9)
    # optimizer_label_classifier = optim.SGD(list(feature_extractor.parameters())
    #                                       + list(label_classifier.parameters()), lr=args.lr_class, momentum=0.9)
    optimizer_domain_classifier = optim.Adam(list(feature_extractor.parameters())
                                             + list(domain_classifier.parameters()), lr=args.lr_domain)
    optimizer_label_classifier = optim.Adam(list(feature_extractor.parameters())
                                            + list(label_classifier.parameters()), lr=args.lr_class)

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

        for source1_data, source2_data, source3_data in zip(source1_loader, source2_loader, source3_loader):
            source1_images, source1_labels = source1_data
            source1_images, source1_labels = source1_images.to(device), source1_labels.to(device)
            source2_images, source2_labels = source2_data
            source2_images, source2_labels = source2_images.to(device), source2_labels.to(device)
            source3_images, source3_labels = source3_data
            source3_images, source3_labels = source3_images.to(device), source3_labels.to(device)

            # combined source and target data
            source1_domain = torch.full((source1_images.size(0), 1), 0, dtype=torch.int, device=device)
            source2_domain = torch.full((source2_images.size(0), 1), 1, dtype=torch.int, device=device)
            source3_domain = torch.full((source3_images.size(0), 1), 2, dtype=torch.int, device=device)
            combined_data = torch.cat((source1_images, source2_images, source3_images), dim=0)
            combined_label = torch.cat((source1_labels, source2_labels, source3_labels), dim=0)
            combined_domain = torch.cat((source1_domain, source2_domain, source3_domain), dim=0)

            # training of domain classifier
            combined_features = feature_extractor(combined_data)
            combined_domain = combined_domain.view(-1,).long()

            # indices = torch.randperm(combined_features.size(0))
            # combined_features_shuffled = combined_features[indices]
            # combined_domain_shuffled = combined_domain[indices]

            domain_preds = domain_classifier(combined_features)
            domain_preds = F.log_softmax(domain_preds, dim=1)
            domain_loss = criterion(domain_preds, combined_domain)

            optimizer_domain_classifier.zero_grad()
            domain_loss.backward()
            optimizer_domain_classifier.step()

            # training of label classifier
            label_preds = label_classifier(combined_features)
            label_preds = F.log_softmax(label_preds, dim=1)
            label_loss = criterion(label_preds, combined_label)

            optimizer_label_classifier.zero_grad()
            label_loss.backward()
            optimizer_label_classifier.step()

            total_loss = label_loss + domain_loss

            loss_label_epoch += label_loss.item()
            loss_domain_epoch += domain_loss.item()
            total_loss_epoch += total_loss.item()

            label_acc = (torch.argmax(label_preds, dim=1) == combined_label).sum().item() / combined_label.size(0)
            domain_acc = (torch.argmax(domain_preds, dim=1) == combined_domain).sum().item() / combined_domain.size(0)

            print(f'Batches [{i + 1}/{min(len(source1_loader), len(source2_loader), len(source3_loader))}], '
                    f'Domain Loss: {domain_loss.item():.4f}, '
                    f'Label Loss: {label_loss.item():.4f}, '
                    f'Total Loss: {total_loss.item():.4f}, '
                    f'Label Accuracy: {label_acc * 100:.3f}%, '
                    f'Domain Accuracy: {domain_acc * 100:.3f}%'
                    )

            i += 1

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
            lc_tester(source1_loader_test, 'Source1')
            lc_tester(source2_loader_test, 'Source2')
            lc_tester(source3_loader_test, 'Source3')
            dc_tester(source1_loader_test, 'Source1', 0)
            dc_tester(source2_loader_test, 'Source2', 1)
            dc_tester(source3_loader_test, 'Source3', 2)


if __name__ == '__main__':
    main()
