import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functions import ReverseLayerF
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.relu(self.conv2(x))
        return x


class LabelClassifier(nn.Module):
    def __init__(self):
        super(LabelClassifier, self).__init__()
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, lambda_p):
        x = x.view(-1, 64 * 16 * 16)
        x = ReverseLayerF.apply(x, lambda_p)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x


def main():
    # MNIST, SVHN, CIFAR10, STL10
    args = argparse.ArgumentParser()
    args.add_argument('--epoch', type=int, default=5000)
    args.add_argument('--batch_size', type=int, default=2048)
    args.add_argument('--source', type=str, default='SVHN')
    args.add_argument('--target', type=str, default='MNIST')
    args.add_argument('--domain_lr', type=float, default=0.1)
    args.add_argument('--label_lr', type=float, default=0.1)
    args = args.parse_args()

    num_epochs = args.epoch
    batch_size = args.batch_size

    # Initialize Weights and Biases
    wandb.init(project="Efficient_Model_Research",
               entity="hails",
               config=args.__dict__,
               name="[Control] DAbB_S:" + args.source + "_T:" + args.target + "_OverFit"
               )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnist = transforms.Compose([
        transforms.Pad(2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_stl10 = transforms.Compose([
        transforms.Resize((32, 32), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # https://pytorch.org/vision/main/generated/torchvision.datasets.EMNIST.html

    if args.source == 'MNIST':
        source_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        source_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    elif args.source == 'SVHN':
        source_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        source_dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    elif args.source == 'CIFAR10':
        source_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        source_dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif args.source == 'STL10':
        source_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_stl10)
        source_dataset_test = datasets.STL10(root='./data', split='test', download=True, transform=transform_stl10)
    else:
        print("no source")

    if args.target == 'MNIST':
        target_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        target_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    elif args.target == 'SVHN':
        target_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        target_dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    elif args.target == 'CIFAR10':
        target_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        target_dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif args.target == 'STL10':
        target_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_stl10)
        target_dataset_test = datasets.STL10(root='./data', split='test', download=True, transform=transform_stl10)
    else:
        print("no target")

    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    source_loader_test = DataLoader(source_dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)
    target_loader_test = DataLoader(target_dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)

    print("data load complete, start training")

    feature_extractor = FeatureExtractor().to(device)
    domain_classifier = DomainClassifier().to(device)
    label_classifier = LabelClassifier().to(device)

    optimizer_domain_classifier = optim.Adam(
        list(feature_extractor.parameters()) + list(domain_classifier.parameters()), lr=args.domain_lr
    )

    optimizer_label_classifier = optim.Adam(
        list(feature_extractor.parameters()) + list(label_classifier.parameters()), lr=args.label_lr
    )

    criterion_d = nn.CrossEntropyLoss()
    criterion_l = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        start_time = time.time()

        feature_extractor.train()
        domain_classifier.train()
        label_classifier.train()
        i = 0

        loss_domain_epoch = 0
        loss_label_epoch = 0
        total_loss_epoch = 0

        for source_data, target_data in zip(source_loader, target_loader):
            p = float(i + epoch * min(len(source_loader), len(target_loader))) / num_epochs / min(len(source_loader), len(target_loader))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            source_images, source_labels = source_data
            source_images, source_labels = source_images.to(device), source_labels.to(device)
            target_images, _ = target_data
            target_images = target_images.to(device)

            source_features = feature_extractor(source_images)
            target_features = feature_extractor(target_images)
            source_dlabel = torch.full((source_features.size(0), 1), 1, dtype=torch.int, device=device)
            target_dlabel = torch.full((target_features.size(0), 1), 0, dtype=torch.int, device=device)

            combined_features = torch.cat((source_features, target_features), dim=0)
            combined_dlabel = torch.cat((source_dlabel, target_dlabel), dim=0)

            indices = torch.randperm(combined_features.size(0))
            combined_features_shuffled = combined_features[indices]
            combined_dlabel_shuffled = combined_dlabel[indices]

            label_preds = label_classifier(source_features)
            label_loss = criterion_l(label_preds, source_labels)

            domain_preds = domain_classifier(combined_features_shuffled, lambda_p)
            domain_loss = criterion_d(domain_preds, combined_dlabel_shuffled.view(-1,).long())

            total_loss = domain_loss + label_loss

            optimizer_label_classifier.zero_grad()
            optimizer_domain_classifier.zero_grad()
            total_loss.backward()
            optimizer_label_classifier.step()
            optimizer_domain_classifier.step()

            loss_label_epoch += label_loss.item()
            loss_domain_epoch += domain_loss.item()
            total_loss_epoch += total_loss.item()

            i += 1

        end_time = time.time()

        # 결과 출력
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Domain Loss: {loss_domain_epoch:.4f}, '
              f'Label Loss: {loss_label_epoch:.4f}, '
              f'Total Loss: {total_loss_epoch:.4f}, '
              f'Time: {end_time - start_time:.2f} seconds')

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

        with torch.no_grad():
            correct_source, total_source = 0, 0
            for source_images, source_labels in source_loader_test:
                source_images, source_labels = source_images.to(device), source_labels.to(device)

                source_features = feature_extractor(source_images)
                source_preds = label_classifier(source_features)

                _, predicted_source = torch.max(source_preds.data, 1)
                total_source += source_labels.size(0)
                correct_source += (predicted_source == source_labels).sum().item()

            source_accuracy = correct_source / total_source
            wandb.log({'[Label] Source Accuracy': source_accuracy}, step=epoch + 1)
            print(f'[Label] Source Accuracy: {source_accuracy * 100:.3f}%')

            correct_target, total_target = 0, 0
            for target_images, target_labels in target_loader_test:
                target_images, target_labels = target_images.to(device), target_labels.to(device)

                target_features = feature_extractor(target_images)
                target_preds = label_classifier(target_features)

                _, predicted_target = torch.max(target_preds.data, 1)
                total_target += target_labels.size(0)
                correct_target += (predicted_target == target_labels).sum().item()

            target_accuracy = correct_target / total_target
            wandb.log({'[Label] Target Accuracy': target_accuracy}, step=epoch + 1)
            print(f'[Label] Target Accuracy: {target_accuracy * 100:.3f}%')

            correct_source_domain, total_source_domain = 0, 0
            for source_images, _ in source_loader_test:
                source_images = source_images.to(device)

                source_features = feature_extractor(source_images)
                domain_preds = domain_classifier(source_features, lambda_p=0.0)

                _, predicted_domain_source = torch.max(domain_preds.data, 1)
                total_source_domain += source_images.size(0)
                correct_source_domain += (predicted_domain_source == 1).sum().item()

            source_domain_accuracy = correct_source_domain / total_source_domain
            wandb.log({'[Domain] Source Accuracy': source_domain_accuracy}, step=epoch + 1)
            print(f'[Domain] Source Accuracy: {source_domain_accuracy * 100:.3f}%')

            correct_target_domain, total_target_domain = 0, 0
            for target_images, _ in target_loader_test:
                target_images = target_images.to(device)

                target_features = feature_extractor(target_images)
                domain_preds = domain_classifier(target_features, lambda_p=0.0)

                _, predicted_domain_target = torch.max(domain_preds.data, 1)
                total_target_domain += target_images.size(0)
                correct_target_domain += (predicted_domain_target == 0).sum().item()

            target_domain_accuracy = correct_target_domain / total_target_domain
            wandb.log({'[Domain] Target Accuracy': target_domain_accuracy}, step=epoch + 1)
            print(f'[Domain] Target Accuracy: {target_domain_accuracy * 100:.3f}%')


if __name__ == '__main__':
    main()
