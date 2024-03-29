import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.autograd import Function

# https://github.com/fungtion/DANN/blob/master/models/functions.py
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

# 모델 정의
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LabelClassifier(nn.Module):
    def __init__(self):
        super(LabelClassifier, self).__init__()
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    # MNIST, SVHN, CIFAR10, STL10
    args = argparse.ArgumentParser()
    args.add_argument('--epoch', type=int, default=100)
    args.add_argument('--source1', type=str, default='SVHN')
    args.add_argument('--source2', type=str, default='CIFAR10')
    args.add_argument('--target', type=str, default='MNIST')
    args.add_argument('--domain_lr', type=float, default=0.001)
    args.add_argument('--label_lr', type=float, default=0.001)
    args = args.parse_args()

    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="Efficient_Model_Research",
               entity="hails",
               config=args.__dict__,
               name="[Control] DAbB_S:" + args.source1 + "/" + args.source2 + "_T:" + args.target
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

    if args.source1 == 'MNIST':
        source1_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        source1_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    elif args.source1 == 'SVHN':
        source1_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        source1_dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    elif args.source1 == 'CIFAR10':
        source1_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        source1_dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif args.source1 == 'STL10':
        source1_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_stl10)
        source1_dataset_test = datasets.STL10(root='./data', split='test', download=True, transform=transform_stl10)
    else:
        print("no source1")

    if args.source2 == 'MNIST':
        source2_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        source2_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    elif args.source2 == 'SVHN':
        source2_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        source2_dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    elif args.source2 == 'CIFAR10':
        source2_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        source2_dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif args.source2 == 'STL10':
        source2_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_stl10)
        source2_dataset_test = datasets.STL10(root='./data', split='test', download=True, transform=transform_stl10)
    else:
        print("no source2")

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

    combined_source = ConcatDataset([source1_dataset_test, source2_dataset_test])
    source_loader_test = DataLoader(combined_source, batch_size=64, shuffle=True, num_workers=4)
    source1_loader = DataLoader(source1_dataset, batch_size=64, shuffle=True, num_workers=4)
    source1_loader_test = DataLoader(source1_dataset_test, batch_size=64, shuffle=True, num_workers=4)
    source2_loader = DataLoader(source2_dataset, batch_size=64, shuffle=True, num_workers=4)
    source2_loader_test = DataLoader(source2_dataset_test, batch_size=64, shuffle=True, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True, num_workers=4)
    target_loader_test = DataLoader(target_dataset_test, batch_size=64, shuffle=True, num_workers=4)

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

        for source1_data, source2_data, target_data in zip(source1_loader, source2_loader, target_loader):
            p = float(i + epoch * min(len(source1_loader), len(source2_loader), len(target_loader) )) / num_epochs / min(len(source1_loader), len(source2_loader), len(target_loader) )
            labmda_p = 2. / (1. + np.exp(-10 * p)) - 1


            source1_images, source1_labels = source1_data
            source2_images, source2_labels = source2_data
            target_images, _ = target_data

            source1_images, source1_labels = source1_images.to(device), source1_labels.to(device)
            source2_images, source2_labels = source2_images.to(device), source2_labels.to(device)
            target_images = target_images.to(device)

            # Feature Extract (source1 = 0, source2 = 1, target = 2)
            source1_features = feature_extractor(source1_images)
            source2_features = feature_extractor(source2_images)
            target_features = feature_extractor(target_images)

            # Flatten
            source1_features_ = source1_features.view(source1_features.size(0), -1)
            source2_features_ = source2_features.view(source2_features.size(0), -1)
            target_features_ = target_features.view(target_features.size(0), -1)

            source1_features_ = ReverseLayerF.apply(source1_features_, labmda_p)
            source2_features_ = ReverseLayerF.apply(source2_features_, labmda_p)
            target_features_ = ReverseLayerF.apply(target_features_, labmda_p)

            # revert back to original shape
            source1_features = source1_features_.view(source1_features.size())
            source2_features = source2_features_.view(source2_features.size())
            target_features = target_features_.view(target_features.size())

            source1_labels_domain = torch.full((source1_features.size(0), 1), 0, dtype=torch.int, device=device)
            source2_labels_domain = torch.full((source2_features.size(0), 1), 1, dtype=torch.int, device=device)
            target_labels_domain = torch.full((target_features.size(0), 1), 2, dtype=torch.int, device=device)

            # 소스만 합친거
            source_features = torch.cat((source1_features, source2_features), dim=0)
            source_labels = torch.cat((source1_labels, source2_labels), dim=0)
            source_labels_domain = torch.cat((source1_labels_domain, source2_labels_domain), dim=0)

            # 소스 섞어
            indices = torch.randperm(source_features.size(0))
            source_features = source_features[indices]
            source_labels = source_labels[indices]
            source_labels_domain = source_labels_domain[indices]

            # 소스/타겟 합친거
            combined_features = torch.cat((source1_features, source2_features, target_features), dim=0)
            combined_labels_domain = torch.cat((source1_labels_domain, source2_labels_domain, target_labels_domain), dim=0)

            # 소스/타겟 데이터 섞기
            indices = torch.randperm(combined_features.size(0))
            combined_features_shuffled = combined_features[indices]
            combined_labels_domain_shuffled = combined_labels_domain[indices]

            # 연산
            preds_domain = domain_classifier(combined_features_shuffled)
            loss_domain = criterion_d(preds_domain, combined_labels_domain_shuffled.view(-1,).long())
            preds_label = label_classifier(source_features)
            loss_label = criterion_l(preds_label, source_labels)

            # domina loss 비정상의 원인인듯
            total_loss = loss_domain + loss_label

            optimizer_domain_classifier.zero_grad()
            optimizer_label_classifier.zero_grad()
            total_loss.backward()
            optimizer_domain_classifier.step()
            optimizer_label_classifier.step()

        end_time = time.time()

        # 결과 출력
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Domain Loss: {loss_domain:.4f}, '
              f'Label Loss: {loss_label:.4f}, '
              f'Total Loss: {total_loss:.4f}, '
              f'Time: {end_time - start_time:.2f} seconds')

        wandb.log({
            'Domain Loss': loss_domain.item(),
            'Label Loss': loss_label.item(),
            'Total Loss': total_loss,
            'Training Time': end_time - start_time
        })

        # 테스트
        feature_extractor.eval()
        label_classifier.eval()

        correct_source, total_source = 0, 0
        correct_target, total_target = 0, 0
        correct_domain_source, total_domain_source = 0, 0
        correct_domain_target, total_domain_target = 0, 0

        """"# Source Combine 검증
        with torch.no_grad():
            for source_images, source_labels in source_loader_test:
                source_images, source_labels = source_images.to(device), source_labels.to(device)

                source_features = feature_extractor(source_images)
                source_preds = label_classifier(source_features)

                _, predicted_source = torch.max(source_preds.data, 1)
                total_source += source_labels.size(0)
                correct_source += (predicted_source == source_labels).sum().item()

        source_accuracy = correct_source / total_source
        wandb.log({'[Label] Source Accuracy': source_accuracy}, step=epoch+1)
        print(f'[Label] Source Accuracy: {source_accuracy * 100:.2f}%')
        """

        # Source 1 검증
        with torch.no_grad():
            for source_images, source_labels in source1_loader_test:
                source_images, source_labels = source_images.to(device), source_labels.to(device)

                source_features = feature_extractor(source_images)
                source_preds = label_classifier(source_features)

                _, predicted_source = torch.max(source_preds.data, 1)
                total_source += source_labels.size(0)
                correct_source += (predicted_source == source_labels).sum().item()

        source_accuracy = correct_source / total_source
        wandb.log({'[Label] Source_1 Accuracy': source_accuracy}, step=epoch+1)
        print(f'[Label] Source_1 Accuracy: {source_accuracy * 100:.3f}%')

        # Source 2 검증
        with torch.no_grad():
            for source_images, source_labels in source2_loader_test:
                source_images, source_labels = source_images.to(device), source_labels.to(device)

                source_features = feature_extractor(source_images)
                source_preds = label_classifier(source_features)

                _, predicted_source = torch.max(source_preds.data, 1)
                total_source += source_labels.size(0)
                correct_source += (predicted_source == source_labels).sum().item()

        source_accuracy = correct_source / total_source
        wandb.log({'[Label] Source_2 Accuracy': source_accuracy}, step=epoch+1)
        print(f'[Label] Source_2 Accuracy: {source_accuracy * 100:.3f}%')

        # Target 검증
        with torch.no_grad():
            for target_images, target_labels in target_loader_test:
                target_images, target_labels = target_images.to(device), target_labels.to(device)

                target_features = feature_extractor(target_images)
                target_preds = label_classifier(target_features)

                _, predicted_target = torch.max(target_preds.data, 1)
                total_target += target_labels.size(0)
                correct_target += (predicted_target == target_labels).sum().item()

        target_accuracy = correct_target / total_target
        wandb.log({'[Label] Target Accuracy': target_accuracy}, step=epoch+1)
        print(f'[Label] Target Accuracy: {target_accuracy * 100:.3f}%')

        with torch.no_grad():
            for source_images, _ in source1_loader_test:
                source_images = source_images.to(device)

                source_features = feature_extractor(source_images)
                source_preds_domain = domain_classifier(source_features)
                source_preds_domain = nn.Softmax()(source_preds_domain)

                predicted_domain_source = torch.round(source_preds_domain).long()
                total_domain_source += source_preds_domain.size(0)
                correct_domain_source += (predicted_domain_source == 0).sum().item()

        domain_accuracy_source = correct_domain_source / total_domain_source
        wandb.log({'[Domain] Source_1 Accuracy': domain_accuracy_source}, step=epoch+1)
        print(f'[Domain] Source_1 Accuracy: {domain_accuracy_source * 100:.3f}%')

        with torch.no_grad():
            for source_images, _ in source2_loader_test:
                source_images = source_images.to(device)

                source_features = feature_extractor(source_images)
                source_preds_domain = domain_classifier(source_features)
                source_preds_domain = nn.Softmax()(source_preds_domain)

                predicted_domain_source = torch.round(source_preds_domain).long()
                total_domain_source += source_preds_domain.size(0)
                correct_domain_source += (predicted_domain_source == 1).sum().item()

        domain_accuracy_source = correct_domain_source / total_domain_source
        wandb.log({'[Domain] Source_2 Accuracy': domain_accuracy_source}, step=epoch+1)
        print(f'[Domain] Source_2 Accuracy: {domain_accuracy_source * 100:.3f}%')

        with torch.no_grad():
            for target_images, _ in target_loader_test:
                target_images = target_images.to(device)

                target_features = feature_extractor(target_images)
                target_preds_domain = domain_classifier(target_features)
                target_preds_domain = nn.Softmax()(target_preds_domain)

                predicted_domain_target = torch.round(target_preds_domain).long()
                total_domain_target += target_preds_domain.size(0)
                correct_domain_target += (predicted_domain_target == 2).sum().item()

        domain_accuracy_target = correct_domain_target / total_domain_target
        wandb.log({'[Domain] Target Accuracy': domain_accuracy_target}, step=epoch+1)
        print(f'[Domain] Target Accuracy: {domain_accuracy_target * 100:.3f}%')


if __name__ == '__main__':
    main()
