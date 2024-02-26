import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
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
    args = argparse.ArgumentParser()
    args = args.parse_args()

    # Initialize Weights and Biases
    wandb.init(project="Efficient_Model_Research",
               entity="hails",
               config=args.__dict__,
               name="DABB")

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

    source_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True, num_workers=4)
    source_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    source_loader_test = DataLoader(source_dataset, batch_size=64, shuffle=True, num_workers=4)

    target_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True, num_workers=4)
    target_dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    target_loader_test = DataLoader(target_dataset, batch_size=64, shuffle=True, num_workers=4)

    print("data load complete, start training")

    feature_extractor = FeatureExtractor().to(device)
    domain_classifier = DomainClassifier().to(device)
    label_classifier = LabelClassifier().to(device)

    # 각 네트워크에 대한 optimizer 설정
    optimizer_feature_extractor_domain_classifier = optim.Adam(
        list(feature_extractor.parameters()) + list(domain_classifier.parameters()), lr=0.001
    )

    optimizer_label_classifier = optim.Adam(
        list(feature_extractor.parameters()) + list(label_classifier.parameters()), lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    # 훈련
    num_epochs = 10
    for epoch in range(num_epochs):
        start_time = time.time()

        feature_extractor.train()
        domain_classifier.train()
        label_classifier.train()

        for source_data, target_data in zip(source_loader, target_loader):
            source_images, source_labels = source_data
            target_images, _ = target_data

            source_images, source_labels = source_images.to(device), source_labels.to(device)
            target_images = target_images.to(device)

            # Source domain에 대한 손실 계산
            source_features = feature_extractor(source_images)
            source_preds_domain = domain_classifier(source_features)
            source_labels_domain = torch.ones(source_preds_domain.size(0), 1).to(device)
            source_loss_domain = criterion(source_preds_domain, source_labels_domain)

            source_preds_label = label_classifier(source_features)
            source_loss_label = criterion(source_preds_label, source_labels)

            # Target domain에 대한 손실 계산
            target_features = feature_extractor(target_images)
            target_preds_domain = domain_classifier(target_features)
            target_labels_domain = torch.zeros(target_preds_domain.size(0), 1).to(device)
            target_loss_domain = criterion(target_preds_domain, target_labels_domain)

            # 총 손실 계산 및 역전파
            total_loss = source_loss_domain + source_loss_label + target_loss_domain

            optimizer_feature_extractor_domain_classifier.zero_grad()
            optimizer_label_classifier.zero_grad()
            total_loss.backward()
            optimizer_feature_extractor_domain_classifier.step()
            optimizer_label_classifier.step()

        end_time = time.time()

        # 결과 출력
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Total Loss: {total_loss:.4f}, '
              f'Time: {end_time - start_time:.2f} seconds')

        # 테스트
        feature_extractor.eval()
        label_classifier.eval()

        correct_source = 0
        total_source = 0

        with torch.no_grad():
            for source_images, source_labels in source_loader_test:
                source_images, source_labels = source_images.to(device), source_labels.to(device)

                source_features = feature_extractor(source_images)
                source_preds = label_classifier(source_features)

                _, predicted_source = torch.max(source_preds.data, 1)
                total_source += source_labels.size(0)
                correct_source += (predicted_source == source_labels).sum().item()

        source_accuracy = correct_source / total_source
        print(f'Source Accuracy: {source_accuracy * 100:.2f}%')

        correct_target = 0
        total_target = 0

        with torch.no_grad():
            for target_images, target_labels in target_loader_test:
                target_images, target_labels = target_images.to(device), target_labels.to(device)

                target_features = feature_extractor(target_images)
                target_preds = label_classifier(target_features)

                _, predicted_target = torch.max(target_preds.data, 1)
                total_target += target_labels.size(0)
                correct_target += (predicted_target == target_labels).sum().item()

        target_accuracy = correct_target / total_target
        print(f'Target Accuracy: {target_accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()
