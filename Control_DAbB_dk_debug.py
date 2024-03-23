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
        self.fc2 = nn.Linear(128, 2)

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
    args = argparse.ArgumentParser()
    args.add_argument('--epoch', type=int, default=100)
    args.add_argument('--source', type=str, default='MNIST')
    args.add_argument('--target', type=str, default='SVHN')
    args.add_argument('--domain_lr', type=float, default=0.001)
    args.add_argument('--label_lr', type=float, default=0.001)
    args = args.parse_args()

    num_epochs = args.epoch

    # Initialize Weights and Biases
    # wandb.init(project="Efficient_Model_Research",
    #            entity="hails",
    #            config=args.__dict__,
    #            name="[Control] DAbB_S:" + args.source + "_T:" + args.target
    #            # name="[Control] DAbB_" + str(args.epoch) + "_dom:" + str(args.domain_lr) + "_lab:" + str(args.label_lr)
    #            )

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

    transform_emnist = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # https://pytorch.org/vision/main/generated/torchvision.datasets.EMNIST.html

    if args.source == 'MNIST':
        source_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        source_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    elif args.source == 'EMNIST':
        source_dataset = datasets.EMNIST(root='./data', split='digits', train=True, download=True, transform=transform_mnist)
        source_dataset_test = datasets.EMNIST(root='./data', split='digits', train=False, download=True, transform=transform_mnist)
    elif args.source == 'SVHN':
        source_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        source_dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    else:
        print("no source")

    if args.target == 'MNIST':
        target_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        target_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    elif args.target == 'EMNIST':
        target_dataset = datasets.EMNIST(root='./data', split='digits', train=True, download=True, transform=transform_mnist)
        target_dataset_test = datasets.EMNIST(root='./data', split='digits', train=False, download=True, transform=transform_mnist)
    elif args.target == 'SVHN':
        target_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        target_dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    else:
        print("no target")

    source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True, num_workers=4)
    source_loader_test = DataLoader(source_dataset_test, batch_size=64, shuffle=True, num_workers=4)
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

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()


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

            # Feature Extract
            source_features = feature_extractor(source_images)
            target_features = feature_extractor(target_images)
            source_labels_domain = torch.full((source_features.size(0), 1), 0, dtype=torch.float, device=device)
            target_labels_domain = torch.full((target_features.size(0), 1), 1, dtype=torch.float, device=device)

            # Concatenate source_features and target_features
            combined_features = torch.cat((source_features, target_features), dim=0)
            combined_labels_domain = torch.cat((source_labels_domain, target_labels_domain), dim=0)

            # Shuffle indices
            indices = torch.randperm(combined_features.size(0))

            # Shuffle combined_features and combined_labels_domain using shuffled indices
            combined_features_shuffled = combined_features[indices]
            combined_labels_domain_shuffled = combined_labels_domain[indices]

            preds_domain = domain_classifier(combined_features_shuffled)

            loss_domain = criterion(preds_domain, combined_labels_domain_shuffled.squeeze().long())


            preds_label = label_classifier(source_features)
            loss_label = criterion(preds_label, source_labels)

            total_loss = loss_domain + loss_domain + loss_label


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

        # wandb.log({
        #     'Domain Loss': loss_domain.item(),
        #     'Label Loss': loss_label.item(),
        #     'Total Loss': total_loss,
        #     'Training Time': end_time - start_time
        # })

        # 테스트
        feature_extractor.eval()
        label_classifier.eval()

        correct_source, total_source = 0, 0
        correct_target, total_target = 0, 0
        correct_domain_source, total_domain_source = 0, 0
        correct_domain_target, total_domain_target = 0, 0

        with torch.no_grad():
            for source_images, source_labels in source_loader_test:
                source_images, source_labels = source_images.to(device), source_labels.to(device)

                source_features = feature_extractor(source_images)
                source_preds = label_classifier(source_features)

                _, predicted_source = torch.max(source_preds.data, 1)
                total_source += source_labels.size(0)
                correct_source += (predicted_source == source_labels).sum().item()

        source_accuracy = correct_source / total_source
        # wandb.log({'[Label] Source Accuracy': source_accuracy}, step=epoch+1)
        print(f'[Label] Source Accuracy: {source_accuracy * 100:.2f}%')

        with torch.no_grad():
            for target_images, target_labels in target_loader_test:
                target_images, target_labels = target_images.to(device), target_labels.to(device)

                target_features = feature_extractor(target_images)
                target_preds = label_classifier(target_features)

                _, predicted_target = torch.max(target_preds.data, 1)
                total_target += target_labels.size(0)
                correct_target += (predicted_target == target_labels).sum().item()

        target_accuracy = correct_target / total_target
        # wandb.log({'[Label] Target Accuracy': target_accuracy}, step=epoch+1)
        print(f'[Label] Target Accuracy: {target_accuracy * 100:.2f}%')

        with torch.no_grad():
            for source_images, _ in source_loader_test:
                source_images = source_images.to(device)

                source_features = feature_extractor(source_images)
                source_preds_domain = domain_classifier(source_features)

                predicted_domain_source = torch.round(source_preds_domain).long()
                total_domain_source += source_preds_domain.size(0)
                correct_domain_source += (predicted_domain_source == 1).sum().item()

        domain_accuracy_source = correct_domain_source / total_domain_source
        # wandb.log({'[Domain] Source Accuracy': domain_accuracy_source}, step=epoch+1)
        print(f'[Domain] Source Accuracy: {domain_accuracy_source * 100:.2f}%')

        with torch.no_grad():
            for target_images, _ in target_loader_test:
                target_images = target_images.to(device)

                target_features = feature_extractor(target_images)
                target_preds_domain = domain_classifier(target_features)

                predicted_domain_target = torch.round(target_preds_domain).long()
                total_domain_target += target_preds_domain.size(0)
                correct_domain_target += (predicted_domain_target == 0).sum().item()

        domain_accuracy_target = correct_domain_target / total_domain_target
        # wandb.log({'[Domain] Target Accuracy': domain_accuracy_target}, step=epoch+1)
        print(f'[Domain] Target Accuracy: {domain_accuracy_target * 100:.2f}%')


if __name__ == '__main__':
    main()
