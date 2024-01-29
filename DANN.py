import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10, FashionMNIST
from torch.utils.data import ConcatDataset
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the Backbone network (SageConv)
class SageConvBackbone(nn.Module):
    def __init__(self):
        super(SageConvBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 32 * 32, 256)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc(x))
        return x


# Define the Domain Discriminator
class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the Label Classifier
class LabelClassifier(nn.Module):
    def __init__(self):
        super(LabelClassifier, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(backbone, domain_discriminator, label_classifier, dataloader, device, optimizer_backbone, optimizer_domain, optimizer_classifier, criterion_label, criterion_domain, alpha=0.1):
    backbone.train()
    domain_discriminator.train()
    label_classifier.train()

    total_loss_backbone = 0.0
    total_loss_domain = 0.0
    total_loss_classifier = 0.0
    correct_classifier = 0
    total_classifier = 0
    correct_domain = 0
    total_domain = 0

    start_time = time.time()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        features = backbone(inputs)
        domain_output = domain_discriminator(features)
        label_output = label_classifier(features)

        # Calculate losses
        label_loss = criterion_label(label_output, labels)
        domain_loss = criterion_domain(domain_output, torch.ones_like(domain_output))

        total_loss_classifier += label_loss.item()
        total_loss_domain += domain_loss.item()

        # Backward and optimize
        optimizer_backbone.zero_grad()
        optimizer_domain.zero_grad()
        optimizer_classifier.zero_grad()

        total_loss = label_loss + alpha * domain_loss
        total_loss.backward()

        optimizer_backbone.step()
        optimizer_domain.step()
        optimizer_classifier.step()

        # Accuracy calculation
        _, predicted = label_output.max(1)
        total_classifier += labels.size(0)
        correct_classifier += predicted.eq(labels).sum().item()

        _, predicted_domain = (domain_output > 0.5).long().squeeze().max(0)
        total_domain += labels.size(0)
        correct_domain += predicted_domain.eq(torch.ones_like(predicted_domain)).sum().item()

    end_time = time.time()
    elapsed_time = end_time - start_time

    accuracy_classifier = correct_classifier / total_classifier
    accuracy_domain = correct_domain / total_domain
    average_loss_classifier = total_loss_classifier / len(dataloader)
    average_loss_domain = total_loss_domain / len(dataloader)

    return accuracy_classifier, accuracy_domain, average_loss_classifier, average_loss_domain, elapsed_time


def test_model(backbone, domain_discriminator, label_classifier, dataloader, device, criterion_label, criterion_domain):
    backbone.eval()
    domain_discriminator.eval()
    label_classifier.eval()

    total_loss_backbone = 0.0
    total_loss_domain = 0.0
    total_loss_classifier = 0.0
    correct_classifier = 0
    total_classifier = 0
    correct_domain = 0
    total_domain = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            features = backbone(inputs)
            domain_output = domain_discriminator(features)
            label_output = label_classifier(features)

            # Calculate losses
            label_loss = criterion_label(label_output, labels)
            domain_loss = criterion_domain(domain_output, torch.ones_like(domain_output))

            total_loss_classifier += label_loss.item()
            total_loss_domain += domain_loss.item()

            # Accuracy calculation
            _, predicted = label_output.max(1)
            total_classifier += labels.size(0)
            correct_classifier += predicted.eq(labels).sum().item()

            _, predicted_domain = (domain_output > 0.5).long().squeeze().max(0)
            total_domain += labels.size(0)
            correct_domain += predicted_domain.eq(torch.ones_like(predicted_domain)).sum().item()

    accuracy_classifier = correct_classifier / total_classifier
    accuracy_domain = correct_domain / total_domain
    average_loss_classifier = total_loss_classifier / len(dataloader)
    average_loss_domain = total_loss_domain / len(dataloader)

    return accuracy_classifier, accuracy_domain, average_loss_classifier, average_loss_domain



def main():
    # Load MNIST, SVHN, CIFAR10, and FashionMNIST datasets
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])

    # Load MNIST, SVHN, CIFAR10, and FashionMNIST datasets for training
    mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    svhn_dataset = SVHN(root='./data', split='train', download=True, transform=transform)
    cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    fashionmnist_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)

    # Load MNIST, SVHN, CIFAR10, and FashionMNIST datasets for testing/validation
    mnist_test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    svhn_test_dataset = SVHN(root='./data', split='test', download=True, transform=transform)
    cifar10_test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    fashionmnist_test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Combine datasets
    combined_dataset = ConcatDataset([mnist_dataset, svhn_dataset, cifar10_dataset, fashionmnist_dataset])

    # Combine test datasets
    combined_test_dataset = ConcatDataset([mnist_test_dataset, svhn_test_dataset, cifar10_test_dataset, fashionmnist_test_dataset])

    # Create DataLoader
    dataloader = DataLoader(combined_dataset, batch_size=64, shuffle=True, num_workers=2)

    # Create test DataLoader
    test_dataloader = DataLoader(combined_test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Initialize the models and optimizers
    backbone = SageConvBackbone().to(device)
    domain_discriminator = DomainDiscriminator().to(device)
    label_classifier = LabelClassifier().to(device)

    optimizer_backbone = optim.Adam(backbone.parameters(), lr=0.001)
    optimizer_domain = optim.Adam(domain_discriminator.parameters(), lr=0.001)
    optimizer_classifier = optim.Adam(label_classifier.parameters(), lr=0.001)

    criterion_label = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        accuracy_classifier, accuracy_domain, average_loss_classifier, average_loss_domain, elapsed_time = train_model(
            backbone, domain_discriminator, label_classifier, dataloader, device,
            optimizer_backbone, optimizer_domain, optimizer_classifier,
            criterion_label, criterion_domain
        )
        print(f'Epoch [{epoch + 1}/{num_epochs}], Classifier Accuracy: {accuracy_classifier:.4f}, Domain Accuracy: {accuracy_domain:.4f}, Classifier Loss: {average_loss_classifier:.4f}, Domain Loss: {average_loss_domain:.4f}, Time: {elapsed_time:.2f} seconds')

        accuracy_classifier, accuracy_domain, average_loss_classifier, average_loss_domain = test_model(
            backbone, domain_discriminator, label_classifier, test_dataloader, device,
            criterion_label, criterion_domain
        )
        print(f'Test Classifier Accuracy: {accuracy_classifier:.4f}, Test Domain Accuracy: {accuracy_domain:.4f}, Test Classifier Loss: {average_loss_classifier:.4f}, Test Domain Loss: {average_loss_domain:.4f}')


if __name__ == '__main__':
    main()
