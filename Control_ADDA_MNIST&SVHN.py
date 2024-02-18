import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN, MNIST
import torchvision.transforms as transforms
import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 16 * 16)
        return x


class LabelClassifier(nn.Module):
    def __init__(self):
        super(LabelClassifier, self).__init__()
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = torch.sigmoid(self.fc2(x))
        return x


def main():
    # Initialize Weights and Biases
    wandb.init(project="Efficient_Model_Research",
               entity="hails",
               name="Controlgroup_ADDA_MNIST&SVHN")

    transform_MNIST = transforms.Compose([
        transforms.Pad(2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform_MNIST)
    mnist_loader = DataLoader(mnist_dataset, batch_size=4, shuffle=True, num_workers=2)
    mnist_testset = MNIST(root='./data', train=False, download=True, transform=transform_MNIST)
    mnist_testloader = DataLoader(mnist_testset, batch_size=4, shuffle=False, num_workers=2)
    svhn_dataset = SVHN(root='./data', split='train', download=True, transform=transform)
    svhn_loader = DataLoader(svhn_dataset, batch_size=4, shuffle=True, num_workers=2)
    svhn_testset = SVHN(root='./data', split='test', download=True, transform=transform)
    svhn_testloader = DataLoader(svhn_testset, batch_size=4, shuffle=False, num_workers=2)

    feature_extractor = FeatureExtractor().to(device)
    classifier = LabelClassifier().to(device)
    domain_discriminator = DomainDiscriminator().to(device)

    criterion_lc = nn.CrossEntropyLoss()
    optimizer_lc = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=0.001)

    criterion_dd = nn.BCELoss()
    optimizer_dd = optim.Adam(domain_discriminator.parameters(), lr=0.001)

    for epoch in range(20):
        feature_extractor.train()
        classifier.train()
        domain_discriminator.train()

        start_time = time.time()

        for i, (svhn_data, mnist_data) in enumerate(zip(svhn_loader, mnist_loader), 0):
            mnist_inputs, mnist_labels = mnist_data
            svhn_inputs, _ = svhn_data

            mnist_inputs, mnist_labels, svhn_inputs = mnist_inputs.to(device), mnist_labels.to(device), svhn_inputs.to(device)

            # Train Domain Discriminator
            optimizer_dd.zero_grad()

            mnist_features = feature_extractor(mnist_inputs)
            svhn_features = feature_extractor(svhn_inputs)

            mnist_domain_labels = torch.ones(mnist_features.size(0), 1).to(device)
            svhn_domain_labels = torch.zeros(svhn_features.size(0), 1).to(device)

            mnist_predictions = domain_discriminator(mnist_features.detach())
            svhn_predictions = domain_discriminator(svhn_features.detach())

            loss_dd_mnist = criterion_dd(mnist_predictions, mnist_domain_labels)
            loss_dd_svhn = criterion_dd(svhn_predictions, svhn_domain_labels)

            loss_dd = (loss_dd_mnist + loss_dd_svhn) / 2
            loss_dd.backward(retain_graph=True)  # Use retain_graph=True to avoid freeing intermediate values
            optimizer_dd.step()

            loss_dd_mnist_classifier = criterion_dd(mnist_predictions, svhn_domain_labels)

            optimizer_lc.zero_grad()
            mnist_classifier_predictions = classifier(mnist_features)
            loss_lc = criterion_lc(mnist_classifier_predictions, mnist_labels)

            total_loss = loss_lc - loss_dd_mnist_classifier
            total_loss.backward()
            optimizer_lc.step()

        end_time = time.time()  # Record the end time
        training_time = end_time - start_time

        wandb.log({"domain_discriminator_loss": loss_dd.item(), "classifier_loss": loss_lc.item(), "total_loss": total_loss.item(),
                   "training_time": training_time}, step=epoch)

        # Print loss values
        print(f"[Epoch {epoch}] Loss (Domain Disc.): {loss_dd.item():.4f}, Loss (Classifier): {loss_lc.item():.4f}, Loss (Total): {total_loss.item():.4f}, Training Time: {training_time:.2f} seconds")

        # Testing
        feature_extractor.eval()
        classifier.eval()
        domain_discriminator.eval()

        total_domain_discriminator_correct = 0
        total_classifier_correct = 0
        total_samples = 0

        with torch.no_grad():
            for svhn_data, mnist_data in zip(svhn_testloader, mnist_testloader):
                svhn_inputs, svhn_labels = svhn_data
                mnist_inputs, mnist_labels = mnist_data

                svhn_inputs, svhn_labels = svhn_inputs.to(device), svhn_labels.to(device)
                mnist_inputs, mnist_labels = mnist_inputs.to(device), mnist_labels.to(device)

                svhn_features = feature_extractor(svhn_inputs)
                mnist_features = feature_extractor(mnist_inputs)

                svhn_domain_labels = torch.ones(svhn_features.size(0), 1).to(device)
                mnist_domain_labels = torch.zeros(mnist_features.size(0), 1).to(device)

                svhn_predictions = domain_discriminator(svhn_features)
                mnist_predictions = domain_discriminator(mnist_features)

                svhn_domain_discriminator_correct = torch.sum((svhn_predictions > 0.5).squeeze() == svhn_domain_labels.byte())
                mnist_domain_discriminator_correct = torch.sum((mnist_predictions <= 0.5).squeeze() == mnist_domain_labels.byte())

                total_domain_discriminator_correct += svhn_domain_discriminator_correct.item() + mnist_domain_discriminator_correct.item()

                svhn_classifier_predictions = classifier(svhn_features)
                svhn_classifier_correct = torch.sum(torch.argmax(svhn_classifier_predictions, dim=1) == svhn_labels)
                total_classifier_correct += svhn_classifier_correct.item()

                total_samples += svhn_labels.size(0)

        domain_discriminator_accuracy = total_domain_discriminator_correct / (2 * total_samples)
        classifier_accuracy = total_classifier_correct / total_samples
        wandb.log({"domain_discriminator_accuracy": domain_discriminator_accuracy,
                   "classifier_accuracy": classifier_accuracy}, step=epoch)

        print(f"[Epoch {epoch}] Domain Disc. Acc: {domain_discriminator_accuracy:.4f}, Classifier Acc: {classifier_accuracy:.4f}")

    wandb.finish()


if __name__ == '__main__':
    main()
