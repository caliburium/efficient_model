"""https://github.com/mashaan14/ADDA/blob/main/ADDA.ipynb"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Set your W&B project name and API key
wandb.init(project="ADDA-MNIST-to-USPS")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the encoder, discriminator, and classifier networks
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.encoder(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.discriminator(x)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def main():
    # Load MNIST dataset as source domain
    source_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    source_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=source_transform)
    source_dataloader = DataLoader(source_dataset, batch_size=64, shuffle=True, num_workers=2)

    # Load USPS dataset as target domain
    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    target_dataset = datasets.USPS(root='./data', train=True, download=True, transform=target_transform)
    target_dataloader = DataLoader(target_dataset, batch_size=64, shuffle=True, num_workers=2)

    # Instantiate networks
    encoder = Encoder().to(device)
    discriminator = Discriminator().to(device)
    classifier = Classifier().to(device)

    # Define loss functions
    adversarial_loss = nn.BCELoss()
    classification_loss = nn.CrossEntropyLoss()

    # Define optimizers
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=0.001)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.001)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        # Create tqdm progress bar for training
        train_bar = tqdm(source_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for source_data, target_data in train_bar:
            # Zero gradients
            optimizer_encoder.zero_grad()
            optimizer_discriminator.zero_grad()
            optimizer_classifier.zero_grad()

            # Source domain data
            source_images, source_labels = source_data
            source_images, source_labels = source_images.to(device), source_labels.to(device)

            # Target domain data
            target_images, _ = target_data
            target_images = target_images.to(device)

            # Forward pass through encoder
            source_features = encoder(source_images)
            target_features = encoder(target_images)

            # Forward pass through discriminator
            source_domain_preds = discriminator(source_features.detach())
            target_domain_preds = discriminator(target_features.detach())

            # Adversarial loss
            source_domain_labels = torch.ones_like(source_domain_preds)
            target_domain_labels = torch.zeros_like(target_domain_preds)

            adv_loss_source = adversarial_loss(source_domain_preds, source_domain_labels)
            adv_loss_target = adversarial_loss(target_domain_preds, target_domain_labels)

            # Classification loss on source domain
            source_preds = classifier(source_features)
            class_loss_source = classification_loss(source_preds, source_labels)

            # Total loss
            total_loss = adv_loss_source + adv_loss_target + class_loss_source

            # Backward pass and optimization
            total_loss.backward()
            optimizer_encoder.step()
            optimizer_discriminator.step()
            optimizer_classifier.step()

            # Log metrics using Weight and Bias
            wandb.log({"Adversarial Loss Source": adv_loss_source.item(),
                       "Adversarial Loss Target": adv_loss_target.item(),
                       "Classification Loss Source": class_loss_source.item(),
                       "Total Loss": total_loss.item()})

            # Log training metrics in tqdm progress bar
            train_bar.set_postfix({"Adversarial Loss Source": adv_loss_source.item(),
                                   "Adversarial Loss Target": adv_loss_target.item(),
                                   "Classification Loss Source": class_loss_source.item(),
                                   "Total Loss": total_loss.item()})

            # Print metrics for real-time monitoring
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Batch [{train_bar.n}/{len(train_bar)}], "
                  f"Adversarial Loss Source: {adv_loss_source.item():.4f}, "
                  f"Adversarial Loss Target: {adv_loss_target.item():.4f}, "
                  f"Classification Loss Source: {class_loss_source.item():.4f}, "
                  f"Total Loss: {total_loss.item():.4f}")

        # Test loop
        correct = 0
        total = 0
        test_bar = tqdm(target_dataloader, desc=f"Testing", leave=False)

        with torch.no_grad():
            for test_data in tqdm(target_dataloader, total=len(target_dataloader)):
                test_images, test_labels = test_data
                test_images, test_labels = test_images.to(device), test_labels.to(device)

                test_features = encoder(test_images)
                test_preds = classifier(test_features)

                _, predicted = torch.max(test_preds.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()

        accuracy = correct / total
        # Log testing metrics in tqdm progress bar
        test_bar.set_postfix({"Test Accuracy": accuracy})
        wandb.log({"Test Accuracy": accuracy})

    # Save final model in the 'model' folder
    torch.save(encoder.state_dict(), 'model/encoder_final.pth')
    torch.save(discriminator.state_dict(), 'model/discriminator_final.pth')
    torch.save(classifier.state_dict(), 'model/classifier_final.pth')


if __name__ == '__main__':
    main()
