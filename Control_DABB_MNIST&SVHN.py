import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# Set device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the feature extractor (simple CNN)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

# Define the domain classifier
class DomainClassifier(nn.Module):
    def __init__(self, input_size):
        super(DomainClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Define the main model
class DomainAdaptationModel(nn.Module):
    def __init__(self):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        self.domain_classifier = DomainClassifier(512)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_output = self.classifier(features)
        domain_output = self.domain_classifier(features)
        return class_output, domain_output

# Set up data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load MNIST and SVHN datasets
source_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
target_dataset = datasets.SVHN(root='./data', split='train', transform=transform, download=True)

# Create data loaders
source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True, num_workers=4)
target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True, num_workers=4)

# Create the model, optimizer, and loss function
model = DomainAdaptationModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.BCELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for source_data, target_data in zip(source_loader, target_loader):
        source_images, source_labels = source_data
        target_images, _ = target_data

        # Move data to device
        source_images, source_labels = source_images.to(device), source_labels.to(device)
        target_images = target_images.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        source_class_output, source_domain_output = model(source_images)
        target_class_output, target_domain_output = model(target_images)

        # Calculate losses
        class_loss = criterion_class(source_class_output, source_labels)
        domain_loss = criterion_domain(source_domain_output, torch.zeros(source_domain_output.size(0), 1).to(device)) + \
                      criterion_domain(target_domain_output, torch.ones(target_domain_output.size(0), 1).to(device))

        # Total loss
        total_loss = class_loss + domain_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Class Loss: {class_loss.item():.4f}, Domain Loss: {domain_loss.item():.4f}')

# Test the model on the target domain
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in target_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, _ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy on Target Domain: {accuracy * 100:.2f}%')