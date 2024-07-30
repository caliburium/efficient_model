import argparse
import deeplake
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

from functions import ReverseLayerF, lr_lambda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepLakeDataset(Dataset):
    def __init__(self, deeplake_ds, transform=None):
        self.ds = deeplake_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        image = sample['images'].numpy()
        label = sample['labels'].numpy()

        if self.transform:
            image = self.transform(image)

        return image, label


class CaffeNet(nn.Module):
    def __init__(self, num_classes=100):
        super(CaffeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.norm = nn.LocalResponseNorm(5, 1.e-4, 0.75)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = self.norm(self.pool(self.relu(self.conv1(x*57.6))))
        x = self.norm(self.pool(self.relu(self.conv2(x))))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool(x)

        x = x.view(-1, 256 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, num_domains):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(4096, 1024),
        self.fc2 = nn.Linear(1024, 1024),
        self.fc3 = nn.Linear(1024, num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lambda_p):
        x = ReverseLayerF.apply(x, lambda_p)
        x = self.model(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pacs_train = deeplake.load("hub://activeloop/pacs-train")
    pacs_val = deeplake.load("hub://activeloop/pacs-val")
    pacs_test = deeplake.load("hub://activeloop/pacs-test")

    train_dataset = DeepLakeDataset(pacs_train, transform=transform)
    val_dataset = DeepLakeDataset(pacs_val, transform=transform)
    test_dataset = DeepLakeDataset(pacs_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    def train(model, train_loaders, criterion, optimizer, num_epochs=25):
        for epoch in range(num_epochs):
            model.train()
            for i, train_loader in enumerate(train_loaders):
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Domain [{i + 1}/{len(train_loaders)}], Loss: {loss.item():.4f}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    train(model, train_loaders, criterion, optimizer)

    def evaluate(model, test_loaders):
        model.eval()
        total_correct = 0
        total_images = 0

        with torch.no_grad():
            for test_loader in test_loaders:
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                total_correct += correct
                total_images += total
                print(f'Accuracy: {accuracy:.2f}%')

        overall_accuracy = 100 * total_correct / total_images
        print(f'Overall Accuracy: {overall_accuracy:.2f}%')

    evaluate(model, test_loaders)

if __name__ == '__main__':
    main()
