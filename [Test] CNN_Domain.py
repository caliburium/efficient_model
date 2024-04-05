import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN, MNIST
import torchvision.transforms as transforms
import wandb
import time
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.bn2(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.drop(self.relu(self.bn(self.fc1(x))))
        x = F.softmax(self.fc2(x), dim=1)
        return x


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=256)
    args.add_argument('--learning_rate', type=float, default=0.001)

    args = args.parse_args()
    batch_size = args.batch_size

    # Initialize Weights and Biases
    wandb.init(project="Efficient_Model_Research",
               entity="hails",
               config=args.__dict__,
               name="[Test] CNN_Domain_S:MNIST_T:SVHN"
               )

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
    trainloader_mnist = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    mnist_test_dataset = MNIST(root='./data', train=False, download=True, transform=transform_MNIST)
    testloader_mnist = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # svhn_dataset = SVHN(root='./data', split='train', download=True, transform=transform)
    # trainloader_svhn = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    svhn_test_dataset = SVHN(root='./data', split='test', download=True, transform=transform)
    testloader_svhn = DataLoader(svhn_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("data load complete, start training")

    net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    for epoch in range(50):
        net.train()
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader_mnist, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = net(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        wandb.log({"Label Loss": running_loss / len(trainloader_mnist)}, step=epoch)
        end_time = time.time()
        training_time = end_time - start_time
        wandb.log({"Training Time": training_time}, step=epoch)
        print('[%d] loss: %.3f, training time: %.3f seconds' % (epoch + 1, running_loss / len(trainloader_mnist), training_time))
        net.eval()

        def tester(loader, dataset):
            correct, total = 0, 0
            for data in loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            wandb.log({'[Label] ' + dataset + "Accuracy": accuracy}, step=epoch)
            print('[Label] ' + dataset + f'Accuracy: {accuracy * 100:.3f}%')

        with torch.no_grad():
            tester(testloader_mnist, "Source_ ")
            tester(testloader_svhn, "Target ")

    wandb.finish()


if __name__ == '__main__':
    main()
