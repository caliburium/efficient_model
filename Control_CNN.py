import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import CIFAR10, SVHN, MNIST, FashionMNIST
import torchvision.transforms as transforms
import wandb
import time
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleCNN(nn.Module):
    def __init__(self, channel, linear):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, channel//2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channel//2)
        self.conv2 = nn.Conv2d(channel//2, channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.fc1 = nn.Linear(channel * 16 * 16, linear)
        self.fc2 = nn.Linear(linear, 10)

    def forward(self, x, channel):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, channel * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='combine')
    args.add_argument('--channel', type=int, default=64)
    args.add_argument('--linear', type=int, default=40)
    args = args.parse_args()

    # Initialize Weights and Biases
    wandb.init(project="Efficient_Model_Research",
               entity="hails",
               config=args.__dict__,
               name="Controlgroup_Conv2D_" + args.mode + "_" + str(args.channel) + "/" + str(args.linear))

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

    if args.mode == 'combine':
        # Load MNIST, SVHN, CIFAR10, and FashionMNIST datasets for training
        mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform_MNIST)
        svhn_dataset = SVHN(root='./data', split='train', download=True, transform=transform)
        cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        fashionmnist_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform_MNIST)

        # Load MNIST, SVHN, CIFAR10, and FashionMNIST datasets for testing/validation
        mnist_test_dataset = MNIST(root='./data', train=False, download=True, transform=transform_MNIST)
        svhn_test_dataset = SVHN(root='./data', split='test', download=True, transform=transform)
        cifar10_test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        fmnist_test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform_MNIST)

        # Combine datasets
        combined_dataset = ConcatDataset([mnist_dataset, svhn_dataset, cifar10_dataset, fashionmnist_dataset])

        # Combine test datasets
        combined_test_dataset = ConcatDataset(
            [mnist_test_dataset, svhn_test_dataset, cifar10_test_dataset, fmnist_test_dataset])

        # Create DataLoader
        trainloader = DataLoader(combined_dataset, batch_size=64, shuffle=True, num_workers=2)
        mnistloader = DataLoader(mnist_test_dataset, batch_size=64, shuffle=False, num_workers=2)
        svhnloader = DataLoader(svhn_test_dataset, batch_size=64, shuffle=False, num_workers=2)
        cifar10loader = DataLoader(cifar10_test_dataset, batch_size=64, shuffle=False, num_workers=2)
        fmnistloader = DataLoader(fmnist_test_dataset, batch_size=64, shuffle=False, num_workers=2)
        testloader = DataLoader(combined_test_dataset, batch_size=64, shuffle=False, num_workers=2)

    elif args.mode == 'mnist':
        mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform_MNIST)
        mnist_test_dataset = MNIST(root='./data', train=False, download=True, transform=transform_MNIST)
        trainloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True, num_workers=2)
        testloader = DataLoader(mnist_test_dataset, batch_size=64, shuffle=False, num_workers=2)

    elif args.mode == 'svhn':
        svhn_dataset = SVHN(root='./data', split='train', download=True, transform=transform)
        svhn_test_dataset = SVHN(root='./data', split='test', download=True, transform=transform)
        trainloader = DataLoader(svhn_dataset, batch_size=64, shuffle=True, num_workers=2)
        testloader = DataLoader(svhn_test_dataset, batch_size=64, shuffle=False, num_workers=2)

    elif args.mode == 'cifar10':
        cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        cifar10_test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        trainloader = DataLoader(cifar10_dataset, batch_size=64, shuffle=True, num_workers=2)
        testloader = DataLoader(cifar10_test_dataset, batch_size=64, shuffle=False, num_workers=2)

    elif args.mode == 'fmnist':
        fashionmnist_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform_MNIST)
        fmnist_test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform_MNIST)
        trainloader = DataLoader(fashionmnist_dataset, batch_size=64, shuffle=True, num_workers=2)
        testloader = DataLoader(fmnist_test_dataset, batch_size=64, shuffle=False, num_workers=2)

    else:
        print("mode error")

    print("data load complete, start training")

    net = SimpleCNN(int(args.channel), int(args.linear)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):
        net.train()
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs, int(args.channel))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        wandb.log({"training_loss": running_loss / len(trainloader)}, step=epoch)
        end_time = time.time()
        training_time = end_time - start_time
        wandb.log({"training_time": training_time}, step=epoch)
        print('[%d] loss: %.3f, training time: %.3f seconds' % (epoch + 1, running_loss / len(trainloader), training_time))
        net.eval()

        def tester(loader, dataset):
            correct, total = 0, 0
            for data in loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images, int(args.channel))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            wandb.log({dataset + "_accuracy": accuracy}, step=epoch)
            print(dataset + ' accuracy: %.3f' % accuracy)

        with torch.no_grad():
            if args.mode == 'combine':
                tester(mnistloader, "mnist")
                tester(svhnloader, "svhn")
                tester(cifar10loader, "cifar10")
                tester(fmnistloader, "fmnist")
                tester(testloader, "total")
            elif args.mode == 'mnist':
                tester(testloader, "mnist")
            elif args.mode == 'svhn':
                tester(testloader, "svhn")
            elif args.mode == 'cifar10':
                tester(testloader, "cifar10")
            elif args.mode == 'mnist':
                tester(testloader, "fmnist")

    wandb.finish()


if __name__ == '__main__':
    main()
