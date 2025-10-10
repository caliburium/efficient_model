import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloader.data_loader import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, hidden_size=4096):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=4096)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    parser.add_argument('--lr_alpha', type=float, default=0.1)
    parser.add_argument('--lr_beta', type=float, default=0.25)
    args = parser.parse_args()

    num_epochs = args.epoch
    # Initialize Weights and Biases
    wandb.init(entity="hails",
               project="Efficient Model - Partition",
               config=args.__dict__,
               name="[CNN&SDLoss]MSC_" + str(args.hidden_size)
                    + "_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
                    + "_Adam"
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    cifar10_loader, cifar10_loader_test = data_loader('CIFAR10', args.batch_size)

    print("Data load complete, start training")

    def lr_lambda(progress):
        alpha = args.lr_alpha
        beta = args.lr_beta
        return (1 + alpha * progress) ** (-beta)

    model = SimpleCNN(num_classes=10, hidden_size=args.hidden_size).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        total_mnist_loss, total_svhn_loss, total_cifar10_loss, total_loss = 0, 0, 0, 0
        total_mnist_correct, total_svhn_correct, total_cifar10_correct = 0, 0, 0
        total_mnist_samples, total_svhn_samples, total_cifar10_samples = 0, 0, 0

        for mnist_data, svhn_data, cifar10_data in zip(mnist_loader, svhn_loader, cifar10_loader):
            mnist_images, mnist_labels = mnist_data
            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            svhn_images, svhn_labels = svhn_data
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            cifar10_images, cifar10_labels = cifar10_data
            cifar10_images, cifar10_labels = cifar10_images.to(device), cifar10_labels.to(device)

            mnist_outputs = model(mnist_images)
            svhn_outputs = model(svhn_images)
            cifar10_outputs = model(cifar10_images)

            mnist_loss = criterion(mnist_outputs, mnist_labels)
            svhn_loss = criterion(svhn_outputs, svhn_labels)
            cifar10_loss = criterion(cifar10_outputs, cifar10_labels)

            mnist_probs = torch.softmax(mnist_outputs, dim=1)
            svhn_probs = torch.softmax(svhn_outputs, dim=1)
            cifar10_probs = torch.softmax(cifar10_outputs, dim=1)

            num_probs = torch.cat((mnist_probs, svhn_probs), dim=0)
            loss_specialization_numbers = -torch.sum(num_probs * torch.log(num_probs + 1e-8))
            loss_specialization_cifar = -torch.sum(cifar10_probs * torch.log(cifar10_probs + 1e-8))
            loss_specialization = loss_specialization_numbers + loss_specialization_cifar

            all_probs = torch.cat((mnist_probs, svhn_probs, cifar10_probs), dim=0)
            avg_prob_global = torch.mean(all_probs, dim=0)
            loss_diversity = torch.sum(avg_prob_global * torch.log(avg_prob_global + 1e-8))

            loss = mnist_loss + svhn_loss + cifar10_loss + (loss_specialization * 0.01) + (loss_diversity * 0.1)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mnist_loss += mnist_loss.item() * mnist_labels.size(0)
            total_svhn_loss += svhn_loss.item() * svhn_labels.size(0)
            total_cifar10_loss += cifar10_loss.item() * cifar10_labels.size(0)
            total_loss += loss.item() * (mnist_labels.size(0) + svhn_labels.size(0) + cifar10_labels.size(0))

            mnist_correct = (torch.argmax(mnist_outputs, dim=1) == mnist_labels).sum().item()
            svhn_correct = (torch.argmax(svhn_outputs, dim=1) == svhn_labels).sum().item()
            cifar10_correct = (torch.argmax(cifar10_outputs, dim=1) == cifar10_labels).sum().item()

            total_mnist_correct += mnist_correct
            total_svhn_correct += svhn_correct
            total_cifar10_correct += cifar10_correct

            total_mnist_samples += mnist_labels.size(0)
            total_svhn_samples += svhn_labels.size(0)
            total_cifar10_samples += cifar10_labels.size(0)

            i += 1

        scheduler.step()

        mnist_acc_epoch = total_mnist_correct / total_mnist_samples * 100
        svhn_acc_epoch = total_svhn_correct / total_svhn_samples * 100
        cifar10_acc_epoch = total_cifar10_correct / total_cifar10_samples * 100

        mnist_avg_loss = total_mnist_loss / total_mnist_samples
        svhn_avg_loss = total_svhn_loss / total_svhn_samples
        cifar10_avg_loss = total_cifar10_loss / total_cifar10_samples
        total_avg_loss = total_loss / (total_mnist_samples + total_svhn_samples + total_cifar10_samples)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], | '
            f'Total Loss: {total_avg_loss:.4f}, | '
            f'MNIST Loss: {mnist_avg_loss:.4f}, | '
            f'SVHN Loss: {svhn_avg_loss:.4f}, | '
            f'CIFAR-10 Loss: {cifar10_avg_loss:.4f} | '
        )

        print(
            f'MNIST Acc: {mnist_acc_epoch:.3f}% | '
            f'SVHN Acc: {svhn_acc_epoch:.3f}% | '
            f'CIFAR Acc: {cifar10_acc_epoch:.3f}% | '
        )

        wandb.log({
            'Train/Label Loss': total_avg_loss,
            'Train/MNIST Label Loss': mnist_avg_loss,
            'Train/SVHN Label Loss': svhn_avg_loss,
            'Train/CIFAR Label Loss': cifar10_avg_loss,
            'Train/MNIST Label Accuracy': mnist_acc_epoch,
            'Train/SVHN Label Accuracy': svhn_acc_epoch,
            'Train/CIFAR Label Accuracy': cifar10_acc_epoch,
        }, step=epoch)

        model.eval()

        def tester(loader, group):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output = model(images)
                total += labels.size(0)
                correct += (torch.argmax(class_output, dim=1) == labels).sum().item()

            accuracy = correct / total * 100
            wandb.log({f'Test/Label {group} Accuracy': accuracy}, step=epoch)
            print(f'Test {group} | Label Acc: {accuracy:.3f}%')

        with torch.no_grad():
            tester(mnist_loader_test, 'MNIST')
            tester(svhn_loader_test, 'SVHN')
            tester(cifar10_loader_test, 'CIFAR')

if __name__ == '__main__':
    main()
