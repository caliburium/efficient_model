import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from dataloader.data_loader import data_loader
import torchvision.models as models

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    args = parser.parse_args()

    num_epochs = args.epoch
    # Initialize Weights and Biases
    wandb.init(entity="hails",
               project="Efficient Model",
               config=args.__dict__,
               name="[VGG]MSC_pretrain:False_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    MNIST_loader, MNIST_loader_test = data_loader('MNIST', args.batch_size)
    SVHN_loader, SVHN_loader_test = data_loader('SVHN', args.batch_size)
    CIFAR10_loader, CIFAR10_loader_test = data_loader('CIFAR10', args.batch_size)
    # _, STL10_loader_test = data_loader('STL10', args.batch_size)

    print("Data load complete, start training")

    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_mnist_loss, total_svhn_loss, total_cifar10_loss, total_loss = 0, 0, 0, 0


        for mnist_data, svhn_data, cifar10_data in zip(MNIST_loader, SVHN_loader, CIFAR10_loader):
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
            loss = mnist_loss + svhn_loss + cifar10_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mnist_loss += mnist_loss.item()
            total_svhn_loss += svhn_loss.item()
            total_cifar10_loss += cifar10_loss.item()
            total_loss += loss.item()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}] | Total Loss: {total_loss:.4f} | '
            f'MNIST LOSS: {total_mnist_loss:.4f} | SVHN Loss: {total_svhn_loss:.4f} | CIFAR10 Loss: {total_cifar10_loss:.4f}')

        """
        wandb.log({
            'Total Loss': total_loss,
            'SVHN Loss': total_svhn_loss,
            'CIFAR10 Loss': total_cifar10_loss,
        })
        """

        model.eval()

        def tester(loader, group):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output = model(images)
                preds = F.log_softmax(class_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total * 100
            log_data = {
                f'Test/Label {group} Accuracy': accuracy,
            }
            wandb.log(log_data, step=epoch + 1)

            print(f'Test {group} | Label Acc: {accuracy:.3f}%')

        with torch.no_grad():
            tester(MNIST_loader_test, 'MNIST')
            tester(SVHN_loader_test, 'SVHN')
            tester(CIFAR10_loader_test, 'CIFAR')

if __name__ == '__main__':
    main()
