import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
from dataloader.data_loader import data_loader
from model.SimpleCNN import CNN32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    num_epochs = args.epoch
    # Initialize Weights and Biases
    wandb.init(project="Efficient Model - MetaLearning & Domain Adaptation",
               entity="hails",
               config=args.__dict__,
               name="[CNN]_2Source_SVHN&MNIST" + "_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)

    print("Data load complete, start training")

    model = CNN32().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        i = 0
        loss_svhn = 0
        loss_mnist = 0
        loss_total = 0

        for svhn_data, mnist_data in zip(mnist_loader, tqdm(svhn_loader)):
            svhn_images, svhn_labels = svhn_data
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            mnist_images, mnist_labels = mnist_data
            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)

            _, svhn_outputs = model(svhn_images)
            svhn_loss = criterion(svhn_outputs, svhn_labels)

            _, mnist_outputs = model(mnist_images)
            mnist_loss = criterion(mnist_outputs, mnist_labels)

            loss = svhn_loss + mnist_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_mnist += mnist_loss.item()
            loss_svhn += svhn_loss.item()
            loss_total += loss.item()

            i += 1

        print(f'Epoch [{epoch + 1}/{num_epochs}], SVHN Loss: {loss_svhn:.4f}, '
              f'MNIST Loss: {loss_mnist:.4f}, Total Loss: {loss_total:.4f}')
        wandb.log({
            'SVHN Loss': loss_svhn,
            'MNIST Loss': loss_mnist,
            'Total Loss': loss_total
        })

        model.eval()

        def tester(loader, group):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                _ , class_output = model(images)
                preds = F.log_softmax(class_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            wandb.log({group + ' Accuracy': accuracy}, step=epoch + 1)
            print(group + f' Accuracy: {accuracy * 100:.3f}%')

        with torch.no_grad():
            tester(svhn_loader_test, 'SVHN')
            tester(mnist_loader_test, 'MNIST')

if __name__ == '__main__':
    main()
