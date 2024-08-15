import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from functions import ReverseLayerF, lr_lambda
from dataloader.pacs_loader import pacs_loader
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

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
