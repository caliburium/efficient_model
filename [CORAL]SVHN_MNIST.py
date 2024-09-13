import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
from functions.coral_loss import coral_loss
from dataloader.data_loader import data_loader
from model.SimpleCNN import CNN32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--source', type=str, default='SVHN')
    parser.add_argument('--target', type=str, default='MNIST')
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    num_epochs = args.epoch
    # Initialize Weights and Biases
    wandb.init(project="Efficient Model - MetaLearning & Domain Adaptation",
               entity="hails",
               config=args.__dict__,
               name="[CORAL]_S:" + args.source + "_T:" + args.target
                    + "_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    source_loader, source_loader_test = data_loader(args.source, args.batch_size)
    target_loader, target_loader_test = data_loader(args.target, args.batch_size)

    print("Data load complete, start training")

    model = CNN32(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        i = 0
        loss_classification = 0
        loss_coral = 0
        loss_total = 0

        for source_data, target_data in zip(source_loader, tqdm(target_loader)):
            source_images, source_labels = source_data
            source_images, source_labels = source_images.to(device), source_labels.to(device)
            target_images, _ = target_data
            target_images = target_images.to(device)

            # Forward pass for source domain (SVHN)
            _, source_outputs = model(source_images)
            classification_loss = criterion(source_outputs, source_labels)

            # Forward pass for target domain (MNIST)
            _, target_outputs = model(target_images)
            coral_loss_value = coral_loss(source_outputs, target_outputs)

            # Total loss (classification loss + CORAL loss)
            total_loss = classification_loss + coral_loss_value

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_classification += classification_loss.item()
            loss_coral += coral_loss_value.item()
            loss_total += total_loss.item()

            i += 1

        print(f'Epoch [{epoch + 1}/{num_epochs}], Classification Loss: {loss_classification:.4f}, '
              f'Coral Loss: {loss_coral:.4f}, Total Loss: {loss_total:.4f}')
        wandb.log({
            'Classification Loss': loss_classification,
            'Coral Loss': loss_coral,
            'Total Loss': loss_total,
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
            tester(source_loader_test, 'SVHN')
            tester(target_loader_test, 'MNIST')

if __name__ == '__main__':
    main()
