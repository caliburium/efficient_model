import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from dataloader.data_loader import data_loader
from model.AlexNet import AlexNet32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    num_epochs = args.epoch
    # Initialize Weights and Biases
    wandb.init(project="Efficient Model - MetaLearning & Domain Adaptation",
               entity="hails",
               config=args.__dict__,
               name="[Alex]SingleTask_MNIST/SVHN_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    # domain 'train' = artpaintings, cartoon, sketch
    SVHN_loader, SVHN_loader_test = data_loader('SVHN', args.batch_size)
    _, MNIST_loader_test = data_loader('MNIST', args.batch_size)

    print("Data load complete, start training")

    model = AlexNet32(pretrained=True, num_class=10)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        i = 0
        total_loss = 0

        for svhn_images, svhn_labels in tqdm(SVHN_loader):
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)

            svhn_outputs = model(svhn_images)
            loss = criterion(svhn_outputs, svhn_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            i += 1

        print(
            f'Epoch [{epoch + 1}/{num_epochs}],SVHN Loss: {total_loss:.4f}')

        wandb.log({
            'SVHN Loss': total_loss,
        })

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

            accuracy = correct / total
            wandb.log({group + ' Accuracy': accuracy}, step=epoch + 1)
            print(group + f' Accuracy: {accuracy * 100:.3f}%')

        with torch.no_grad():
            tester(SVHN_loader_test, 'SVHN')
            tester(MNIST_loader_test, 'MNIST')

if __name__ == '__main__':
    main()
