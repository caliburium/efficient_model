import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloader.data_loader import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, num_classes=10, hidden_size=4096):
        super(MLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(3 * 32 * 32, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--hidden_size', type=int, default=4096)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    parser.add_argument('--lr_alpha', type=float, default=0.1)
    parser.add_argument('--lr_beta', type=float, default=0.25)
    parser.add_argument('--data', type=str, default='SVHN')
    args = parser.parse_args()

    num_epochs = args.epoch
    dataset = args.data

    # Initialize Weights and Biases
    wandb.init(entity="hails",
               project="Efficient Model - Partition",
               config=args.__dict__,
               name="[MLP]" + str(args.data)
                    + "_" + str(args.hidden_size)
                    + "_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
               )

    if args.data == 'MNIST' :
        loader, test_loader = data_loader('MNIST', args.batch_size)
    elif args.data == 'SVHN' :
        loader, test_loader = data_loader('SVHN', args.batch_size)
    elif args.data == 'CIFAR' :
        loader, test_loader = data_loader('CIFAR10', args.batch_size)
    else :
        print('[Error] Wrong data type')

    print("Data load complete, start training")

    def lr_lambda(progress):
        alpha = args.lr_alpha
        beta = args.lr_beta
        return (1 + alpha * progress) ** (-beta)

    model = MLP(num_classes=10, hidden_size=args.hidden_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(num_epochs):
        model.train()
        i = 0

        total_loss, total_correct, total_samples = 0, 0, 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

            correct = (torch.argmax(outputs, dim=1) == labels).sum().item()

            total_correct += correct
            total_samples += labels.size(0)

            i += 1

        scheduler.step()

        acc_epoch = total_correct / total_samples * 100
        total_avg_loss = total_loss / total_samples

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], | '
            f'Total Loss: {total_avg_loss:.4f}, | '
            f'{dataset} Acc: {acc_epoch:.3f}% | '
        )

        wandb.log({
            f'Train/{dataset} Label Loss': total_avg_loss,
            f'Train/{dataset} Label Accuracy': acc_epoch,
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
            wandb.log({f'Test/{group} Label Accuracy': accuracy}, step=epoch)
            print(f'Test {group} | Label Acc: {accuracy:.3f}%')

        with torch.no_grad():
            tester(test_loader, dataset)

if __name__ == '__main__':
    main()
