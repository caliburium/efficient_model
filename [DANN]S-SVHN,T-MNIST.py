import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from functions.lr_lambda import lr_lambda
import wandb
from dataloader.data_loader import data_loader
from model.DANN import DANN, dann_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=4096)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--fc_weight', type=float, default=1.0)
    parser.add_argument('--disc_weight', type=float, default=1.0)

    args = parser.parse_args()
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(entity="hails",
               project="Efficient Model",
               config=args.__dict__,
               name="[DANN]S:SVHN/T:MNIST_" + str(args.hidden_size)
                    + "_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
                    + "_fc:" + str(args.fc_weight) + "_disc:" + str(args.disc_weight)
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)

    print("Data load complete, start training")

    model = DANN(hidden_size=args.hidden_size).to(device)
    param = dann_weights(model, args.lr, args.feature_weight, args.fc_weight, args.disc_weight)
    optimizer = optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_mnist_domain_loss, total_svhn_domain_loss, total_domain_loss = 0, 0, 0
        total_mnist_domain_correct, total_svhn_domain_correct = 0, 0
        total_label_loss = 0
        total_mnist_correct, total_svhn_correct = 0, 0
        total_loss, total_samples = 0, 0

        for i, (mnist_data, svhn_data) in enumerate(zip(mnist_loader, svhn_loader)):
            p = epoch / num_epochs
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            mnist_images, mnist_labels = mnist_data
            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            svhn_images, svhn_labels = svhn_data
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            svhn_dlabels = torch.full((svhn_images.size(0),), 1, dtype=torch.long, device=device)

            optimizer.zero_grad()

            mnist_out, mnist_domain_out = model(mnist_images, alpha=lambda_p)
            svhn_out, svhn_domain_out = model(svhn_images, alpha=lambda_p)

            label_loss = criterion(svhn_out, svhn_labels)

            mnist_domain_loss = criterion(mnist_domain_out, mnist_dlabels)
            svhn_domain_loss = criterion(svhn_domain_out, svhn_dlabels)
            domain_loss = mnist_domain_loss + svhn_domain_loss

            loss = label_loss + domain_loss

            loss.backward()
            optimizer.step()

            total_label_loss += label_loss.item()
            total_loss += loss.item()

            total_domain_loss += domain_loss.item()
            total_mnist_domain_loss += mnist_domain_loss.item()
            total_svhn_domain_loss += svhn_domain_loss.item()

            total_mnist_correct += (torch.argmax(mnist_out, dim=1) == mnist_labels).sum().item()
            total_svhn_correct += (torch.argmax(svhn_out, dim=1) == svhn_labels).sum().item()

            total_mnist_domain_correct += (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
            total_svhn_domain_correct += (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()

            total_samples += svhn_labels.size(0)

        scheduler.step()

        current_lr_feature = optimizer.param_groups[0]['lr']
        current_lr_classifier = optimizer.param_groups[1]['lr']
        current_lr_discriminator = optimizer.param_groups[2]['lr']

        mnist_domain_avg_loss = total_mnist_domain_loss / total_samples
        svhn_domain_avg_loss = total_svhn_domain_loss / total_samples
        domain_avg_loss = total_domain_loss / (total_samples * 2)

        label_avg_loss = total_label_loss / total_samples
        total_avg_loss = total_loss / total_samples

        mnist_acc_epoch = total_mnist_correct / total_samples * 100
        svhn_acc_epoch = total_svhn_correct / total_samples * 100

        mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples * 100
        svhn_domain_acc_epoch = total_svhn_domain_correct / total_samples * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'Label Loss: {label_avg_loss:.8f} | '
              f'Domain Loss: {domain_avg_loss:.8f} | '
              f'Total Loss: {total_avg_loss:.8f} | '
        )
        print(f'MNIST Domain Loss: {mnist_domain_avg_loss:.8f} | '
              f'SVHN Domain Loss: {svhn_domain_avg_loss:.8f} | '
        )
        print(f'MNIST Acc: {mnist_acc_epoch:.3f}% | '
              f'SVHN Acc: {svhn_acc_epoch:.3f}% | '
              f'MNIST Domain Acc: {mnist_domain_acc_epoch:.3f}% | '
              f'SVHN Domain Acc: {svhn_domain_acc_epoch:.3f}% | '
              )

        wandb.log({
            'Train/Label Loss': label_avg_loss,
            'Train/Domain MNIST Loss': mnist_domain_avg_loss,
            'Train/Domain SVHN Loss': svhn_domain_avg_loss,
            'Train/Domain Loss': domain_avg_loss,
            'Train/Total Loss': total_avg_loss,
            'Train/MNIST Label Accuracy': mnist_acc_epoch,
            'Train/SVHN Label Accuracy': svhn_acc_epoch,
            'Train/MNIST Domain Accuracy': mnist_domain_acc_epoch,
            'Train/SVHN Domain Accuracy': svhn_domain_acc_epoch,
            'Learning Rate/Feature Extractor': current_lr_feature,
            'Learning Rate/Classifier': current_lr_classifier,
            'Learning Rate/Discriminator': current_lr_discriminator,
        }, step=epoch + 1)

        model.eval()

        def tester(loader, group, domain_label):
            label_correct, domain_correct, total = 0, 0, 0

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                class_output, domain_output = model(images, alpha=0.0)
                label_correct += (torch.argmax(class_output, dim=1) == labels).sum().item()
                domain_correct += (torch.argmax(domain_output, dim=1) == domain_label).sum().item()
                total += images.size(0)

            label_acc = label_correct / total * 100
            domain_acc = domain_correct / total * 100

            wandb.log({
                f'Test/Label {group} Accuracy': label_acc,
                f'Test/Domain {group} Accuracy': domain_acc,
            }, step=epoch + 1)

            print(f'Test {group} | Label Acc: {label_acc:.3f}% | Domain Acc: {domain_acc:.3f}%')

        with torch.no_grad():
            tester(mnist_loader_test, 'MNIST', 0)
            tester(svhn_loader_test, 'SVHN', 1)

if __name__ == '__main__':
    main()
