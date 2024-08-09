import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import wandb
from tqdm import tqdm
import deeplake

from functions.lr_lambda import lr_lambda
from dataloader.pacs_loader import pacs_loader
from model import AlexNetCaffe, Discriminator
from clustering.kmeans_torch import KMeansTorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--num_domains', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="EM_Domain",
               entity="hails",
               config=args.__dict__,
               name="MMLD_PACS_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    pacs_train = deeplake.load("hub://activeloop/pacs-train")
    pacs_test = deeplake.load("hub://activeloop/pacs-test")

    # Photo, Art_Painting, Cartoon, Sketch
    _, photo_loader_test = pacs_loader(True, 0, args.batch_size, pacs_train, pacs_test)
    _, art_loader_test = pacs_loader(True, 1, args.batch_size, pacs_train, pacs_test)
    _, cartoon_loader_test = pacs_loader(True, 2, args.batch_size, pacs_train, pacs_test)
    _, sketch_loader_test = pacs_loader(True, 3, args.batch_size, pacs_train, pacs_test)
    train_loader, _ = pacs_loader(False, 0, args.batch_size, pacs_train, pacs_test)

    print("Data load complete, start training")

    model = AlexNetCaffe(num_classes=args.numclasses).to(device)
    kmeans = KMeansTorch(num_clusters=args.num_domains, device=device)
    discriminator = Discriminator(num_domains=args.num_domains).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        i = 0

        loss_domain_epoch = 0
        loss_label_epoch = 0

        for train_data in tqdm(train_loader):
            p = (float(i + epoch * len(train_data)) / num_epochs / len(train_data))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            images, labels, _ = train_data
            images, labels = images.to(device), labels.to(device)

            fcl, label_pred = model(images)
            kmeans_pred = kmeans.fit(fcl)
            dis_pred = discriminator(fcl, lambda_p)

            loss_l = criterion(label_pred, labels.squeeze())
            loss_d = criterion(dis_pred, kmeans_pred)
            loss = loss_l + loss_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_label_epoch += loss_l.item()
            loss_domain_epoch += loss_d.item()

            i += 1

        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Label Loss: {loss_label_epoch:.4f}, '
              f'Domain Loss: {loss_domain_epoch:.4f}, '
              f'Total Loss: {loss_label_epoch + loss_domain_epoch:.4f}, '
              )

        wandb.log({
            'Label Loss': loss_label_epoch,
            'Domain Loss': loss_domain_epoch,
            'Total Loss': loss_label_epoch + loss_domain_epoch,
        })

        model.eval()

        def lc_tester(loader, group):
            correct, total = 0, 0
            for image, label, _ in loader:
                image = image.to(device)
                label = label.to(device).long()  # Ensure label is LongTensor

                class_output, _ = model(image, alpha=0.0)
                preds = F.log_softmax(class_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            accuracy = correct / total
            wandb.log({'[Label] ' + group + ' Accuracy': accuracy}, step=epoch + 1)
            print('[Label] ' + group + f' Accuracy: {accuracy * 100:.3f}%')

        def dc_tester(loader, group):
            correct, total = 0, 0
            for image, _, domain in loader:
                image = image.to(device)
                domain = domain.to(device).long()  # Ensure domain is LongTensor

                _, domain_output = model(image, alpha=0.0)
                preds = F.log_softmax(domain_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += domain.size(0)
                correct += (predicted == domain).sum().item()

            accuracy = correct / total
            wandb.log({'[Domain] ' + group + ' Accuracy': accuracy}, step=epoch + 1)
            print('[Domain] ' + group + f' Accuracy: {accuracy * 100:.3f}%')

        with torch.no_grad():
            lc_tester(photo_loader_test, 'Photo')
            lc_tester(art_loader_test, 'Art')
            lc_tester(cartoon_loader_test, 'Cartoon')
            lc_tester(sketch_loader_test, 'Sketch')
            dc_tester(photo_loader_test, 'Photo')
            dc_tester(art_loader_test, 'Art')
            dc_tester(cartoon_loader_test, 'Cartoon')
            dc_tester(sketch_loader_test, 'Sketch')


if __name__ == '__main__':
    main()
