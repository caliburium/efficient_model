import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import wandb
from tqdm import tqdm
import deeplake

from functions.lr_lambda import lr_lambda
from dataloader.pacs_loader import pacs_loader
from model.AlexNetCaffe import AlexNetCaffe
from model.Discriminator import Discriminator
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
    num_classes = args.num_classes
    num_domains = args.num_domains

    # Initialize Weights and Biases
    wandb.init(project="EM_Domain",
               entity="hails",
               config=args.__dict__,
               name="MMLD_PACS_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    pacs_train = deeplake.load("hub://activeloop/pacs-train")
    pacs_test = deeplake.load("hub://activeloop/pacs-test")

    # Photo, Art_Painting, Cartoon, Sketch
    photo_loader_train, photo_loader_test = pacs_loader(True, 0, args.batch_size, pacs_train, pacs_test)
    art_loader_train, art_loader_test = pacs_loader(True, 1, args.batch_size, pacs_train, pacs_test)
    cartoon_loader_train, cartoon_loader_test = pacs_loader(True, 2, args.batch_size, pacs_train, pacs_test)
    sketch_loader_train, sketch_loader_test = pacs_loader(True, 3, args.batch_size, pacs_train, pacs_test)
    train_loader = [photo_loader_train, art_loader_train, cartoon_loader_train, sketch_loader_train]

    print("Data load complete, start training")

    model = AlexNetCaffe(num_classes=num_classes).to(device)
    kmeans = KMeansTorch(num_clusters=num_domains, device=device)
    discriminator = Discriminator(num_domains=num_domains).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Meta-learning stage inside main
    for epoch in range(num_epochs):
        model.train()
        meta_gradient = [torch.zeros_like(param) for param in model.parameters()]

        loss_label_epoch = 0
        loss_domain_epoch = 0

        for domain_idx, data_loader in enumerate(train_loader):
            domain_gradient = [torch.zeros_like(param) for param in model.parameters()]

            for i, train_data in enumerate(tqdm(data_loader)):
                p = (float(i + epoch * len(data_loader)) / num_epochs / len(data_loader))
                lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

                images, labels, _ = train_data
                images, labels = images.to(device), labels.to(device)

                with autocast():
                    fcl, label_pred = model(images)
                    kmeans_pred = kmeans.fit(fcl)
                    dis_pred = discriminator(fcl, lambda_p)

                    loss_l = criterion(label_pred, labels.squeeze())
                    loss_d = criterion(dis_pred, kmeans_pred)
                    loss = loss_l + loss_d

                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # Collect gradients for meta-update
                for j, param in enumerate(model.parameters()):
                    domain_gradient[j] += param.grad.data

                loss_label_epoch += loss_l.item()
                loss_domain_epoch += loss_d.item()

            # Normalize domain gradients and add to meta gradients
            domain_gradient = [g / len(data_loader) for g in domain_gradient]
            for j, g in enumerate(meta_gradient):
                meta_gradient[j] += domain_gradient[j]

        meta_gradient = [torch.zeros_like(param) for param in model.parameters()]

        # Meta-update using the accumulated gradients
        optimizer.zero_grad()
        for j, param in enumerate(model.parameters()):
            if param.grad is not None:
                param.grad.data = meta_gradient[j] / len(train_loader)

        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        # Log losses to wandb
        wandb.log({
            'Label Loss': loss_label_epoch / len(train_loader),
            'Domain Loss': loss_domain_epoch / len(train_loader),
            'Total Loss': (loss_label_epoch + loss_domain_epoch) / len(train_loader),
            'Epoch': epoch + 1
        })

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Label Loss: {loss_label_epoch / len(train_loader):.4f}, '
              f'Domain Loss: {loss_domain_epoch / len(train_loader):.4f}, '
              f'Total Loss: {(loss_label_epoch + loss_domain_epoch) / len(train_loader):.4f}')

        model.eval()

        def lc_tester(loader, group):
            correct, total = 0, 0
            for image, label, _ in loader:
                image = image.to(device)
                label = label.to(device).long()  # Ensure label is LongTensor

                _, class_output = model(image)
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

                fcl, _ = model(image)
                domain_output = discriminator(fcl, lambda_p=1.0)
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
