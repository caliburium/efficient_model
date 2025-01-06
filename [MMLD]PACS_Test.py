import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import numpy as np
from tqdm import tqdm
from functions import nmi
from functions.KMeans import KMeans
from dataloader.pacs_loader import pacs_loader
from model.AlexNet import DANN_Alex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    num_epochs = args.epoch
    # Initialize Weights and Biases
    wandb.init(project="Efficient Model - MetaLearning & Domain Adaptation",
               entity="hails",
               config=args.__dict__,
               name="[MMLD]PACS_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    # domain 'train' = artpaintings, cartoon, sketch
    source_loader = pacs_loader(split='train', domain='train', batch_size=args.batch_size)
    art_loader = pacs_loader(split='test', domain='artpaintings', batch_size=args.batch_size)
    cartoon_loader = pacs_loader(split='test', domain='cartoon', batch_size=args.batch_size)
    sketch_loader = pacs_loader(split='test', domain='sketch', batch_size=args.batch_size)
    target_loader = pacs_loader(split='test', domain='photo', batch_size=args.batch_size)

    print("Data load complete, start training")

    model = DANN_Alex(pretrained=True).to(device)
    kmeans = KMeans(n_clusters=3, device=device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=24, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        loss_total = 0
        i = 0

        for source_images, _, _ in source_loader:
            conv_features = model.conv_features(source_images)
            merged_features = torch.cat(conv_features, dim=1)
            cluster_labels = kmeans.fit(merged_features)


        for source_images, source_labels, _ in tqdm(source_loader):
            p = (float(i + epoch * len(source_loader)) / num_epochs / len(source_loader))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            source_images, source_labels = source_images.to(device), source_labels.to(device)
            label_out, _ = model(source_images, lambda_p=lambda_p)



            classification_loss = criterion(label_out, source_labels)

            cluster_labels = cluster_labels.to(device)
            clustering_loss = criterion(domain_out, cluster_labels)
            total_loss = classification_loss + clustering_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {loss_total:.4f}')
        wandb.log({'Total Loss': loss_total})

        model.eval()

        # Scheduler step
        scheduler.step()

        # Evaluate the model
        def tester(loader, group):
            correct, total = 0, 0
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device)
                label_out, _ = model(images, lambda_p=0.0)
                preds = F.log_softmax(label_out, dim=1)
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            wandb.log({f"{group} Accuracy": accuracy}, step=epoch + 1)
            print(f"{group} Accuracy: {accuracy * 100:.3f}%")

        with torch.no_grad():
            tester(art_loader, 'Art Paintings')
            tester(cartoon_loader, 'Cartoon')
            tester(sketch_loader, 'Sketch')
            tester(target_loader, 'Photo')


if __name__ == '__main__':
    main()
