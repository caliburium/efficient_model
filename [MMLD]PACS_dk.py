import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import numpy as np
from tqdm import tqdm
from functions.EntropyLoss import HLoss
from functions.KMeans import KMeans
from dataloader.pacs_loader import pacs_loader
from functions.nmi import normalized_mutual_info
from model.AlexNet import DANN_Alex, get_model_parts_with_weights, get_model_parts_with_weights_with_lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# https://github.com/mil-tokyo/dg_mmld/tree/master

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=200)

    # Dataset Settings
    parser.add_argument('--color_jitter', action='store_false')
    parser.add_argument('--min_scale', type=float, default=0.8)

    # K-Means Settings
    parser.add_argument('--num_clustering', type=int, default=3)

    # Model Weights Tuning
    parser.add_argument('--fc_weight', type=float, default=15.0)
    parser.add_argument('--disc_weight', type=float, default=15.0)

    # Optimizer Settings
    # parser.add_argument('--lr', type=float, default=1e-3) #TODO: 너무 큰가? 한번 돌면 weight 가 1e17 막 이렇게 뜸
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--nesterov', action='store_true')

    # Schedular Settings
    parser.add_argument('--lr_step', type=int, default=24)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1) #TODO: 너무 크다

    # Grad Reverse Layer
    parser.add_argument('--grl_weight', type=float, default=1.0)
    parser.add_argument('--entropy_weight', type=float, default=1.0)

    args = parser.parse_args()

    num_epochs = args.epoch
    grl_weight = args.grl_weight
    entropy_weight = args.entropy_weight

    # Initialize Weights and Biases
    wandb.init(project="Efficient Model DK",
               entity="hails",
               config=args.__dict__,
               name="[MMLD]PACS_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    # domain 'train' = artpaintings, cartoon, sketch
    source_loader = pacs_loader(split='train', domain='train', batch_size=args.batch_size,
                                min_scale=args.min_scale, color_jitter=args.color_jitter)
    art_loader = pacs_loader(split='test', domain='artpaintings', batch_size=args.batch_size)
    cartoon_loader = pacs_loader(split='test', domain='cartoon', batch_size=args.batch_size)
    sketch_loader = pacs_loader(split='test', domain='sketch', batch_size=args.batch_size)
    target_loader = pacs_loader(split='test', domain='photo', batch_size=args.batch_size)

    print("Data load complete, start training")

    kmeans = KMeans(n_clusters=args.num_clustering, device=device)

    model = DANN_Alex(pretrained=True, num_domain=args.num_clustering).to(device)
    params = get_model_parts_with_weights_with_lr(model, fc_weight=args.fc_weight, disc_weight=args.disc_weight, lr=args.lr)
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay_gamma)
    # inv_lr_scheduler랑 ExponentialLR 옵션이 있긴함. utils/scheduler.py 참고
    all_labels = []
    all_domains = []
    for _, labels, domains in source_loader:
        all_labels.append(labels)
        all_domains.append(domains)

    # Concatenate all labels and domains into a single tensor
    true_labels = torch.cat(all_labels).to(device)
    domain_labels = torch.cat(all_domains).to(device)

    for epoch in range(num_epochs):
        model.eval()
        all_features = []

        for source_images, _, _ in source_loader:
            with torch.no_grad():
                source_images = source_images.to(device)
                conv_features = model.conv_features(source_images)

                batch_features = []
                for feats in conv_features:
                    feat_mean = feats.mean(dim=[2, 3])
                    feat_std = feats.std(dim=[2, 3])
                    batch_features.append(torch.cat((feat_mean, feat_std), dim=1))

                merged_features = torch.cat(batch_features, dim=1)
                all_features.append(merged_features)

        all_features = torch.cat(all_features, dim=0).to(device)
        clustered_labels = kmeans.fit(all_features).to(device)
        print("Cluster finished")

        # true_labels = torch.tensor(source_loader.dataset.label, dtype=torch.long, device=device)
        # domain_labels = torch.tensor(source_loader.dataset.domain, dtype=torch.long, device=device)
        cluster_before = getattr(source_loader.dataset, 'cluster_before', None)

        class_nmi = normalized_mutual_info(clustered_labels, true_labels)
        domain_nmi = normalized_mutual_info(clustered_labels, domain_labels)
        before_nmi = None
        if cluster_before is not None:
            before_nmi = normalized_mutual_info(clustered_labels,
                                                torch.tensor(cluster_before, dtype=torch.long, device=device))

        print(
            f"NMI (Class): {class_nmi:.4f}, NMI (Domain): {domain_nmi:.4f}, NMI (Before): {before_nmi if before_nmi else 'N/A'}")

        wandb_log_data = {
            "Epoch": epoch + 1,
            "NMI/Class": class_nmi,
            "NMI/Domain": domain_nmi,
        }
        if before_nmi is not None:
            wandb_log_data["NMI/Before"] = before_nmi
        wandb.log(wandb_log_data)

        # Data Imbalance 잡는 용도로 보임.
        weight = 1. / torch.bincount(clustered_labels, minlength=args.num_clustering)
        weight = (weight / weight.sum() * args.num_clustering).to(device)
        print(f"Calculated weight: {weight}")

        class_criterion = nn.CrossEntropyLoss()
        entropy_criterion = HLoss()
        domain_criterion = nn.CrossEntropyLoss(weight=weight)

        # p = (float(i + epoch * len(source_loader)) / num_epochs / len(source_loader))
        p = epoch / num_epochs
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * grl_weight
        beta = (2. / (1. + np.exp(-10 * p)) - 1) * entropy_weight

        model.train()
        running_loss_class, running_loss_domain, running_loss_entropy = 0.0, 0.0, 0.0
        running_correct_class, running_correct_domain = 0, 0

        for batch_idx, (images, labels, indices) in enumerate(tqdm(source_loader)):
            images, labels = images.to(device), labels.to(device)

            batch_domains = clustered_labels[indices]

            label_out, domain_out = model(images, lambda_p=alpha)
            _, pred_class = torch.max(label_out, 1)
            _, pred_domain = torch.max(domain_out, 1)

            loss_class = class_criterion(label_out, labels)
            loss_domain = domain_criterion(domain_out, batch_domains)
            loss_entropy = entropy_criterion(label_out)

            total_loss = loss_class + loss_domain + loss_entropy * beta

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss_class += loss_class.item() * images.size(0)
            running_loss_domain += loss_domain.item() * images.size(0)
            running_loss_entropy += loss_entropy.item() * images.size(0)
            running_correct_class += (pred_class == labels).sum().item()
            running_correct_domain += (pred_domain == batch_domains).sum().item()

            log_batch = (
                f"Train: Batch: {batch_idx} | "
                f"Loss Class: {loss_class:.4f} | "
                f"Loss Domain: {loss_domain:.4f} | "
                f"Loss Entropy: {loss_entropy:.4f}"
            )
            print(log_batch)

        epoch_loss_class = running_loss_class / len(source_loader.dataset)
        epoch_loss_domain = running_loss_domain / len(source_loader.dataset)
        epoch_loss_entropy = running_loss_entropy / len(source_loader.dataset)
        epoch_acc_class = running_correct_class / len(source_loader.dataset)
        epoch_acc_domain = running_correct_domain / len(source_loader.dataset)

        log = (
            f"Train: Epoch: {epoch} | "
            f"Alpha: {alpha:.4f} | "
            f"Loss Class: {epoch_loss_class:.4f} | "
            f"Loss Domain: {epoch_loss_domain:.4f} | "
            f"Loss Entropy: {epoch_loss_entropy:.4f}"
            f"Acc Class: {epoch_acc_class:.4f} | "
            f"Acc Domain: {epoch_acc_domain:.4f} | "
        )
        print(log)
        wandb.log({
            "Train/Accuracy/Class": epoch_acc_class,
            "Train/Accuracy/Domain": epoch_acc_domain,
            "Train/Alpha": alpha,
            "Train/Loss/Class": epoch_loss_class,
            "Train/Loss/Domain": epoch_loss_domain,
            "Train/Loss/Entropy": epoch_loss_entropy,
            "Train/Loss/Total": epoch_loss_class + epoch_loss_domain + epoch_loss_entropy * beta,
        })

        model.eval()

        # Scheduler step
        scheduler.step()

        # Evaluate the model
        def tester(loader, group):
            label_correct, total = 0, 0
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device)
                label_out, _ = model(images, lambda_p=0.0)
                label_preds = F.log_softmax(label_out, dim=1)
                _, predicted_labels = torch.max(label_preds.data, 1)
                total += labels.size(0)
                label_correct += (predicted_labels == labels).sum().item()

            label_acc = label_correct / total
            wandb.log({f"{group} Label Accuracy": label_acc}, step=epoch)
            print(f"{group} Label Accuracy: {label_acc * 100:.3f}%")


        with torch.no_grad():
            tester(art_loader, 'Art Paintings')
            tester(cartoon_loader, 'Cartoon')
            tester(sketch_loader, 'Sketch')
            tester(target_loader, 'Photo')


if __name__ == '__main__':
    main()
