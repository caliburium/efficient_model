import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functions.lr_lambda import *
import wandb
import numpy as np
from tqdm import tqdm
from dataloader.pacs_loader import pacs_loader
from trash.AlexNetCaffe import AlexNetCaffe228
from model.SimpleCNN import CNN228
from model.Discriminator import Discriminator224

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--pretrain_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--model', type=str, default='alex') # cnn/alex
    parser.add_argument('--lr_cls', type=float, default=0.01)
    parser.add_argument('--lr_dom', type=float, default=0.01)
    args = parser.parse_args()

    pre_epochs = args.pretrain_epoch
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="Efficient Model - MetaLearning & Domain Adaptation",
               entity="hails",
               config=args.__dict__,
               name="[DANN]PACS_S:Photo_" + args.model +
                    "lr(cls):" + str(args.lr_cls) + "_lr(dom)" + str(args.lr_dom)
                    + "_Batch:" + str(args.batch_size)
               )

    # Photo, Art_Painting, Cartoon, Sketch
    source_loader = pacs_loader(split='train', domain='train', batch_size=args.batch_size)
    photo_train_loader = pacs_loader(split='train', domain='photo', batch_size=args.batch_size)
    art_loader = pacs_loader(split='test', domain='artpaintings', batch_size=args.batch_size)
    cartoon_loader = pacs_loader(split='test', domain='cartoon', batch_size=args.batch_size)
    sketch_loader = pacs_loader(split='test', domain='sketch', batch_size=args.batch_size)
    photo_loader = pacs_loader(split='test', domain='photo', batch_size=args.batch_size)

    print("Data load complete, start training")

    discriminator = Discriminator224(num_domains=4).to(device)
    if args.model == 'cnn':
        model = CNN228(num_classes=7).to(device)
        optimizer_cls = optim.SGD(list(model.feature_extractor.parameters())
                                  + list(model.classifier.parameters()), lr=args.lr_cls, momentum=0.9,
                                  weight_decay=1e-6)
        optimizer_dom = optim.SGD(list(model.feature_extractor.parameters())
                                  + list(discriminator.discriminator.parameters()), lr=args.lr_dom, momentum=0.9,
                                  weight_decay=1e-6)
    elif args.model == 'alex':
        model = AlexNetCaffe228(num_classes=7).to(device)
        optimizer_cls = optim.SGD(list(model.parameters()), lr=args.lr_cls, momentum=0.9, weight_decay=1e-6)
        optimizer_dom = optim.SGD(list(model.features.parameters()) + list(discriminator.parameters())
                                  , lr=args.lr_dom, momentum=0.9, weight_decay=1e-6)
    pre_opt = optim.Adam(model.parameters(), lr=1e-4)

    scheduler_cls = optim.lr_scheduler.LambdaLR(optimizer_cls, lr_lambda)
    scheduler_dom = optim.lr_scheduler.LambdaLR(optimizer_dom, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(pre_epochs):
        model.train()
        i = 0

        for source_data in photo_train_loader:
            # Training with source data
            source_images, source_labels, _ = source_data
            source_images, source_labels = source_images.to(device), source_labels.to(device)

            _, class_output = model(source_images, out='feature')
            loss = criterion(class_output, source_labels)

            pre_opt.zero_grad()
            loss.backward()
            pre_opt.step()

            label_acc = (torch.argmax(class_output, dim=1) == source_labels).sum().item() / source_labels.size(0)

            print(f'Batches [{i + 1}/{len(source_loader)}], '
                  f'Pretrain Loss: {loss.item():.4f}, '
                  f'Pretrain Accuracy: {label_acc * 100:.3f}%, '
                  )

            i += 1

        model.eval()

    print("Pretrain Finished")

    for epoch in range(num_epochs):
        model.train()
        i = 0

        loss_tgt_domain_epoch = 0
        loss_src_domain_epoch = 0
        loss_label_epoch = 0

        for source_data, target_data in zip(photo_train_loader, tqdm(source_loader)):
            p = (float(i + epoch * min(len(source_loader), len(photo_train_loader))) /
                 num_epochs / min(len(source_loader), len(photo_train_loader)))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            source_images, source_labels, source_domain = source_data
            source_images, source_labels, source_domain = source_images.to(device), source_labels.to(device), source_domain.to(device)

            target_images, _, target_domain = target_data
            target_images, target_domain = target_images.to(device), target_domain.to(device)

            optimizer_cls.zero_grad()
            optimizer_dom.zero_grad()

            source_feature, source_class_out = model(source_images, out='feature')
            source_domain_out = discriminator(source_feature, lambda_p=lambda_p)
            target_feature, _ = model(target_images, out='feature')
            target_domain_out = discriminator(target_feature, lambda_p=lambda_p)

            label_src_loss = criterion(source_class_out, source_labels)
            domain_src_loss = criterion(source_domain_out, source_domain)
            domain_tgt_loss = criterion(target_domain_out, target_domain)

            loss_label_epoch += label_src_loss.item()
            loss_src_domain_epoch += domain_src_loss.item()
            loss_tgt_domain_epoch += domain_tgt_loss.item()

            loss = label_src_loss + domain_src_loss + domain_tgt_loss

            loss.backward()
            optimizer_cls.step()
            optimizer_dom.step()

            i += 1

        scheduler_cls.step()
        scheduler_dom.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Domain source Loss: {loss_src_domain_epoch:.4f}, '
              f'Domain target Loss: {loss_tgt_domain_epoch:.4f}, '
              f'Label Loss: {loss_label_epoch:.4f}, '
              f'Total Loss: {loss_src_domain_epoch + loss_tgt_domain_epoch + loss_label_epoch:.4f}, '
              )

        wandb.log({
            'Domain source Loss': loss_src_domain_epoch,
            'Domain target Loss': loss_tgt_domain_epoch,
            'Label Loss': loss_label_epoch,
            'Total Loss': loss_src_domain_epoch + loss_tgt_domain_epoch + loss_label_epoch,
        })

        model.eval()

        def lc_tester(loader, group):
            correct, total = 0, 0
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device)

                _, class_output = model(images, out='feature')
                preds = F.log_softmax(class_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            wandb.log({group + ' Accuracy': accuracy}, step=epoch + 1)
            print(group + f' Accuracy: {accuracy * 100:.3f}%')

        def dc_tester(loader, group):
            correct, total = 0, 0
            for images, _, domains in loader:
                images, domains = images.to(device), domains.to(device)

                features, _ = model(images, out='feature')
                domain_output = discriminator(features, lambda_p=0.0)
                preds = F.log_softmax(domain_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += images.size(0)
                correct += (predicted == domains).sum().item()

            accuracy = correct / total
            wandb.log({'[Domain] ' + group + ' Accuracy': accuracy}, step=epoch + 1)
            print('[Domain] ' + group + f' Accuracy: {accuracy * 100:.3f}%')

        with torch.no_grad():
            lc_tester(art_loader, 'Art Paintings')
            lc_tester(cartoon_loader, 'Cartoon')
            lc_tester(sketch_loader, 'Sketch')
            lc_tester(photo_loader, 'Photo')
            dc_tester(art_loader, 'Art Paintings')
            dc_tester(cartoon_loader, 'Cartoon')
            dc_tester(sketch_loader, 'Sketch')
            dc_tester(photo_loader, 'Photo')


if __name__ == '__main__':
    main()
