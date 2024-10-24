import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functions.ReverseLayerF import ReverseLayerF
from functions.lr_lambda import lr_lambda
from dataloader.data_loader import data_loader
import numpy as np
import wandb
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DANN(nn.Module):
    def __init__(self, n_partition = 4):
        super(DANN, self).__init__()
        self.restored = False
        self.n_partition = n_partition

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),  # 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),  # 9
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4),  # 1
            nn.ReLU()
        )

        self.pre_classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.discriminator_fc = nn.Linear(256, 2)

        self.partition_switcher = nn.Sequential(nn.Linear(256, self.n_partition),
                                                nn.ReLU(inplace=True))


        self.create_partitioned_classifier()
        self.sync_classifier_with_subnetworks()
        print('')

    # Method to partition the classifier into sub-networks
    def create_partitioned_classifier(self):
        self.partitioned_classifier = []  # ModuleList로 초기화

        linear_layers = []
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)

        # 각 파티션에 대해 서브네트워크를 생성하면서 가중치를 공유하도록 설정
        for p_i in range(self.n_partition):
            partitioned_layer = nn.ModuleList()

            for i, linear_layer in enumerate(linear_layers):
                if i == 0:  # 첫 번째 레이어
                    input_size = linear_layer.in_features
                    output_size = linear_layer.out_features
                    partition_size = output_size // self.n_partition

                    # 서브 레이어 생성 (가중치를 공유함)
                    sublayer = nn.Linear(input_size, partition_size)

                    partitioned_layer.append(sublayer)
                    # partitioned_layer.append(nn.BatchNorm1d(partition_size))
                    partitioned_layer.append(nn.ReLU(inplace=True))

                elif i == len(linear_layers) - 1:  # 마지막 레이어
                    input_size = linear_layer.in_features
                    output_size = linear_layer.out_features
                    partition_size = input_size // self.n_partition

                    # 서브 레이어 생성 (가중치를 공유함)
                    sublayer = nn.Linear(partition_size, output_size)
                    partitioned_layer.append(sublayer)

                # else:  # 중간 레이어
                #     input_size = linear_layer.in_features
                #     output_size = linear_layer.out_features
                #     partition_in_size = input_size // self.n_partition
                #     partition_out_size = output_size // self.n_partition
                #
                #     # 서브 레이어 생성 (가중치를 공유함)
                #     sublayer = nn.Linear(partition_in_size, partition_out_size)
                #
                #     partitioned_layer.append(sublayer)
                #     partitioned_layer.append(nn.BatchNorm1d(partition_out_size))
                #     partitioned_layer.append(nn.ReLU(inplace=True))

            self.partitioned_classifier.append(partitioned_layer)
    def sync_classifier_with_subnetworks(self):
        linear_layers = [layer for layer in self.classifier if isinstance(layer, nn.Linear)]
        linear_layers_subnet = [[layer for layer in partitioned_classifier if isinstance(layer, nn.Linear)] for partitioned_classifier in self.partitioned_classifier]

        # weight_classifier = self.classifier[3].bias[0]
        # print(weight_classifier)

        for i, linear_layer in enumerate(linear_layers):
            # get layers weight and bias
            w_ = linear_layer.weight
            b_ = linear_layer.bias

            ws_ = []
            bs_ = []
            for j, subnet_layer in enumerate(linear_layers_subnet):
                if i == 1:
                    with torch.no_grad():
                        subnet_layer[i].bias.copy_(b_)
                ws_.append(subnet_layer[i].weight)
                bs_.append(subnet_layer[i].bias)

            if i == 0:
                ws_ = torch.cat(ws_, dim = 0)
                bs_ = torch.cat(bs_, dim = 0)
            elif i == 1:
                ws_ = torch.cat(ws_, dim = 1)
                bs_ = b_



            with torch.no_grad():
                linear_layer.weight.copy_(ws_)
                linear_layer.bias.copy_(bs_)

        weight_classifier = self.classifier[3].bias[0]
        # print(weight_classifier)
        # print('')

    def forward(self, input_data, alpha=1.0):
        input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 128 * 1 * 1)
        feature = self.pre_classifier(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_penul = self.discriminator(reverse_feature)
        domain_output = self.discriminator_fc(domain_penul)

        # pass copied/detached domain_penul through partition_switcher
        partition_switcher_output = self.partition_switcher(domain_penul.clone().detach())
        partition_switcher_output = F.softmax(partition_switcher_output, dim=1)
        partition_switcher_output = torch.distributions.dirichlet.Dirichlet(partition_switcher_output).sample()

        # sample idx from partition_switcher_output
        partition_idx = torch.argmax(partition_switcher_output, dim=1)

        # run the partitioned classifier # TODO
        class_output = []

        for b_i in range(feature.size(0)):
            xx = feature[b_i].unsqueeze(0)
            for layer in self.partitioned_classifier[partition_idx[b_i]]:
                xx = layer(xx)
            class_output.append(xx)

            # class_output.append(self.partitioned_classifier[partition_idx[b_i]](feature[b_i].unsqueeze(0)))
        self.sync_classifier_with_subnetworks()
        class_output_partitioned = torch.cat(class_output, dim=0)
        class_output = self.classifier(feature)

        return class_output_partitioned, class_output, domain_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--pretrain_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--source1', type=str, default='SVHN')
    parser.add_argument('--source2', type=str, default='MNIST')
    parser.add_argument('--target', type=str, default='CIFAR10')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_cls', type=float, default=0.01)
    parser.add_argument('--lr_dom', type=float, default=0.1)
    args = parser.parse_args()

    pre_epochs = args.pretrain_epoch
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(project="EM_Domain_mstest_dk",
               entity="hails",
               config=args.__dict__,
               name="DANN_MultiSource_LSDS_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
                    + "_S1:" + args.source1 + "/S2:" + args.source2 + "/T:" + args.target
               )

    source1_loader, source1_loader_test = data_loader(args.source1, args.batch_size)
    source2_loader, source2_loader_test = data_loader(args.source2, args.batch_size)
    target_loader, target_loader_test = data_loader(args.target, args.batch_size)

    print("Data load complete, start training")

    model = DANN().to(device)

    pre_opt = optim.Adam(model.parameters(), lr=1e-5)
    optimizer_cls = optim.SGD(list(model.feature.parameters()) + list(model.classifier.parameters()), lr=args.lr_cls, momentum=0.9, weight_decay=1e-6)
    optimizer_dom = optim.SGD(list(model.feature.parameters()) + list(model.discriminator.parameters()), lr=args.lr_dom, momentum=0.9, weight_decay=1e-6)
    scheduler_cls = optim.lr_scheduler.LambdaLR(optimizer_cls, lr_lambda)
    scheduler_dom = optim.lr_scheduler.LambdaLR(optimizer_dom, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(pre_epochs):
        model.train()
        i = 0

        for source1_data, source2_data, target_data in zip(source1_loader, source2_loader, target_loader):
            p = (float(i + epoch * min(len(source1_loader), len(source2_loader))) /
                 num_epochs / min(len(source1_loader), len(source2_loader)))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            # Training with source data
            source1_images, source1_labels = source1_data
            source2_images, source2_labels = source2_data
            target_images,  target_lables  = target_data
            source1_images, source1_labels = source1_images.to(device), source1_labels.to(device)
            source2_images, source2_labels = source2_images.to(device), source2_labels.to(device)
            target_images,  target_lables  = target_images.to(device),  target_lables.to(device)

            class1_output_partitioned, class1_output, _ = model(source1_images, alpha=lambda_p)
            source1_loss = criterion(class1_output_partitioned, source1_labels)
            class2_output_partitioned, class2_output, _ = model(source2_images, alpha=lambda_p)
            source2_loss = criterion(class2_output_partitioned, source2_labels)
            target_output_partitioned, target_output, _ = model(target_images, alpha=lambda_p)
            target_loss = criterion(target_output_partitioned, target_lables)

            loss = source1_loss + source2_loss + target_loss

            pre_opt.zero_grad()
            loss.backward()
            pre_opt.step()
            """
            label_acc = (torch.argmax(class_output, dim=1) == source_labels).sum().item() / source_labels.size(0)
            print(f'Batches [{i + 1}/{len(source_loader)}], '
                  f'Pretrain Loss: {loss.item():.4f}, '
                  f'Pretrain Accuracy: {label_acc * 100:.3f}%, '
                  )
            """
            label1_acc = (torch.argmax(class1_output_partitioned, dim=1) == source1_labels).sum().item() / source1_labels.size(0)
            label2_acc = (torch.argmax(class2_output_partitioned, dim=1) == source2_labels).sum().item() / source2_labels.size(0)
            label3_acc = (torch.argmax(target_output_partitioned, dim=1) == target_lables).sum().item() / target_lables.size(0)

            print(f'Batches [{i + 1}/{min(len(source1_loader), len(source2_loader), len(target_loader))}], '
                    f'Source1 Loss: {source1_loss.item():.4f}, '
                    f'Source2 Loss: {source2_loss.item():.4f}, '
                    f'Target Loss: {target_loss.item():.4f}, '
                    f'Source1 Accuracy: {label1_acc * 100:.3f}%, '
                    f'Source2 Accuracy: {label2_acc * 100:.3f}%, '
                    f'Target Accuracy: {label3_acc * 100:.3f}%'
                    )
            i += 1

        # scheduler.step()
        # model.eval()

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        i = 0

        loss_tgt_domain_epoch = 0
        loss_src_domain_epoch = 0
        loss_label_epoch = 0

        for source1_data, source2_data, target_data in zip(source1_loader, source2_loader, tqdm(target_loader)):
            p = (float(i + epoch * min(len(source1_loader), len(source2_loader), len(target_loader))) /
                 num_epochs / min(len(source1_loader), len(source2_loader), len(target_loader)))
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer_cls.zero_grad()
            optimizer_dom.zero_grad()

            # Training with source data
            source1_images, source1_labels = source1_data
            source1_images, source1_labels = source1_images.to(device), source1_labels.to(device)
            source1_dlabel = torch.full((source1_images.size(0),), 0, dtype=torch.long, device=device)
            source2_images, source2_labels = source2_data
            source2_images, source2_labels = source2_images.to(device), source2_labels.to(device)
            source2_dlabel = torch.full((source1_images.size(0),), 0, dtype=torch.long, device=device)
            target_images,  target_labels = target_data
            target_images,  target_labels = target_images.to(device),  target_labels.to(device)
            target_dlabel = torch.full((target_images.size(0),), 1, dtype=torch.long, device=device)

            source1_label_output, source1_domain_output = model(source1_images, alpha=lambda_p)
            source2_label_output, source2_domain_output = model(source2_images, alpha=lambda_p)
            target_label_output, target_domain_output = model(target_images, alpha=lambda_p)
            source1_label_loss = criterion(source1_label_output, source1_labels)
            source2_label_loss = criterion(source2_label_output, source2_labels)
            target_label_loss = criterion(target_label_output, target_labels)

            label_loss = source1_label_loss + source2_label_loss + target_label_loss
            loss_label_epoch += label_loss.item()

            domain_src1_loss = criterion(source1_domain_output, source1_dlabel)
            domain_src2_loss = criterion(source2_domain_output, source2_dlabel)

            domain_src_loss = domain_src1_loss + domain_src2_loss
            loss_src_domain_epoch += domain_src_loss.item()

            _, domain_output = model(target_images, alpha=lambda_p)
            domain_tgt_loss = criterion(domain_output, target_dlabel)

            loss = domain_tgt_loss + domain_src_loss + label_loss
            loss_tgt_domain_epoch += domain_tgt_loss.item()

            loss.backward()
            optimizer_cls.step()
            optimizer_dom.step()

            label1_acc = (torch.argmax(source1_label_output, dim=1) == source1_labels).sum().item() / source1_labels.size(0)
            label2_acc = (torch.argmax(source2_label_output, dim=1) == source2_labels).sum().item() / source2_labels.size(0)
            label3_acc = (torch.argmax(target_label_output, dim=1) == target_labels).sum().item() / target_labels.size(0)
            domain_source1_acc = (torch.argmax(source1_domain_output, dim=1) == source1_dlabel).sum().item() / source1_dlabel.size(0)
            domain_source2_acc = (torch.argmax(source2_domain_output, dim=1) == source2_dlabel).sum().item() / source2_dlabel.size(0)
            domain_target_acc = (torch.argmax(domain_output, dim=1) == target_dlabel).sum().item() / target_dlabel.size(0)

            """
            label_acc = (torch.argmax(class_output, dim=1) == source_labels).sum().item() / source_labels.size(0)
            domain_acc = (torch.argmax(domain_output, dim=1) == target_dlabel).sum().item() / target_dlabel.size(0)

            print(f'Batches [{i + 1}/{min(len(source_loader), len(target_loader))}], '
                  f'Domain source Loss: {domain_src_loss.item():.4f}, '
                  f'Domain target Loss: {domain_tgt_loss.item():.4f}, '
                  f'Label Loss: {label_loss.item():.4f}, '
                  f'Label Accuracy: {label_acc * 100:.3f}%, '
                  f'Domain Accuracy: {domain_acc * 100:.3f}%')
            """

            print(f'Batches [{i + 1}/{min(len(source1_loader), len(source2_loader), len(target_loader))}], '
                  f'Domain source1 Loss: {domain_src1_loss.item():.4f}, '
                  f'Domain source2 Loss: {domain_src2_loss.item():.4f}, '
                  f'Domain target Loss: {domain_tgt_loss.item():.4f}, '
                  f'Label Loss: {label_loss.item():.4f}, '
                  f'Source1 Accuracy: {label1_acc * 100:.3f}%, '
                  f'Source2 Accuracy: {label2_acc * 100:.3f}%, '
                  f'Target Accuracy: {label3_acc * 100:.3f}%, '
                  f'Source1 Domain Accuracy: {domain_source1_acc * 100:.3f}%, '
                  f'Source2 Domain Accuracy: {domain_source2_acc * 100:.3f}%, '
                  f'Target Domain Accuracy: {domain_target_acc * 100:.3f}%'
                )
            i += 1

        scheduler_cls.step()
        scheduler_dom.step()

        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Domain source Loss: {loss_src_domain_epoch:.4f}, '
              f'Domain target Loss: {loss_tgt_domain_epoch:.4f}, '
              f'Label Loss: {loss_label_epoch:.4f}, '
              f'Total Loss: {loss_src_domain_epoch + loss_tgt_domain_epoch  + loss_label_epoch:.4f}, '
              f'Time: {end_time - start_time:.2f} seconds'
              )

        wandb.log({
            'Domain source Loss': loss_src_domain_epoch,
            'Domain target Loss': loss_tgt_domain_epoch,
            'Label Loss': loss_label_epoch,
            'Total Loss': loss_src_domain_epoch + loss_tgt_domain_epoch + loss_label_epoch,
            'Training Time': end_time - start_time
        })

        model.eval()

        def lc_tester(loader, group):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output, _ = model(images, alpha=0.0)
                preds = F.log_softmax(class_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            wandb.log({'[Label] ' + group + ' Accuracy': accuracy}, step=epoch + 1)
            print('[Label] ' + group + f' Accuracy: {accuracy * 100:.3f}%')

        def dc_tester(loader, group, d_label):
            correct, total = 0, 0
            for images, _ in loader:
                images = images.to(device)

                _, domain_output = model(images, alpha=0.0)
                preds = F.log_softmax(domain_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += images.size(0)
                correct += (predicted == d_label).sum().item()

            accuracy = correct / total
            wandb.log({'[Domain] ' + group + ' Accuracy': accuracy}, step=epoch + 1)
            print('[Domain] ' + group + f' Accuracy: {accuracy * 100:.3f}%')

        with torch.no_grad():
            lc_tester(source1_loader_test, 'Source1')
            lc_tester(source2_loader_test, 'Source2')
            lc_tester(target_loader_test, 'Target')
            dc_tester(source1_loader_test, 'Source1', 1)
            dc_tester(source2_loader_test, 'Source2', 1)
            dc_tester(target_loader_test, 'Target', 0)


if __name__ == '__main__':
    main()
