import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from functions.GradientReversal import GradientReversal
from torch.nn.functional import gumbel_softmax
from model.SimpleCNN import SimpleCNN


class Prunus(nn.Module):
    def __init__(self, feature_extractor='SimpleCNN', pretrained=True, num_classes=10,
                 pre_classifier_out=1024, n_partition=2, part_layer=384, num_domains=2,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(Prunus, self).__init__()
        self.restored = False
        self.num_classes = num_classes
        self.pre_classifier_out = pre_classifier_out
        self.n_partition = n_partition
        self.part_layer = part_layer
        self.device = device

        if feature_extractor == 'SimpleCNN':  # 32*32 -> 128*4*4 | 228*228 -> 128*28*28
            self.features = SimpleCNN().features
        elif feature_extractor == 'Alexnet':  # 228*228 -> 256*6*6
            alexnet = models.alexnet(pretrained=pretrained)
            self.features = alexnet.features
        elif feature_extractor == 'VGG16':  # 32*32 -> 512*1*1 | 228*228 -> 512*7*7
            vgg16 = models.vgg16(pretrained=pretrained)
            self.features = vgg16.features
        elif feature_extractor == 'ResNet50':  # 228*228 -> 2048*1*1
            resnet50 = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(resnet50.children())[:-2])

        self.pre_classifier = nn.Sequential(nn.Identity())

        self.classifier = nn.Sequential(
            nn.Linear(pre_classifier_out, part_layer),
            nn.BatchNorm1d(part_layer),
            nn.ReLU(inplace=True),
            nn.Linear(part_layer, num_classes)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(pre_classifier_out, part_layer),
            nn.BatchNorm1d(part_layer),
            nn.ReLU(inplace=True),
        )
        self.discriminator_fc = nn.Linear(part_layer, num_domains)

        self.partition_switcher = nn.Sequential(
            nn.Linear(part_layer, self.n_partition),
            nn.ReLU(inplace=True)
        )

        self.create_partitioned_classifier()
        self.sync_classifier_with_subnetworks()
        self.feature_dim = None
        self.to(self.device)

    # Pre_classifier that can correspond to any input value
    def _initialize_pre_classifier(self, feature_size):
        self.feature_dim = feature_size
        self.pre_classifier = nn.Sequential(
            nn.Linear(feature_size, self.pre_classifier_out),
            nn.BatchNorm1d(self.pre_classifier_out),
            nn.ReLU(inplace=True),
        )

    # Method to partition the classifier into sub-networks
    def create_partitioned_classifier(self):
        self.partitioned_classifier = nn.ModuleList()

        linear_layers = []
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)

        for p_i in range(self.n_partition):
            partitioned_layer = nn.ModuleList()

            for i, linear_layer in enumerate(linear_layers):
                if i == 0:
                    input_size = linear_layer.in_features
                    output_size = linear_layer.out_features
                    partition_size = output_size // self.n_partition

                    sublayer = nn.Linear(input_size, partition_size)

                    partitioned_layer.append(sublayer)
                    partitioned_layer.append(nn.ReLU(inplace=True))

                elif i == len(linear_layers) - 1:  # ¸¶Áö¸· ·¹ÀÌ¾î
                    input_size = linear_layer.in_features
                    output_size = linear_layer.out_features
                    partition_size = input_size // self.n_partition

                    sublayer = nn.Linear(partition_size, output_size)
                    partitioned_layer.append(sublayer)

            self.partitioned_classifier.append(partitioned_layer)

    def sync_classifier_with_subnetworks(self):
        linear_layers = [layer for layer in self.classifier if isinstance(layer, nn.Linear)]
        linear_layers_subnet = [[layer for layer in partitioned_classifier if isinstance(layer, nn.Linear)] for
                                partitioned_classifier in self.partitioned_classifier]

        for i, linear_layer in enumerate(linear_layers):
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
                ws_ = torch.cat(ws_, dim=0)
                bs_ = torch.cat(bs_, dim=0)
            elif i == 1:
                ws_ = torch.cat(ws_, dim=1)
                bs_ = b_

            with torch.no_grad():
                linear_layer.weight = nn.Parameter(ws_.detach().clone())
                linear_layer.bias = nn.Parameter(bs_.detach().clone())


    def forward(self, input_data, alpha=1.0):
        feature = self.features(input_data)
        feature = feature.view(feature.size(0), -1)

        if self.feature_dim is None:
            self._initialize_pre_classifier(feature.size(1))
            self.to(input_data.device)
        feature = self.pre_classifier(feature)

        grl = GradientReversal(lambd=alpha)
        reverse_feature = grl(feature)
        domain_penul = self.discriminator(reverse_feature)
        domain_output = self.discriminator_fc(domain_penul)

        partition_switcher_output = self.partition_switcher(domain_penul)
        gumbel_output = gumbel_softmax(partition_switcher_output, tau=0.1, hard=False)
        partition_idx = torch.argmax(gumbel_output, dim=1)
        class_output = []

        for b_i in range(feature.size(0)):
            xx_list = []
            p_i = partition_idx[b_i].item()
            xx = feature[b_i].unsqueeze(0)
            for layer in self.partitioned_classifier[p_i]:
                xx = layer(xx)
            xx_list.append(xx)

            stacked = torch.stack(xx_list, dim=0)  # [n_partition, 1, num_classes]
            weighted = torch.sum(gumbel_output[b_i].view(-1, 1, 1) * stacked, dim=0)
            class_output.append(weighted)

        if self.training:
            self.sync_classifier_with_subnetworks()
        class_output_partitioned = torch.cat(class_output, dim=0)
        class_output = self.classifier(feature)

        return class_output_partitioned, domain_output, partition_idx

    def pretrain_fwd(self, pindex_in, input_data, alpha):
        feature = self.features(input_data)
        feature = feature.view(feature.size(0), -1)

        if self.feature_dim is None:
            self._initialize_pre_classifier(feature.size(1))
            self.to(input_data.device)
        feature = self.pre_classifier(feature)

        grl = GradientReversal(lambd=alpha)
        reverse_feature = grl(feature)
        domain_penul = self.discriminator(reverse_feature)
        domain_output = self.discriminator_fc(domain_penul)
        partition_switcher_output = self.partition_switcher(domain_penul)

        partition_idx = torch.full((partition_switcher_output.size(0),), pindex_in, dtype=torch.long, device=domain_penul.device)

        class_output = []

        for b_i in range(feature.size(0)):
            xx = feature[b_i].unsqueeze(0)
            for layer in self.partitioned_classifier[partition_idx[b_i]]:
                xx = layer(xx)
            class_output.append(xx)

        if self.training:
            self.sync_classifier_with_subnetworks()
        class_output_partitioned = torch.cat(class_output, dim=0)

        return class_output_partitioned, domain_output, partition_switcher_output


def prunus_weights(model, lr, pre_weight=1.0, fc_weight=1.0, disc_weight=1.0, switcher_weight=1.0):

    return [
        {'params': model.features.parameters(), 'lr': lr},
        {'params': model.pre_classifier.parameters(), 'lr': lr * pre_weight},
        {'params': model.classifier.parameters(), 'lr': lr * fc_weight},
        {'params': model.discriminator.parameters(), 'lr': lr * disc_weight},
        {'params': model.discriminator_fc.parameters(), 'lr': lr * disc_weight},
        {'params': model.partition_switcher.parameters(), 'lr': lr * switcher_weight},
    ]
