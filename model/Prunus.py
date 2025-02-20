import torch
import torch.nn as nn
import torch.nn.functional as F
from functions.ReverseLayerF import ReverseLayerF


class Prunus(nn.Module):
    def __init__(self, n_partition = 4):
        super(Prunus, self).__init__()
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

        # run the partitioned classifier
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


def prunus_weights(model, lr, fc_weight=1.0, disc_weight=1.0):

    return [
        {'params': model.features.parameters(), 'lr': lr},
        {'params': model.pre_classifier.parameters(), 'lr': lr * fc_weight},
        {'params': model.classifier.parameters(), 'lr': lr * fc_weight},
        {'params': model.discriminator.parameters(), 'lr': lr * disc_weight},
        {'params': model.discriminator_fc.parameters(), 'lr': lr * disc_weight},
        {'params': model.partition_switcher.parameters(), 'lr': lr},
        {'params': model.create_partitioned_classifier.parameters(), 'lr': lr},
        {'params': model.sync_classifier_with_subnetworks.parameters(), 'lr': lr},
    ]
