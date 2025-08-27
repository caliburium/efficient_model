import torch
import torch.nn as nn
from functions.ReverseLayerF import ReverseLayerF
from torch.nn.functional import gumbel_softmax


class Prunus(nn.Module):
    def __init__(self, num_classes=10, pre_classifier_out=1024, n_partition=2, part_layer=384, num_domains=2,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(Prunus, self).__init__()
        self.device = device
        self.n_partition = n_partition

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.pre_classifier = nn.Sequential(
            nn.Linear(16 * 16 * 16, pre_classifier_out),
            nn.BatchNorm1d(pre_classifier_out),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(pre_classifier_out, part_layer),
            nn.BatchNorm1d(part_layer),
            nn.ReLU(),
            nn.Linear(part_layer, num_classes)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(pre_classifier_out, part_layer),
            nn.BatchNorm1d(part_layer),
            nn.ReLU(),
        )

        self.discriminator_fc = nn.Linear(part_layer, num_domains)

        self.partition_switcher = nn.Linear(part_layer, n_partition)

        self.create_partitioned_classifier()
        self.sync_classifier_with_subnetworks()
        self.to(self.device)

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

                elif i == len(linear_layers) - 1:
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

    def forward(self, input_data, alpha=1.0, tau=0.1, inference=False):
        feature = self.features(input_data)
        feature = feature.view(feature.size(0), -1)
        feature = self.pre_classifier(feature)

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_penul = self.discriminator(reverse_feature)
        domain_output = self.discriminator_fc(domain_penul)

        partition_switcher_output = self.partition_switcher(domain_penul)

        # hard=True를 사용하면 gumbel_output이 one-hot 벡터가 되어 라우팅이 명확해집니다.
        # Straight-Through Estimator 덕분에 그래디언트는 스위처로 잘 전달됩니다.
        gumbel_output = gumbel_softmax(partition_switcher_output, tau=tau, hard=True)

        # 각 데이터 샘플이 어떤 파티션으로 갈지 결정합니다.
        partition_idx = torch.argmax(gumbel_output, dim=1)

        # 최종 출력을 담을 빈 텐서를 생성합니다.
        class_output_partitioned = torch.zeros(feature.size(0), self.classifier[-1].out_features, device=self.device)

        # for 루프를 배치 크기만큼 돌지 않고, 파티션 개수(n_partition)만큼만 돕니다.
        for p_i in range(self.n_partition):
            # p_i번 파티션으로 할당된 데이터들의 인덱스를 찾습니다.
            indices = torch.where(partition_idx == p_i)[0]

            # 이 파티션에 할당된 데이터가 없으면 건너뜁니다.
            if len(indices) == 0:
                continue

            # 해당 인덱스의 데이터(feature)들만 모읍니다.
            selected_features = feature[indices]

            # 선택된 데이터들을 p_i번 파티션 네트워크에 '한 번에' 통과시킵니다.
            xx = selected_features
            for layer in self.partitioned_classifier[p_i]:
                xx = layer(xx)

            # 계산된 결과를 최종 출력 텐서의 원래 위치에 맞게 다시 채워 넣습니다.
            class_output_partitioned[indices] = xx

        if self.training:
            self.sync_classifier_with_subnetworks()

        # 이 'class_output'은 sync 확인 및 디버깅용으로 남겨둘 수 있으나,
        # 실제 학습에는 class_output_partitioned를 사용해야 합니다.
        class_output = self.classifier(feature)

        return class_output_partitioned, domain_output, partition_idx, gumbel_output


def prunus_weights(model, lr, pre_weight=1.0, fc_weight=1.0, disc_weight=1.0, switcher_weight=1.0):

    return [
        {'params': model.features.parameters(), 'lr': lr},
        {'params': model.pre_classifier.parameters(), 'lr': lr * pre_weight},
        {'params': model.classifier.parameters(), 'lr': lr * fc_weight},
        {'params': model.discriminator.parameters(), 'lr': lr * disc_weight},
        {'params': model.discriminator_fc.parameters(), 'lr': lr * disc_weight},
        {'params': model.partition_switcher.parameters(), 'lr': lr * switcher_weight},
    ]
