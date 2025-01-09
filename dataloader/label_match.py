import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from dataloader.ToBlackAndWhite import ToBlackAndWhite


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label_map, exclude_classes):
        self.dataset = dataset
        self.label_map = label_map
        self.exclude_classes = exclude_classes

        self.filtered_data = [
            (img, label_map[label]) for img, label in dataset
            if label not in exclude_classes
        ]

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        img, label = self.filtered_data[idx]
        return img, label


def data_loader(source, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_stl10 = transforms.Compose([
        transforms.Resize((32, 32), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar10_label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 8: 7, 9: 8}
    cifar10_exclude_classes = [7]

    stl10_label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 9: 8}
    stl10_exclude_classes = [8]

    if source == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        dataset = CustomDataset(dataset, cifar10_label_map, cifar10_exclude_classes)
        dataset_test = CustomDataset(dataset_test, cifar10_label_map, cifar10_exclude_classes)

    elif source == 'STL10':
        dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_stl10)
        dataset_test = datasets.STL10(root='./data', split='test', download=True, transform=transform_stl10)

        dataset = CustomDataset(dataset, stl10_label_map, stl10_exclude_classes)
        dataset_test = CustomDataset(dataset_test, stl10_label_map, stl10_exclude_classes)

    else:
        raise ValueError("Invalid source dataset")

    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=8)

    return loader, test_loader