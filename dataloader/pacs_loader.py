from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class PACS_all(Dataset):
    def __init__(self, deeplake_dataset, transform=None):
        self.deeplake_dataset = deeplake_dataset
        self.transform = transform
        self.length = len(self.deeplake_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Retrieve the data item from the Deep Lake dataset
        data_item = self.deeplake_dataset[idx]
        images = data_item['images'].numpy()
        labels = data_item['labels'].numpy().astype(np.int64)  # Ensure labels are integer class indices
        domains = data_item['domains'].numpy().astype(np.int64)  # Ensure domains are integer class indices

        # Convert NumPy array to PIL Image
        images = Image.fromarray(np.uint8(images))

        if self.transform:
            images = self.transform(images)

        return images, labels, domains


class PACS_Domain(Dataset):
    def __init__(self, deeplake_dataset, domain, transform=None):
        self.deeplake_dataset = deeplake_dataset
        self.transform = transform
        self.domain = domain
        self.indices = self._get_domain_indices()
        self.length = len(self.indices)

    def _get_domain_indices(self):
        domain_indices = []
        for i, data_item in enumerate(self.deeplake_dataset):
            item_domain = data_item['domains'].numpy()
            if item_domain == self.domain:
                domain_indices.append(i)
        return domain_indices

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Retrieve the data item for the given index within the specified domain
        data_index = self.indices[idx]
        data_item = self.deeplake_dataset[data_index]
        images = data_item['images'].numpy()
        labels = data_item['labels'].numpy().astype(np.int64)  # Ensure labels are integer class indices
        domains = data_item['domains'].numpy().astype(np.int64)  # Ensure domains are integer class indices

        # Convert NumPy array to PIL Image
        images = Image.fromarray(np.uint8(images))

        if self.transform:
            images = self.transform(images)

        return images, labels, domains


def pacs_loader(split, domain, batch_size, pacs_train, pacs_test):
    transform = transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not split:
        train_dataset = PACS_all(pacs_train, transform=transform)
        test_dataset = PACS_all(pacs_test, transform=transform)
    else:
        train_dataset = PACS_Domain(pacs_train, domain=domain, transform=transform)
        test_dataset = PACS_Domain(pacs_test, domain=domain, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader
