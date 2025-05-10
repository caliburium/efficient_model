import os
import deeplake
import torch
from torch.utils.data import Dataset, DataLoader


class DomainNetDataset(Dataset):
    def __init__(self, domain, labels, split='train', transform=None, root_dir="./domainnet_data"):
        self.root_dir = root_dir
        self.domain = domain
        self.labels = labels
        self.split = split
        self.transform = transform

        # Define dataset path
        dataset_path = os.path.join(self.root_dir, f"{domain}-{split}")
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
            print(f"Downloading {domain} {split} dataset from Deeplake...")
            ds = deeplake.load(f"hub://activeloop/domainnet-{domain}-{split}")
            ds.export(dataset_path)  # Save locally
        else:
            print(f"Using existing dataset at {dataset_path}")

        # Load dataset from local files
        self.dataset = deeplake.load(dataset_path)

        # Filter dataset by selected labels
        self.data, self.targets = self._filter_labels()

    def _filter_labels(self):
        data = []
        targets = []
        label_map = {old_label: new_label for new_label, old_label in enumerate(self.labels)}

        for i in range(len(self.dataset)):
            sample_label = int(self.dataset["labels"][i].numpy())
            if sample_label in self.labels:
                img = self.dataset["images"][i].numpy()
                data.append(img)
                targets.append(label_map[sample_label])  # Relabel

        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1), torch.tensor(label, dtype=torch.long)


def dn_loader(domain, label, batch_size=32, root_dir="./domainnet_data", transform=None):
    train_dataset = DomainNetDataset(domain=domain, labels=label, split='train', transform=transform, root_dir=root_dir)
    test_dataset = DomainNetDataset(domain=domain, labels=label, split='test', transform=transform, root_dir=root_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage:
# sketch_loader, sketch_loader_test = data_loader(domain='sketch', label=[0, 9, 11, 15, 17, 22], batch_size=32)


