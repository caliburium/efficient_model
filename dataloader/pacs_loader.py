import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import deeplake


class PACS_Dataset(Dataset):
    VALID_DOMAINS = {
        'full': [0, 1, 2, 3],
        'train': [0, 1, 2],  # Artpaintings, Cartoon, Sketch
        'artpaintings': 0,
        'cartoon': 1,
        'sketch': 2,
        'photo': 3
    }

    def __init__(self, split, domain=None, transform=None, download=True, root='./data'):
        self.split = split
        self.domain = domain
        self.transform = transform
        self.root = os.path.join(root, f'pacs_{split}')
        os.makedirs(self.root, exist_ok=True)

        if download and not self._data_exists():
            self._download_data()

        self.image_paths, self.label_paths = self._load_data()
        self.length = len(self.image_paths)

        if self.domain is not None:
            self._filter_by_domain()

    def _data_exists(self):
        """ Check if the dataset is already downloaded locally. """
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0

    def _download_data(self):
        print(f"Downloading and processing dataset into {self.root}...")

        # Save datasets locally
        if self.split == 'train':
            pacs_train = deeplake.load("hub://activeloop/pacs-train")
            self._save_dataset(pacs_train, "train")
        elif self.split == 'test':
            pacs_test = deeplake.load("hub://activeloop/pacs-test")
            self._save_dataset(pacs_test, "test")
        elif self.split == 'val':
            pacs_val = deeplake.load("hub://activeloop/pacs-val")
            self._save_dataset(pacs_val, "val")

    def _save_dataset(self, deeplake_dataset, split_name):
        for i, data_item in enumerate(deeplake_dataset):
            image_path = os.path.join(self.root, f"{split_name}_img_{i}.png")
            label_path = os.path.join(self.root, f"{split_name}_label_domain_{i}.npz")

            images = data_item['images'].numpy()
            labels = data_item['labels'].numpy().astype(np.int64)
            domains = data_item['domains'].numpy().astype(np.int64)

            # Save image
            Image.fromarray(np.uint8(images)).save(image_path)

            # Save labels and domains
            np.savez(label_path, labels=labels, domains=domains)

    def _load_data(self):
        """ Load images and labels from the local directory. """
        image_paths = sorted(
            [os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith(".png")])
        label_paths = sorted(
            [os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith(".npz")])

        return image_paths, label_paths

    def _filter_by_domain(self):
        if self.domain == 'full':
            valid_domains = self.VALID_DOMAINS['full']
        elif self.domain == 'train':
            valid_domains = self.VALID_DOMAINS['train']
        else:
            if isinstance(self.domain, int) and self.domain in self.VALID_DOMAINS.values():
                valid_domains = [self.domain]
            elif isinstance(self.domain, str) and self.domain.lower() in self.VALID_DOMAINS:
                valid_domains = [self.VALID_DOMAINS[self.domain.lower()]]
            else:
                raise ValueError(f"Invalid domain: {self.domain}. Valid domains are 'artpaintings', 'cartoon', 'sketch', 'photo', or 'train'.")

        filtered_image_paths = []
        filtered_label_paths = []

        for img_path, lbl_path in zip(self.image_paths, self.label_paths):
            label_domain_data = np.load(lbl_path)
            domains = label_domain_data['domains']

            if domains in valid_domains:
                filtered_image_paths.append(img_path)
                filtered_label_paths.append(lbl_path)

        self.image_paths = filtered_image_paths
        self.label_paths = filtered_label_paths
        self.length = len(self.image_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx])

        # Load labels and domains
        label_domain_data = np.load(self.label_paths[idx])
        labels = label_domain_data['labels']
        domains = label_domain_data['domains']

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(labels).squeeze()
        domains = torch.tensor(domains).squeeze()

        return image, labels, domains


def pacs_loader(split, domain=None, batch_size=128, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = PACS_Dataset(split=split, domain=domain, transform=transform, download=True)
    loader = DataLoader(dataset, drop_last=True, batch_size=batch_size, shuffle=True, num_workers=0)

    return loader