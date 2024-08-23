import os
import numpy as np
import gzip
import shutil
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import deeplake


class PACS_Dataset(Dataset):
    def __init__(self, split, domain=None, transform=None, download=False, root='./data'):
        self.split = split
        self.domain = domain
        self.transform = transform
        self.root = os.path.join(root, f'pacs_{split}.gz')

        if download:
            self._download_data()

        self.extracted_root = os.path.join(root, f'pacs_{split}_extracted')
        os.makedirs(self.extracted_root, exist_ok=True)

        self.image_paths, self.label_paths = self._load_extracted_data()
        self.length = len(self.image_paths)

        if self.domain is not None:
            self._filter_by_domain()

    def _download_data(self):
        if os.path.exists(self.root):
            print(f"Dataset {self.root} already exists. Skipping download.")
            return

        print(f"Downloading and processing dataset into {self.root}...")

        os.makedirs(os.path.dirname(self.root), exist_ok=True)

        # Load Deep Lake datasets
        pacs_train = deeplake.load("hub://activeloop/pacs-train")
        pacs_test = deeplake.load("hub://activeloop/pacs-test")
        pacs_val = deeplake.load("hub://activeloop/pacs-val")  # Assume a validation dataset exists

        # Save datasets locally
        if self.split == 'train':
            self._save_dataset(pacs_train, "train")
        elif self.split == 'test':
            self._save_dataset(pacs_test, "test")
        elif self.split == 'val':
            self._save_dataset(pacs_val, "val")

        self._compress_data()

    def _save_dataset(self, deeplake_dataset, split_name):
        temp_dir = os.path.join(os.path.dirname(self.root), f'pacs_{split_name}_temp')
        os.makedirs(temp_dir, exist_ok=True)

        for i, data_item in enumerate(deeplake_dataset):
            image_path = os.path.join(temp_dir, f"{split_name}_img_{i}.png")
            label_path = os.path.join(temp_dir, f"{split_name}_label_domain_{i}.npz")

            images = data_item['images'].numpy()
            labels = data_item['labels'].numpy().astype(np.int64)
            domains = data_item['domains'].numpy().astype(np.int64)

            # Save image
            Image.fromarray(np.uint8(images)).save(image_path)

            # Save labels and domains
            np.savez(label_path, labels=labels, domains=domains)

    def _compress_data(self):
        with gzip.open(self.root, 'wb') as f_out:
            shutil.make_archive(base_name=self.extracted_root, format='gztar', root_dir=self.extracted_root)
            with open(f"{self.extracted_root}.tar.gz", 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
        print(f"Dataset compressed and saved as {self.root}.")
        shutil.rmtree(self.extracted_root)  # Remove temporary extraction directory

    def _load_extracted_data(self):
        if not os.path.exists(self.extracted_root):
            print(f"Extracting {self.root}...")
            with gzip.open(self.root, 'rb') as f_in:
                with open(f"{self.extracted_root}.tar.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            shutil.unpack_archive(f"{self.extracted_root}.tar.gz", self.extracted_root, 'gztar')

        image_paths = sorted(
            [os.path.join(self.extracted_root, f) for f in os.listdir(self.extracted_root) if f.endswith(".png")])
        label_paths = sorted(
            [os.path.join(self.extracted_root, f) for f in os.listdir(self.extracted_root) if f.endswith(".npz")])

        return image_paths, label_paths

    def _filter_by_domain(self):
        filtered_image_paths = []
        filtered_label_paths = []

        for img_path, lbl_path in zip(self.image_paths, self.label_paths):
            label_domain_data = np.load(lbl_path)
            domains = label_domain_data['domains']

            if domains == self.domain:
                filtered_image_paths.append(img_path)
                filtered_label_paths.append(lbl_path)

        self.image_paths = filtered_image_paths
        self.label_paths = filtered_label_paths

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

        return image, labels, domains


def pacs_loader(split, domain=None, batch_size=32, transform=None, download=False, root='./data'):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = PACS_Dataset(split=split, domain=domain, transform=transform, download=download, root=root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=0)

    return loader
