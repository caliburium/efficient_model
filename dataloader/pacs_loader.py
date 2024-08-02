from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import deeplake
from PIL import Image
import numpy as np


class DeepLakePACS(Dataset):
    def __init__(self, deeplake_ds, domain, transform=None):
        # Define the domain mapping
        self.domain_mapping = {
            0: "photo",
            1: "art_painting",
            2: "cartoon",
            3: "sketch"
        }

        # Validate domain argument
        if isinstance(domain, str):
            reverse_mapping = {v: k for k, v in self.domain_mapping.items()}
            if domain not in reverse_mapping:
                raise ValueError(f"Domain '{domain}' is not valid. Choose from {list(reverse_mapping.keys())}.")
            self.domain = reverse_mapping[domain]
        elif isinstance(domain, int):
            if domain not in self.domain_mapping:
                raise ValueError(f"Domain index {domain} is not valid. Choose from {list(self.domain_mapping.keys())}.")
            self.domain = domain
        else:
            raise TypeError("Domain must be a string or integer.")

        self.ds = deeplake_ds
        self.transform = transform

        # Verify if the correct tensor name exists
        self.domain_tensor_name = 'domains'  # Use the actual tensor name here
        if self.domain_tensor_name not in self.ds.tensors:
            raise KeyError(f"Tensor '{self.domain_tensor_name}' does not exist in the dataset.")

        # Filter indices by domain index
        self.indices = [
            i for i, sample in enumerate(self.ds) if sample[self.domain_tensor_name].numpy()[0] == self.domain
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Use filtered indices to get the correct sample
        actual_idx = self.indices[idx]
        sample = self.ds[actual_idx]

        # Convert the numpy image to PIL image
        image = sample['images'].numpy()
        image = Image.fromarray(image.astype(np.uint8))  # Convert numpy array to PIL Image

        label = sample['labels'].numpy()
        domain = sample[self.domain_tensor_name].numpy()[0]  # As integer index

        if self.transform:
            image = self.transform(image)

        return image, label, domain


class PACS(Dataset):
    def __init__(self, deeplake_ds, transform=None):
        # Define the domain mapping to new numeric indices
        self.domain_mapping = {
            0: "photo",
            1: "art_painting",
            2: "cartoon",
            3: "sketch"
        }

        self.ds = deeplake_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        # Convert the numpy image to PIL image
        image = sample['images'].numpy()
        image = Image.fromarray(image.astype(np.uint8))  # Convert numpy array to PIL Image

        label = sample['labels'].numpy()
        domain = sample['domains'].numpy()[0]  # Use the correct tensor name

        if self.transform:
            image = self.transform(image)

        return image, label, domain


def pacs_loader(domain, batch_size):
    transform = transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pacs_train = deeplake.load("hub://activeloop/pacs-train")
    # pacs_val = deeplake.load("hub://activeloop/pacs-val")
    pacs_test = deeplake.load("hub://activeloop/pacs-test")

    if domain == 'all':
        train_dataset = PACS(pacs_train, transform=transform)
        # val_dataset = PACS(pacs_val, transform=transform)
        test_dataset = PACS(pacs_test, transform=transform)
    else:
        train_dataset = DeepLakePACS(pacs_train, domain=domain, transform=transform)
        # val_dataset = DeepLakePACS(pacs_val, domain=domain, transform=transform)
        test_dataset = DeepLakePACS(pacs_test, domain=domain, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader  # , val_loader
