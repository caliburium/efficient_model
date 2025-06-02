import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class DomainNetDataset(Dataset):
    def __init__(self,
                 data_root="../data/DomainNet",
                 domains=None,
                 selected_classes=None,
                 split="train",
                 transform=None,
                 return_domain_label=True):
        """
        DomainNet Dataset with custom class and domain selection

        Args:
            data_root: Path to DomainNet data directory
            domains: List of domains to use (e.g., ['clipart', 'real', 'sketch'])
                    If None, uses all available domains
            selected_classes: List of class names to use (e.g., ['airplane', 'bird', 'car'])
                            If None, uses all available classes
            split: 'train' or 'test'
            transform: torchvision transforms
            return_domain_label: Whether to return domain labels along with class labels
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.return_domain_label = return_domain_label

        # Available domains
        available_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

        # Set domains to use
        if domains is None:
            self.domains = available_domains
        else:
            self.domains = [d for d in domains if d in available_domains]

        print(f"Using domains: {self.domains}")

        # Load and process data
        self.data = []
        all_classes = set()

        # Load data from each domain
        for domain in self.domains:
            domain_data = self._load_domain_data(domain)
            self.data.extend(domain_data)
            all_classes.update([item[1] for item in domain_data])

        # Handle class selection and mapping
        if selected_classes is None:
            self.selected_classes = sorted(list(all_classes))
        else:
            self.selected_classes = selected_classes

        # Create class name to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.selected_classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        # Create domain name to index mapping
        self.domain_to_idx = {domain: idx for idx, domain in enumerate(self.domains)}
        self.idx_to_domain = {idx: domain for domain, idx in self.domain_to_idx.items()}

        # Filter data to only include selected classes
        self.data = [(path, cls_name, domain) for path, cls_name, domain in self.data
                     if cls_name in self.class_to_idx]

        print(f"Selected classes ({len(self.selected_classes)}): {self.selected_classes}")
        print(f"Total samples: {len(self.data)}")
        print(f"Samples per domain: {self._get_domain_stats()}")

    def _load_domain_data(self, domain):
        """Load data for a specific domain"""
        label_file = os.path.join(self.data_root, domain, f"{domain}_{self.split}.txt")
        domain_data = []

        if not os.path.exists(label_file):
            print(f"Warning: Label file not found: {label_file}")
            return domain_data

        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        img_path = parts[0]
                        class_idx = int(parts[1])

                        # Extract class name from path (assuming format: domain/class_name/image.jpg)
                        path_parts = img_path.split('/')
                        if len(path_parts) >= 2:
                            class_name = path_parts[-2]  # Class name is the parent directory

                            # Construct full image path
                            full_img_path = os.path.join(self.data_root, domain, img_path)

                            if os.path.exists(full_img_path):
                                domain_data.append((full_img_path, class_name, domain))

        print(f"Loaded {len(domain_data)} samples from {domain}")
        return domain_data

    def _get_domain_stats(self):
        """Get sample count per domain"""
        domain_counts = {}
        for _, _, domain in self.data:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return domain_counts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name, domain = self.data[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='white')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get labels
        class_label = self.class_to_idx[class_name]

        if self.return_domain_label:
            domain_label = self.domain_to_idx[domain]
            return image, class_label, domain_label
        else:
            return image, class_label


def create_domainnet_dataloader(data_root="../data/DomainNet",
                                domains=None,
                                selected_classes=None,
                                split="train",
                                batch_size=32,
                                shuffle=True,
                                num_workers=4,
                                image_size=224,
                                return_domain_label=True):
    """
    Create DomainNet DataLoader with custom settings

    Args:
        data_root: Path to DomainNet data
        domains: List of domains to use
        selected_classes: List of class names to use
        split: 'train' or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        image_size: Size to resize images to
        return_domain_label: Whether to return domain labels

    Returns:
        DataLoader, class_to_idx dict, domain_to_idx dict
    """

    # Define transforms
    if split == "train":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Create dataset
    dataset = DomainNetDataset(
        data_root=data_root,
        domains=domains,
        selected_classes=selected_classes,
        split=split,
        transform=transform,
        return_domain_label=return_domain_label
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if split == "train" else False
    )

    return dataloader, dataset.class_to_idx, dataset.domain_to_idx


# Example usage
if __name__ == "__main__":
    # Example 1: Use specific classes and domains
    selected_classes = ['airplane', 'bird', 'car', 'dog', 'flower']
    selected_domains = ['clipart', 'real', 'sketch']

    train_loader, class_to_idx, domain_to_idx = create_domainnet_dataloader(
        domains=selected_domains,
        selected_classes=selected_classes,
        split="train",
        batch_size=16,
        return_domain_label=True
    )

    print(f"\nClass mapping: {class_to_idx}")
    print(f"Domain mapping: {domain_to_idx}")

    # Test the dataloader
    for batch_idx, (images, class_labels, domain_labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Images shape: {images.shape}")
        print(f"Class labels: {class_labels}")
        print(f"Domain labels: {domain_labels}")

        if batch_idx == 2:  # Just show first 3 batches
            break

    print(f"\nDataLoader created successfully!")
    print(f"Total batches: {len(train_loader)}")