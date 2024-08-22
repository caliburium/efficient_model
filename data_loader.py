import numpy as np
from PIL import ImageCms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset, default_loader
import sys
import os
from copy import deepcopy


class ToBlackAndWhite(object):
    def __call__(self, img):
        img = img.convert('L')
        img = np.array(img)
        img = (img > 127).astype(np.uint8) * 255
        img = Image.fromarray(img)
        return img


def data_loader(source, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_bw = transforms.Compose([
        ToBlackAndWhite(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnist = transforms.Compose([
        transforms.Pad(2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnist_rs = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_stl10 = transforms.Compose([
        transforms.Resize((32, 32), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if source == 'MNIST':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    elif source == 'MNIST_RS':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist_rs)
        dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist_rs)

    elif source == 'SVHN':
        dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    elif source == 'SVHN_BW':
        dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_bw)
        dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform_bw)

    elif source == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif source == 'STL10':
        dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_stl10)
        dataset_test = datasets.STL10(root='./data', split='test', download=True, transform=transform_stl10)
    else:
        print("no source")

    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=8)

    return loader, test_loader



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class DG_Dataset(Dataset):
    def __init__(self, root_dir, domain, split, get_domain_label=False, get_cluster=False, color_jitter=True, min_scale=0.8):
        self.root_dir = root_dir
        self.domain = domain
        self.split = split
        self.get_domain_label = get_domain_label
        self.get_cluster = get_cluster
        self.color_jitter = color_jitter
        self.min_scale = min_scale
        self.set_transform(self.split)
        self.loader = default_loader
        
        self.load_dataset()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        image = self.loader(path)
        image = self.transform(image)
        output = [image, target]        
        
        if self.get_domain_label:
            domain = np.copy(self.domains[index])
            domain = np.int64(domain)
            output.append(domain)
            
        if self.get_cluster:
            cluster = np.copy(self.clusters[index])
            cluster = np.int64(cluster)
            output.append(cluster)

        return tuple(output)
    
    def find_classes(self, dir_name):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir_name) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def load_dataset(self):
        total_samples = []
        self.domains = np.zeros(0)
        
        classes, class_to_idx = self.find_classes(self.root_dir + self.domain[0] + '/')
        self.num_class = len(classes)
        for i, item in enumerate(self.domain):
            path = self.root_dir + item + '/' 
            samples = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
            total_samples.extend(samples)
            self.domains = np.append(self.domains, np.ones(len(samples)) * i)
            
        self.clusters = np.zeros(len(self.domains), dtype=np.int64)
        self.images = [s[0] for s in total_samples]
        self.labels = [s[1] for s in total_samples]

    def set_cluster(self, cluster_list):
        if len(cluster_list) != len(self.images):
            raise ValueError("The length of cluster_list must to be same as self.images")
        else:
            self.clusters = cluster_list

    def set_domain(self, domain_list):
        if len(domain_list) != len(self.images):
            raise ValueError("The length of domain_list must to be same as self.images")
        else:
            self.domains = domain_list
            
    def set_transform(self, split):
        if split == 'train':
            if self.color_jitter:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(self.min_scale, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.4, .4, .4, .4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(self.min_scale, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        elif split == 'val' or split == 'test':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            raise Exception('Split must be train or val or test!!')
        

def random_split_dataloader(data, data_root, source_domain, target_domain, batch_size, 
                   get_domain_label=False, get_cluster=False, num_workers=4, color_jitter=True, min_scale=0.8):
    if data=='VLCS': 
        split_rate = 0.7
    else: 
        split_rate = 0.9
    source = DG_Dataset(root_dir=data_root, domain=source_domain, split='val',
                                     get_domain_label=False, get_cluster=False, color_jitter=color_jitter, min_scale=min_scale)
    source_train, source_val = random_split(source, [int(len(source)*split_rate), len(source)-int(len(source)*split_rate)])
    source_train = deepcopy(source_train)
    source_train.dataset.split='train'
    source_train.dataset.set_transform('train')
    source_train.dataset.get_domain_label=get_domain_label
    source_train.dataset.get_cluster=get_cluster
    
    target_test =  DG_Dataset(root_dir=data_root, domain=target_domain, split='test',
                                   get_domain_label=False, get_cluster=False)
    
    print('Train: {}, Val: {}, Test: {}'.format(len(source_train), len(source_val), len(target_test)))
    
    source_train = DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    source_val  = DataLoader(source_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    target_test = DataLoader(target_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return source_train, source_val, target_test


def split_domain(domains, split_idx, print_domain=True):
    source_domain = deepcopy(domains)
    target_domain = [source_domain.pop(split_idx)]
    if print_domain:
        print('Source domain: ', end='')
        for domain in source_domain:
            print(domain, end=', ')
        print('Target domain: ', end='')
        for domain in target_domain:
            print(domain)
    return source_domain, target_domain


domain_map = {
    'PACS': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'PACS_random_split': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'OfficeHome': ['Art', 'Clipart', 'Product', 'RealWorld'],
    'VLCS': ['Caltech', 'Labelme', 'Pascal', 'Sun']
}

def get_domain(name):
    if name not in domain_map:
        raise ValueError('Name of dataset unknown %s' %name)
    return domain_map[name]