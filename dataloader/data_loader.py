from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from ToBlackAndWite import *


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

    elif source == 'USPS':
        dataset = datasets.USPS(root='./data', train=True, download=True, transform=transform_mnist_rs)
        dataset_test = datasets.USPS(root='./data', train=False, download=True, transform=transform_mnist_rs)

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
