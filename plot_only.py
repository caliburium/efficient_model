from torchvision.datasets import CIFAR10, SVHN, Imagenette, STL10
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():
    transform_blur = transforms.Compose([
        transforms.Resize((96, 96), interpolation=InterpolationMode.NEAREST),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 2.5)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform = transforms.Compose([
        transforms.Resize((96, 96), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    imagenette_dataset = Imagenette(root='./data', split='train', download=False, transform=transform_blur)
    stl10_dataset = STL10(root='./data', split='train', download=True, transform=transform_blur)
    svhn_dataset = SVHN(root='./data', split='train', download=True, transform=transform)
    cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Plotting images from each dataset
    plot_images(imagenette_dataset, "Imagenette Dataset")
    plot_images(stl10_dataset, "STL10 Dataset")
    plot_images(svhn_dataset, "SVHN Dataset")
    plot_images(cifar10_dataset, "CIFAR-10 Dataset")


def plot_images(dataset, dataset_name, num_images=6):
    plt.figure(figsize=(20, 4))
    plt.suptitle(dataset_name)
    for i in range(num_images):
        image, _ = dataset[i]
        image = image.permute(1, 2, 0)  # Changing tensor shape to (H, W, C)
        image = (image + 1) / 2  # Unnormalize
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
