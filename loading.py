import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.Imagenette(root='./data', split='val', transform=transform, download=True)

target_class = 0

target_indices = []
for i, target in enumerate(test_dataset._samples):
	if target[1] == target_class:
		target_indices.append(i)

test_dataset_class0 = Subset(test_dataset, target_indices)

