import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def load_dataset():
    validation_split = 0.2

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,), ),
                                    ])
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, train=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST("./data", download=True , train=False, transform=transform)
    dataset_size = len(train_set)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    trainSet = torch.utils.data.Subset(train_set, train_indices)
    valSet = torch.utils.data.Subset(train_set, val_indices)
    train_loader = torch.utils.data.DataLoader(trainSet, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valSet, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    return train_loader, val_loader , test_loader