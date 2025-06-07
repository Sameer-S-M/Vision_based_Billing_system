from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR100
import torch

def get_cifar10(data_path: str = './data'):

    transform = Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR100(data_path, train=True, download=True, transform=transform)
    testset = CIFAR100(data_path, train=False, download=True, transform=transform)

    return testset, trainset

def prepare_dataset(num_partitions: int,
                     batch_size: int,
                       val_ratio: float = 0.1):
    trainset, testset = get_cifar10()

    num_images = len(trainset)//num_partitions

    partition_len = [num_images] * num_partitions

    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))
    
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_,[num_train,num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))
    
    testloaders = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloaders