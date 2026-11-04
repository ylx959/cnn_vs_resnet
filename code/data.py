import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_transforms():
    # 訓練集使用 Data Augmentation 防止 overfitting
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # 驗證集和測試集只做 Normalization
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return transform_train, transform_test


def split_indices(total=50000, val_size=5000):
    # 種子固定，讓每次實驗都用同一份 train/val 切分
    generator = torch.Generator().manual_seed(1234)
    indices = torch.randperm(total, generator=generator).tolist()
    return indices[val_size:], indices[:val_size]


def get_dataloaders(batch_size=128, num_workers=4):
    transform_train, transform_test = get_transforms()
    train_indices, val_indices = split_indices()

    # 同一份訓練資料載入兩次，因為訓練和驗證要用不同的 transform
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset, train_indices), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, persistent_workers=True)
    valloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(valset, val_indices), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, persistent_workers=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader
