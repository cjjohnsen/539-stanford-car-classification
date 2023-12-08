from torch.utils.data import DataLoader
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def get_data_loaders(data_root, batch_size, num_workers):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.75, 1)),
        transforms.RandomRotation(degrees=30),
        #transforms.ColorJitter(hue=0.05, saturation=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_root = f'{data_root}/car_data/train'
    test_root = f'{data_root}/car_data/test'

    if not os.path.exists(train_root) or not os.path.exists(test_root):
        print('ERROR: Dataset is incomplete, be sure to download and process data before getting loaders.')
        exit(1)

    train_dataset = datasets.ImageFolder(
        root=train_root,
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=test_root,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader