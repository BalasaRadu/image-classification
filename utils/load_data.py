import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, class_to_idx=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]))
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[idx, 1]

        if self.class_to_idx is not None:
            label = self.class_to_idx[label]
        else:
            label = int(label) - 1

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(dataset_name, aug_type='none'):
    if dataset_name == 'faces':
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        geo_transforms = [
            transforms.Resize((72, 72)),
            transforms.RandomCrop(64),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        color_transforms = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1)
    ]   
    else:
        mean, std = [0.7646, 0.5464, 0.5712], [0.1410, 0.1532, 0.1706]
        geo_transforms = [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
        ]
        color_transforms = [
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ]

    

    base_transforms = [transforms.Resize((64, 64))]
    end_transforms = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    if aug_type == 'none':
        train_transform = transforms.Compose(base_transforms + end_transforms)
    elif aug_type == 'geo':
        train_transform = transforms.Compose(geo_transforms + end_transforms)
    elif aug_type == 'color':
        train_transform = transforms.Compose(base_transforms + color_transforms + end_transforms)
    elif aug_type == 'all':
        train_transform = transforms.Compose(geo_transforms + color_transforms + end_transforms)
    else:
        train_transform = transforms.Compose(base_transforms + end_transforms)

    test_transform = transforms.Compose(base_transforms + end_transforms)

    return train_transform, test_transform

def get_balanced_sampler(dataset):
    labels = []
    for idx in range(len(dataset)):
        label = dataset.annotations.iloc[idx, 1]
        if dataset.class_to_idx is not None:
            label = dataset.class_to_idx[label]
        else:
            label = int(label) - 1
        labels.append(label)
        
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def create_face_dataloaders(train_csv, test_csv, root_dir, batch_size=64, aug_type='none', use_sampler=False):
    train_transform, test_transform = get_transforms('faces', aug_type)

    train_dataset = CustomImageDataset(csv_file=train_csv, root_dir=root_dir, transform=train_transform)
    val_dataset = CustomImageDataset(csv_file=test_csv, root_dir=root_dir, transform=test_transform)

    if use_sampler:
        sampler = get_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
        print("[INFO] WeightedRandomSampler ACTIVAT pe setul FACES.")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

def create_mole_dataloaders(train_csv, test_csv, train_dir, test_dir, batch_size=64, aug_type='none', use_sampler=False):
    mole_class_to_idx = {
        'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
    }

    train_transform, test_transform = get_transforms('moles', aug_type)

    train_dataset = CustomImageDataset(csv_file=train_csv, root_dir=train_dir, transform=train_transform, class_to_idx=mole_class_to_idx)
    val_dataset = CustomImageDataset(csv_file=test_csv, root_dir=test_dir, transform=test_transform, class_to_idx=mole_class_to_idx)

    if use_sampler:
        sampler = get_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
        print("[INFO] WeightedRandomSampler ACTIVAT pe setul MOLES.")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader