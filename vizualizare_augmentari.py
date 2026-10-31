import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

faces_csv = 'data/facial_expressions/splits/train.csv'
faces_dir = 'data/facial_expressions/'
moles_csv = 'data/skin_moles/train.csv'
moles_dir = 'data/skin_moles/train'

def get_vis_transforms(dataset_name, aug_type='none'):
    if dataset_name == 'faces':
        geo_transforms = [
            transforms.Resize((72, 72)),
            transforms.RandomCrop(64),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=1.0),
        ]
        color_transforms = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=1.0)
        ]   
    else:
        geo_transforms = [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(degrees=180),
        ]
        color_transforms = [
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ]

    base_transforms = [transforms.Resize((64, 64))]

    if aug_type == 'none':
        return transforms.Compose(base_transforms)
    elif aug_type == 'geo':
        return transforms.Compose(geo_transforms)
    elif aug_type == 'color':
        return transforms.Compose(base_transforms + color_transforms)
    elif aug_type == 'all':
        return transforms.Compose(geo_transforms + color_transforms)
    
    return transforms.Compose(base_transforms)
    
def visualize_dataset(csv_path, img_dir, dataset_name):
    if not os.path.exists(csv_path):
        print(f"Error: no file {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    img_col = df.columns[0]
    label_col = df.columns[1]
    
    unique_classes = sorted(df[label_col].unique())
    
    fig, axes = plt.subplots(nrows=len(unique_classes), ncols=2, figsize=(8, 3 * len(unique_classes)))
    fig.suptitle(f'Augmentari set de date: {dataset_name.upper()}', fontsize=16)
    
    for idx, cls in enumerate(unique_classes):
        if idx < 2:
            aug_type = 'geo'
            label_text = 'Geometric'
        elif idx < 4:
            aug_type = 'color'
            label_text = 'Culoare'
        else:
            aug_type = 'all'
            label_text = 'Geometric + Culoare'
            
        # Folosim functia noua din acest script
        transform = get_vis_transforms(dataset_name, aug_type)
            
        sample_row = df[df[label_col] == cls].iloc[0]
        img_name = str(sample_row[img_col])
        img_path = os.path.join(img_dir, img_name)
        
        original_img = Image.open(img_path).convert("RGB")
        augmented_img = transform(original_img)
        
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title(f'Clasa: {cls}\nOriginal')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(augmented_img)
        axes[idx, 1].set_title(f'Clasa: {cls}\nAugmentat ({label_text})')
        axes[idx, 1].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_file = f'grid_augmentari_{dataset_name}.png'
    plt.savefig(out_file)
    plt.close()
    print(f"Image saved as {out_file}")

if __name__ == '__main__':
    visualize_dataset(faces_csv, faces_dir, 'faces')
    visualize_dataset(moles_csv, moles_dir, 'moles')