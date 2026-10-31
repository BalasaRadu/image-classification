import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix

from models.mlp import MLP
from models.cnn import CNN
from utils.load_data import create_face_dataloaders, create_mole_dataloaders, CustomImageDataset, get_balanced_sampler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int = 20,
    writer: SummaryWriter = None,
    scheduler = None
):
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs} [Train]'):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        train_acc = train_correct / train_total
        train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{epochs} [Val]'):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_dataloader)

        history['train_accs'].append(train_acc)
        history['train_losses'].append(train_loss)
        history['val_accs'].append(val_acc)
        history['val_losses'].append(val_loss)

        current_lr = optimizer.param_groups[-1]['lr']

        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f}\n"
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n"
        )

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("Learning_Rate", current_lr, epoch)
            
        if scheduler:
            scheduler.step()

    return history

def evaluate_metrics(model, dataloader, dataset_name, arch_name, run_name):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    f1_micro = f1_score(all_targets, all_preds, average='micro')
    
    print(f"[{arch_name.upper()} - {dataset_name}] Accuracy: {acc:.4f} | F1-Macro: {f1_macro:.4f} | F1-Micro: {f1_micro:.4f}")
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predictii')
    plt.ylabel('Valori Reale')
    
    out_dir = "rezultate_grafice"
    os.makedirs(out_dir, exist_ok=True)
    
    file_name = f'matrice_confuzie_{run_name}.png'
    file_path = os.path.join(out_dir, file_name)
    
    plt.title(f'Matrice Confuzie\n{run_name}', fontsize=10)
    plt.savefig(file_path)
    plt.close()
    
    print(f"Matricea a fost salvata in {file_path}")

def generate_submission(model, dataset_name, arch_name, run_name, custom_test_csv=None):
    model.eval()
    model.to(DEVICE)

    if arch_name == 'resnet18':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        if dataset_name == 'faces':
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.7646, 0.5464, 0.5712], std=[0.1410, 0.1532, 0.1706])
            ])

    col_image, col_label = 'Id', 'label' if dataset_name == 'faces' else 'Prediction'

    if dataset_name == 'faces':
        test_csv = custom_test_csv if custom_test_csv else 'data/facial_expressions/splits/local_test.csv'
        root_dir = 'data/facial_expressions/DATASET/test'
        idx_to_label = {i: i + 1 for i in range(7)}
    else:
        test_csv = custom_test_csv if custom_test_csv else 'data/skin_moles/test.csv'
        root_dir = 'data/skin_moles/kaggle_test'
        idx_to_label = {i: i for i in range(7)}

    if not os.path.exists(test_csv):
        print(f"[EROARE] Nu am gasit fisierul CSV la calea: {test_csv}")
        return

    test_df = pd.read_csv(test_csv)
    image_col = test_df.columns[0]

    image_names = []
    predictions = []

    print(f"\n[SUBMISIE] Incepe generarea predictiilor din '{test_csv}'...")
    for img_rel_path in tqdm(test_df[image_col], desc='Generare predictii'):
        
        nume_fisier = os.path.basename(str(img_rel_path))
        img_full_path = None
        
        for subfolder in ['', '1', '2', '3', '4', '5', '6', '7']:
            cale_temp = os.path.join(root_dir, subfolder, nume_fisier) if subfolder else os.path.join(root_dir, nume_fisier)
            if os.path.exists(cale_temp):
                img_full_path = cale_temp
                break

        if img_full_path is None:
            print(f"[AVERTISMENT] Imaginea nu a fost gasita: {nume_fisier}")
            continue

        try:
            image = Image.open(img_full_path).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)
                pred_idx = torch.argmax(output, dim=1).item()

            image_names.append(nume_fisier)
            predictions.append(idx_to_label[pred_idx])
        except Exception as e:
            print(f"[EROARE] Probleme la procesarea imaginii {nume_fisier}: {e}")

    out_dir = 'submisii'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'submisie_{run_name}.csv')

    submission_df = pd.DataFrame({col_image: image_names, col_label: predictions})
    submission_df.to_csv(out_path, index=False)
    
    print(f'\nSubmisie salvata cu succes in: {out_path}')
    print(submission_df.head(10).to_string(index=False))

def compute_class_weights(dataset):
    labels = []
    for idx in range(len(dataset)):
        label = dataset.annotations.iloc[idx, 1]
        if dataset.class_to_idx is not None:
            label = dataset.class_to_idx[label]
        else:
            label = int(label) - 1
        labels.append(label)
        
    class_counts = np.bincount(labels)
    counts = torch.tensor(class_counts, dtype=torch.float)
    weights = counts.sum() / (len(counts) * counts)
    return weights

def get_imagenet_dataloaders(dataset_name, batch_size, use_sampler=False):
    imagenet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if dataset_name == 'faces':
        train_ds = CustomImageDataset('data/facial_expressions/splits/train.csv', 'data/facial_expressions', transform=imagenet_transform)
        val_ds = CustomImageDataset('data/facial_expressions/splits/local_test.csv', 'data/facial_expressions', transform=imagenet_transform)
    else:
        mole_class_to_idx = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
        train_ds = CustomImageDataset('data/skin_moles/train.csv', 'data/skin_moles/train', transform=imagenet_transform, class_to_idx=mole_class_to_idx)
        val_ds = CustomImageDataset('data/skin_moles/test.csv', 'data/skin_moles/test', transform=imagenet_transform, class_to_idx=mole_class_to_idx)

    if use_sampler:
        sampler = get_balanced_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
        print("[INFO] WeightedRandomSampler ACTIVAT pe dataloaderul de ImageNet.")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['faces', 'moles'], required=True)
    parser.add_argument('--arch', type=str, choices=['mlp', 'cnn', 'resnet18'], default='mlp')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--aug_type', type=str, choices=['none', 'geo', 'color', 'all'], default='none')
    parser.add_argument('--balance_classes', action='store_true')
    parser.add_argument('--balance_method', type=str, choices=['loss', 'sampler'], default='sampler')    
    parser.add_argument('--freeze_backbone', action='store_true', help='Ingheata backbone-ul')
    parser.add_argument('--warmup', action='store_true', help='Activeaza learning rate warmup')
    parser.add_argument('--test_model', type=str, default=None, help='Calea catre modelul .pth salvat')
    parser.add_argument('--test_csv', type=str, default=None, help='Calea catre fisierul kaggle_test.csv')
    
    args = parser.parse_args()

    use_weighted_loss = args.balance_classes and args.balance_method == 'loss'
    use_sampler = args.balance_classes and args.balance_method == 'sampler'

    if args.arch == 'resnet18':
        train_loader, val_loader = get_imagenet_dataloaders(args.dataset, args.batch_size, use_sampler)
    else:
        if args.dataset == 'faces':
            train_loader, val_loader = create_face_dataloaders(
                train_csv='data/facial_expressions/splits/train.csv',
                test_csv='data/facial_expressions/splits/local_test.csv',
                root_dir='data/facial_expressions',
                batch_size=args.batch_size,
                aug_type=args.aug_type,
                use_sampler=use_sampler
            )
        else:
            train_loader, val_loader = create_mole_dataloaders(
                train_csv='data/skin_moles/train.csv',
                test_csv='data/skin_moles/test.csv',
                train_dir='data/skin_moles/train',
                test_dir='data/skin_moles/test',
                batch_size=args.batch_size,
                aug_type=args.aug_type,
                use_sampler=use_sampler
            )

    scheduler = None
    
    if args.arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 7)
        model = model.to(DEVICE)
        
        if args.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
            
            optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
        else:
            backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
            head_params = [p for n, p in model.named_parameters() if n.startswith("fc.")]
            
            optimizer = optim.Adam([
                {"params": backbone_params, "lr": 1e-4},
                {"params": head_params, "lr": args.lr},
            ], weight_decay=1e-4)

        if args.warmup:
            warmup_epochs = 3
            warmup = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
            
    elif args.arch == 'mlp':
        model = MLP(in_features=3*64*64, hidden_dim=512, num_classes=7, use_dropout=args.dropout).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.arch == 'cnn':
        model = CNN(num_classes=7, use_batchnorm=args.batchnorm, dataset_name=args.dataset, use_dropout=args.dropout).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if use_weighted_loss:
        weights = compute_class_weights(train_loader.dataset).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.arch == 'resnet18':
        run_name = f"resnet18_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_name = f"{args.arch}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.test_model is not None:
        print(f"\n[INFO] Modul INFERENCE ACTIVAT.")
        print(f"-> Incarcare ponderi din: {args.test_model}")
        if args.test_csv:
            print(f"-> Folosire dataset test extern: {args.test_csv}")
            
        if os.path.exists(args.test_model):
            model.load_state_dict(torch.load(args.test_model, map_location=DEVICE))
            generate_submission(model, args.dataset, args.arch, f"kaggle_inference_{args.dataset}", custom_test_csv=args.test_csv)
        else:
            print(f"[EROARE] Fisierul de ponderi nu a fost gasit: {args.test_model}")
        exit(0)

    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    history = train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        writer=writer,
        scheduler=scheduler 
    )

    evaluate_metrics(model, val_loader, args.dataset, args.arch, run_name)
    writer.close()

    model_dir = "modele_salvate"
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f"{run_name}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n[INFO] Ponderile modelului au fost salvate in: {checkpoint_path}")

    generate_submission(model, args.dataset, args.arch, run_name)