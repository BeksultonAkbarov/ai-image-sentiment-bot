import os
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
from sklearn.metrics import accuracy_score


def prepare_dataloaders(data_dir, batch_size=16, val_pct=0.2):
    # Standard ImageNet normalization
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir)
    # split dataset
    val_size = int(len(dataset) * val_pct)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    # assign transforms
    train_ds.dataset.transform = train_transforms
    val_ds.dataset.transform = val_transforms

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # compute class counts for weighting
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        class_counts[label] += 1

    return train_loader, val_loader, dataset.classes, class_counts


def build_model(num_classes, device, freeze_backbone=True):
    model = models.resnet18(pretrained=True)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def train(model, train_loader, val_loader, device, class_counts, epochs=12, lr=1e-4, out_path='image_sentiment.pth'):
    # class weights (inverse frequency)
    counts = torch.tensor(class_counts, dtype=torch.float32)
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.sum()
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        since = time.time()
        model.train()
        train_losses = []
        train_preds, train_trues = [], []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_preds.extend(torch.argmax(out, 1).cpu().tolist())
            train_trues.extend(yb.cpu().tolist())

        train_loss = sum(train_losses) / max(1, len(train_losses))
        train_acc = accuracy_score(train_trues, train_preds) if train_trues else 0.0

        # validation
        model.eval()
        val_losses = []
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                val_preds.extend(torch.argmax(out, 1).cpu().tolist())
                val_trues.extend(yb.cpu().tolist())

        val_loss = sum(val_losses) / max(1, len(val_losses))
        val_acc = accuracy_score(val_trues, val_preds) if val_trues else 0.0

        scheduler.step(val_loss)

        elapsed = time.time() - since
        print(f'Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}  time={elapsed:.1f}s')

        # early stopping + save best
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({'model_state': best_model_wts, 'classes': getattr(train_loader.dataset.dataset, 'classes', None)}, out_path)
            patience_counter = 0
            print('  Saved best model')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')
                break

    # load best weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    data_dir = 'data'
    batch_size = 16
    epochs = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, classes, class_counts = prepare_dataloaders(data_dir, batch_size=batch_size)
    print('Classes:', classes)
    print('Class counts:', class_counts)

    model = build_model(num_classes=len(classes), device=device, freeze_backbone=True)
    # fine-tune last layer only for a few epochs, then unfreeze optionally
    model = train(model, train_loader, val_loader, device, class_counts, epochs=epochs, lr=1e-4, out_path='image_sentiment.pth')

    print('Training finished. Best model saved to image_sentiment.pth')


if __name__ == '__main__':
    main()
