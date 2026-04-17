import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from adam_type import RMSpropAdaptiveMomentum
from sgd_am import SGDAdaptiveMomentum
from fsgd_am import FractionalSGDAdaptiveMomentum


DATASET_NAME = "cifar100"
NUM_CLASSES = 100

NUM_EPOCH = 500
BATCH_SIZE = 256
NUM_WORKERS = 8

OPTIMIZER_NAME = "adam_type"
MODEL_NAME = "resnet50"
OUTPUT_DIR = Path("outputs")


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                        std=(0.2675, 0.2565, 0.2761))
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                        std=(0.2675, 0.2565, 0.2761))
])


def setup_dataloader(dataset_name: str):
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=TRAIN_TRANSFORM)
        val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=VAL_TRANSFORM)
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=TRAIN_TRANSFORM)
        val_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=VAL_TRANSFORM)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = setup_dataloader(dataset_name=DATASET_NAME)

    model = models.resnet50(num_classes=NUM_CLASSES)
    model.to(device)

    if OPTIMIZER_NAME == "adam_type":
        optimizer = RMSpropAdaptiveMomentum(model.parameters(), lr=0.1, alpha=0.99, eps=1, weight_decay=5e-7, momentum=0.9)
    elif OPTIMIZER_NAME == "sgd_am":
        optimizer = SGDAdaptiveMomentum(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-7)
    elif OPTIMIZER_NAME == "fsgd_am":
        optimizer = FractionalSGDAdaptiveMomentum(model.parameters())
    else:
        raise ValueError(f"Optimizer {OPTIMIZER_NAME} not supported")

    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=1e-4)


    criterion = nn.CrossEntropyLoss(reduction="mean")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(NUM_EPOCH):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            train_correct += (predicted == target).sum().item()
            train_total += target.size(0)
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item()}")
        scheduler.step()

        avg_train_loss = train_loss / train_total
        train_accuracy = 100. * train_correct / train_total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item()}")
        avg_val_loss = val_loss / total
        val_accuracy = 100. * correct / total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_accuracy)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
        )

    history_path = OUTPUT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_{OPTIMIZER_NAME}_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path.resolve()}")


if __name__ == "__main__":
    main()
