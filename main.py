import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import os
import random
import numpy as np

from fosgd_am import FractionalOrderSGDAdaptiveMomentum
from fosgdmr import FractionalOrderSGDMomentum


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def seed_worker(worker_id):
    worker_seed = worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


DATASET_NAME = "cifar10"
NUM_CLASSES = 10

NUM_EPOCH = 200
BATCH_SIZE = 128
NUM_WORKERS = 16

OPTIMIZER_NAME = "fosgdmr"
MODEL_NAME = "resnet34"
OUTPUT_DIR = Path("outputs")


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])


def setup_dataloader(dataset_name: str):
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=TRAIN_TRANSFORM
        )
        val_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=VAL_TRANSFORM
        )

    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=TRAIN_TRANSFORM
        )
        val_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=VAL_TRANSFORM
        )
    else:
        raise ValueError(f"{dataset_name} not supported")

    g = torch.Generator()
    g.manual_seed(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    return train_loader, val_loader


def main():
    seed_everything(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = setup_dataloader(DATASET_NAME)

    model = models.resnet34(num_classes=NUM_CLASSES)
    model = model.to(device)

    # =============================
    # Optimizer
    # =============================
    
    if OPTIMIZER_NAME == "fosgd_am":
        optimizer = FractionalOrderSGDAdaptiveMomentum(model.parameters(), lr=0.1)
        
    elif OPTIMIZER_NAME == "fosgdmr":
        optimizer = FractionalOrderSGDMomentum(model.parameters(), lr=0.1, beta=0.9)
        print("OK")

    else:
        raise ValueError("optimizer not supported")

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCH,
        eta_min=1e-4
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")

    # =============================
    # History lưu metric
    # =============================
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": []
    }

    # =============================
    # Train loop
    # =============================
    for epoch in range(NUM_EPOCH):

        # -------------------------
        # TRAIN
        # -------------------------
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0 

        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)

            if batch_idx % 100 == 0:
                print(
                    f"[Train] Epoch {epoch} "
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss {loss.item():.4f}"
                )

        avg_train_loss = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):

                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item() * data.size(0)

                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # -------------------------
        # Save history
        # -------------------------
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCH}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

    # =============================
    # Save history json
    # =============================
    save_path = OUTPUT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_{OPTIMIZER_NAME}_history.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"Saved history: {save_path}")


if __name__ == "__main__":
    main()