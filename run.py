import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from torch.optim import SGD
from fosgd_am import FractionalOrderSGDAdaptiveMomentum
from fosgdmr import FractionalOrderSGDMomentum
from fosgdr import FractionalOrderSGD

# =============================
# Config
# =============================
DATASET_NAME = "cifar10"
NUM_CLASSES = 10

NUM_EPOCH = 200
BATCH_SIZE = 128
NUM_WORKERS = 16

ALPHAS = [0.9, 0.99, 0.999, 1.001, 1.01, 1.1]
MODEL_NAMES = ["densenet121"]
OPTIMIZER_NAMES = ["fosgdr"]
SEEDS = [0, 1, 2]

OUTPUT_DIR = Path("outputs")
RUN_DIR = OUTPUT_DIR / "runs"
SUMMARY_PATH = OUTPUT_DIR / "summary_all.json"


TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomRotation(10),      
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])


# =============================
# Seed
# =============================
def set_seed(seed: int = 42):
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


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================
# Data
# =============================
def setup_dataloader(dataset_name: str, seed: int):
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
    g.manual_seed(seed)

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


# =============================
# Model
# =============================
def build_model(model_name: str, num_classes: int):
    name = model_name.lower()

    if name == "resnet34":
        model = models.resnet34(weights=None)

        model.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)

    elif name == "densenet121":
        model = models.densenet121(weights=None)

        model.features.conv0 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        model.features.pool0 = nn.Identity()

        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


# =============================
# Optimizer
# =============================
def build_optimizer(optimizer_name: str, params, alpha: float):
    name = optimizer_name.lower()

    if name == "fosgd_am":
        try:
            return FractionalOrderSGDAdaptiveMomentum(params, fractional_alpha=alpha)
        except TypeError:
            return FractionalOrderSGDAdaptiveMomentum(params)
    
    elif name == "fosgdmr":
        try:
            return FractionalOrderSGDMomentum(params, fractional_alpha=alpha)
        except TypeError:
            return FractionalOrderSGDMomentum(params)
    elif name == "fosgdr":
        try:
            return FractionalOrderSGD(params, fractional_alpha=alpha)
        except TypeError:
            return FractionalOrderSGD(params)

    elif name == "sgdm":
        return SGD(
                params,
                lr=0.1,
                momentum=0.9,
                weight_decay=5e-4
            )
    elif name == "sgd":
        return SGD(params, lr=0.1)
    else:
        raise ValueError(f"Optimizer not supported: {optimizer_name}")


# =============================
# Train / Eval
# =============================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * data.size(0)
        pred = output.argmax(dim=1)
        total_correct += (pred == target).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f"  [Train] Batch {batch_idx}/{len(loader)} | Loss {loss.item():.4f}")

    return total_loss / total, 100.0 * total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(data)
        loss = criterion(output, target)

        total_loss += float(loss.item()) * data.size(0)
        pred = output.argmax(dim=1)
        total_correct += (pred == target).sum().item()
        total += target.size(0)

    return total_loss / total, 100.0 * total_correct / total


# =============================
# Run 1 experiment
# =============================
def run_experiment(alpha: float, model_name: str, optimizer_name: str, seed: int, device):
    set_seed(seed)
    train_loader, val_loader = setup_dataloader(DATASET_NAME, seed)

    model = build_model(model_name, NUM_CLASSES).to(device)
    optimizer = build_optimizer(optimizer_name, model.parameters(), alpha=alpha)
    
    scheduler = CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCH,
    eta_min=0
    )
    
    criterion = nn.CrossEntropyLoss(reduction="mean")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    for epoch in range(NUM_EPOCH):
        print(
            f"\n=== Alpha={alpha} | Model={model_name} | Optimizer={optimizer_name} | Seed={seed} | Epoch {epoch+1}/{NUM_EPOCH} ==="
        )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(current_lr))

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCH}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

    run_record = {
        "Alpha": alpha,
        "Model": model_name,
        "Seed": seed,
        "Optimizer": optimizer_name,
        "Train loss": history["train_loss"],
        "Train accuracy": history["train_acc"],
        "Val loss": history["val_loss"],
        "Val accuracy": history["val_acc"],
        "LR": history["lr"],
        "final": {
            "Train loss": history["train_loss"][-1],
            "Train accuracy": history["train_acc"][-1],
            "Val loss": history["val_loss"][-1],
            "Val accuracy": history["val_acc"][-1],
        }
    }

    return run_record


# =============================
# Main
# =============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    all_runs: List[Dict[str, Any]] = []

    for alpha in ALPHAS:
        for model_name in MODEL_NAMES:
            for optimizer_name in OPTIMIZER_NAMES:
                for seed in SEEDS:
                    run_record = run_experiment(alpha, model_name, optimizer_name, seed, device)
                    all_runs.append(run_record)

                    run_file = RUN_DIR / (
                        f"{DATASET_NAME}_{model_name}_{optimizer_name}_alpha{alpha}_seed{seed}.json"
                    )
                    with open(run_file, "w", encoding="utf-8") as f:
                        json.dump(run_record, f, ensure_ascii=False, indent=2)

                    print(f"Saved run: {run_file}")

    # =============================
    # Summary trung bình theo combo
    # =============================
    grouped = defaultdict(list)
    for run in all_runs:
        key = (run["Alpha"], run["Model"], run["Optimizer"])
        grouped[key].append(run)

    summary = []
    for (alpha, model_name, optimizer_name), runs in grouped.items():
        train_loss_curves = np.array([r["Train loss"] for r in runs], dtype=np.float64)
        train_acc_curves = np.array([r["Train accuracy"] for r in runs], dtype=np.float64)
        val_loss_curves = np.array([r["Val loss"] for r in runs], dtype=np.float64)
        val_acc_curves = np.array([r["Val accuracy"] for r in runs], dtype=np.float64)
        lr_curves = np.array([r["LR"] for r in runs], dtype=np.float64)

        summary.append({
            "Alpha": alpha,
            "Model": model_name,
            "Optimizer": optimizer_name,
            "Seeds": [r["Seed"] for r in runs],
            "mean_history": {
                "train_loss": train_loss_curves.mean(axis=0).tolist(),
                "train_acc": train_acc_curves.mean(axis=0).tolist(),
                "val_loss": val_loss_curves.mean(axis=0).tolist(),
                "val_acc": val_acc_curves.mean(axis=0).tolist(),
                "lr": lr_curves.mean(axis=0).tolist(),
            },
            "std_history": {
                "train_loss": train_loss_curves.std(axis=0).tolist(),
                "train_acc": train_acc_curves.std(axis=0).tolist(),
                "val_loss": val_loss_curves.std(axis=0).tolist(),
                "val_acc": val_acc_curves.std(axis=0).tolist(),
                "lr": lr_curves.std(axis=0).tolist(),
            },
            "final_mean": {
                "Train loss": float(np.mean([r["final"]["Train loss"] for r in runs])),
                "Train accuracy": float(np.mean([r["final"]["Train accuracy"] for r in runs])),
                "Val loss": float(np.mean([r["final"]["Val loss"] for r in runs])),
                "Val accuracy": float(np.mean([r["final"]["Val accuracy"] for r in runs])),
            },
            "final_std": {
                "Train loss": float(np.std([r["final"]["Train loss"] for r in runs])),
                "Train accuracy": float(np.std([r["final"]["Train accuracy"] for r in runs])),
                "Val loss": float(np.std([r["final"]["Val loss"] for r in runs])),
                "Val accuracy": float(np.std([r["final"]["Val accuracy"] for r in runs])),
            },
        })

    full_output = {
        "dataset": DATASET_NAME,
        "num_epoch": NUM_EPOCH,
        "batch_size": BATCH_SIZE,
        "alphas": ALPHAS,
        "models": MODEL_NAMES,
        "optimizers": OPTIMIZER_NAMES,
        "seeds": SEEDS,
        "runs": all_runs,
        "summary": summary,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
