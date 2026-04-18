import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from adam_type import RMSpropAdaptiveMomentum
from sgd_am import SGDAdaptiveMomentum


DATASET_NAME = "cifar100"
NUM_CLASSES = 100

NUM_EPOCH = 200
BATCH_SIZE = 128
NUM_WORKERS = 8

OPTIMIZER_NAME = "sgd_am"
MODEL_NAME = "resnet50"


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

    train_loader, val_loader = setup_dataloader(dataset_name=DATASET_NAME)

    model = models.resnet50(num_classes=NUM_CLASSES)
    model.to(device)

    if OPTIMIZER_NAME == "adam_type":
        optimizer = RMSpropAdaptiveMomentum(model.parameters(), lr=0.1, alpha=0.99, eps=1, weight_decay=5e-7, momentum=0.9)
    elif OPTIMIZER_NAME == "sgd_am":
        optimizer = SGDAdaptiveMomentum(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-7)
    else:
        raise ValueError(f"Optimizer {OPTIMIZER_NAME} not supported")

    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=1e-4)


    criterion = nn.CrossEntropyLoss(reduction="mean")

    for epoch in range(NUM_EPOCH):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item()}")
        scheduler.step()

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
        accuracy = 100. * correct / total
        print(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()