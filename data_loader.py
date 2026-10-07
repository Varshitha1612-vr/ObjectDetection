import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



DATA_DIR = Path("data")

BATCH_SIZE = 128
NUM_WORKERS = 0
SEED = 42

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]



torch.manual_seed(SEED)

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616
)



train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])



test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])



print("Downloading/loading CIFAR-10 dataset...")


train_full = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform
)


val_full = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=test_transform
)


test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=test_transform
)



generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    len(train_full),
    generator=generator
)


train_indices = indices[:45000]
val_indices = indices[45000:]


train_dataset = Subset(
    train_full,
    train_indices
)


val_dataset = Subset(
    val_full,
    val_indices
)



train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)



print("\nDataset information")
print("-------------------")

print("Training images   :", len(train_dataset))
print("Validation images :", len(val_dataset))
print("Test images       :", len(test_dataset))

print("\nClasses:")
print(CLASS_NAMES)


images, labels = next(iter(train_loader))

print("\nBatch information")
print("-----------------")

print("Image batch shape :", images.shape)
print("Label batch shape :", labels.shape)

print("\nFirst few labels:")
print(labels[:10])



def show_images(images, labels):

    # Unnormalize images
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    images = images.cpu() * std + mean
    images = torch.clamp(images, 0, 1)

    plt.figure(figsize=(10, 8))

    for i in range(min(16, len(images))):

        plt.subplot(4, 4, i + 1)

        image = images[i].permute(1, 2, 0)

        plt.imshow(image)

        plt.title(CLASS_NAMES[labels[i].item()])

        plt.axis("off")

    plt.tight_layout()

    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "dataset_samples.png"

    plt.savefig(output_path)

    plt.show()

    print("\nSaved sample images to:")
    print(output_path)


show_images(images, labels)