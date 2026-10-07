import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

from data_loader import train_loader, val_loader
from resnet18 import resnet18




EPOCHS = 20
LEARNING_RATE = 0.001

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results/plots")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)



if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("Using device:", device)



model = resnet18(num_classes=10)

model = model.to(device)



criterion = nn.CrossEntropyLoss()



optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)



train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

best_val_accuracy = 0.0




for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )


    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)


        # Clear previous gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Statistics

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


        progress_bar.set_postfix(
            loss=loss.item()
        )


    train_loss = running_loss / total

    train_accuracy = (
        100.0 * correct / total
    )



    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_running_loss += (
                loss.item() * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_running_loss / val_total
    )

    val_accuracy = (
        100.0 * val_correct / val_total
    )



    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)


    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )



    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        model_path = (
            MODEL_DIR /
            "best_resnet18.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            f"Best model saved: {model_path}"
        )



plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "loss_curve.png"
)

plt.close()



plt.figure(figsize=(8, 5))

plt.plot(
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "accuracy_curve.png"
)

plt.close()


print("\nTraining completed.")

print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.2f}%"
)

print(
    "Best model:",
    MODEL_DIR / "best_resnet18.pth"
)