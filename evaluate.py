import torch
import torch.nn as nn

from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import (
    test_loader,
    CLASS_NAMES
)

from resnet18 import resnet18


if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("Using device:", device)




model = resnet18(
    num_classes=10
)

model_path = Path(
    "models/best_resnet18.pth"
)


model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)


model = model.to(device)



criterion = nn.CrossEntropyLoss()




model.eval()

test_loss = 0.0

correct = 0
total = 0

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        test_loss += (
            loss.item() * images.size(0)
        )


        _, predictions = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )



test_loss = test_loss / total

test_accuracy = (
    100.0 * correct / total
)


precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.2f}%"
)

print(
    f"Precision     : {precision:.4f}"
)

print(
    f"Recall        : {recall:.4f}"
)

print(
    f"F1 Score      : {f1:.4f}"
)


print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)



cm = confusion_matrix(
    all_labels,
    all_predictions
)


output_dir = Path(
    "results/confusion_matrix"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title(
    "CIFAR-10 Confusion Matrix"
)

plt.tight_layout()


plt.savefig(
    output_dir /
    "confusion_matrix.png"
)

plt.close()


print(
    "\nConfusion matrix saved to:",
    output_dir / "confusion_matrix.png"
)