import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from resnet18 import ResNet18


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 2. CIFAR-10 classes
# ============================================================

classes = [
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


# ============================================================
# 3. Test transform
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# 4. Load CIFAR-10 test dataset
# ============================================================

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# 5. Load model
# ============================================================

model = ResNet18(num_classes=10)

model.load_state_dict(
    torch.load(
        "models/best_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# 6. Prediction function
# ============================================================

def predict_image(index):

    image, true_label = test_dataset[index]

    # Add batch dimension
    image_batch = image.unsqueeze(0).to(device)

    # Disable gradient calculation
    with torch.no_grad():

        outputs = model(image_batch)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted_label = torch.max(
            probabilities, dim=1
        )

    predicted_label = predicted_label.item()
    confidence = confidence.item()

    return (
        image,
        true_label,
        predicted_label,
        confidence
    )


# ============================================================
# 7. Display predictions
# ============================================================

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flat):

    image, true_label, predicted_label, confidence = predict_image(i)

    # Convert normalized image back for displaying
    image_display = image.permute(1, 2, 0)

    mean = torch.tensor(
        [0.4914, 0.4822, 0.4465]
    )

    std = torch.tensor(
        [0.2470, 0.2435, 0.2616]
    )

    image_display = image_display * std + mean

    image_display = torch.clamp(
        image_display,
        0,
        1
    )

    ax.imshow(image_display)

    ax.set_title(
        f"Actual: {classes[true_label]}\n"
        f"Pred: {classes[predicted_label]}\n"
        f"Confidence: {confidence:.2%}"
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "results/predictions/sample_predictions.png",
    dpi=200
)

plt.show()

print(
    "\nSaved predictions to:"
    " results/predictions/sample_predictions.png"
)