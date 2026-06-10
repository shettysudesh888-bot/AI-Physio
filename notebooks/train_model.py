"""
notebooks/train_model.py
------------------------
Binary fracture classifier using DenseNet121 transfer learning.

Usage (from project root):
    python notebooks/train_model.py

Output:
    backend/fracture_model.pth   - best model checkpoint
    backend/class_names.json     - class index to label mapping
"""

import json
import os
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image, ImageFile

# Optional: rich confusion matrix + classification report
try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from backend.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

DATASET_DIR = PROJECT_ROOT / "dataset"
BACKEND_DIR = PROJECT_ROOT / "backend"
SAVE_MODEL = BACKEND_DIR / "fracture_model.pth"
SAVE_CLASSES = BACKEND_DIR / "class_names.json"
SKIPPED_IMAGES_REPORT = PROJECT_ROOT / "dataset_skipped_images.json"

BATCH_SIZE    = int(os.getenv("AI_PHYSIO_BATCH_SIZE", "48"))
EPOCHS        = int(os.getenv("AI_PHYSIO_EPOCHS", "5"))
LR            = 1e-4               # lower LR for fine-tuning pretrained weights
UNFREEZE_ALL  = False              # CPU-friendly mode: fine-tune final DenseNet block
FINE_TUNE_LAST_BLOCK = os.getenv("AI_PHYSIO_FINE_TUNE_LAST_BLOCK", "1") == "1"
MAX_TRAIN_PER_CLASS = int(os.getenv("AI_PHYSIO_MAX_TRAIN_PER_CLASS", "2500"))
RANDOM_SEED = int(os.getenv("AI_PHYSIO_SEED", "42"))
NUM_WORKERS = int(os.getenv("AI_PHYSIO_NUM_WORKERS", "0"))  # safest on Windows CPU
PIN_MEMORY = torch.cuda.is_available()

# ImageNet normalization uses values from backend.config

ImageFile.LOAD_TRUNCATED_IMAGES = False
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_known_bad_paths() -> set[str]:
    """Load cached corrupted image paths to avoid rescanning every image on startup."""
    if os.getenv("AI_PHYSIO_RESCAN_DATASET") == "1" or not SKIPPED_IMAGES_REPORT.exists():
        return set()

    try:
        skipped = json.loads(SKIPPED_IMAGES_REPORT.read_text(encoding="utf-8"))
        paths = {
            str(Path(item["path"]).resolve())
            for item in skipped
            if isinstance(item, dict) and item.get("path")
        }
        if paths:
            print(f"Using cached corrupted-image list: {len(paths)} files")
        return paths
    except Exception as exc:
        print(f"Could not read skipped-image cache ({exc}); rescanning dataset.")
        return set()


def verify_image(path: str) -> tuple[bool, str | None]:
    """Return whether PIL can fully open and load an image."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB").load()
        return True, None
    except Exception as exc:
        return False, repr(exc)


def filter_corrupted_samples(
    dataset: datasets.ImageFolder,
    split_name: str,
    known_bad_paths: set[str],
) -> datasets.ImageFolder:
    """Remove unreadable images from an ImageFolder dataset before DataLoader runs."""
    valid_samples = []
    skipped = []
    should_verify_all = not known_bad_paths

    for path, label in dataset.samples:
        resolved_path = str(Path(path).resolve())
        if resolved_path in known_bad_paths:
            skipped.append({
                "split": split_name,
                "path": resolved_path,
                "error": "cached corrupted/unreadable image",
            })
            continue

        if should_verify_all:
            is_valid, error = verify_image(path)
        else:
            is_valid, error = True, None

        if is_valid:
            valid_samples.append((path, label))
        else:
            skipped.append({
                "split": split_name,
                "path": resolved_path,
                "error": error,
            })

    dataset.samples = valid_samples
    dataset.imgs = valid_samples
    dataset.targets = [label for _, label in valid_samples]

    if skipped:
        print(f"{split_name}: skipped {len(skipped)} corrupted/unreadable images")
    else:
        print(f"{split_name}: no corrupted images found")

    return dataset, skipped


def limit_balanced_training_samples(
    dataset: datasets.ImageFolder,
    max_per_class: int,
) -> datasets.ImageFolder:
    """Keep a deterministic, balanced subset for fast CPU training."""
    samples_by_class: dict[int, list[tuple[str, int]]] = {}
    for path, label in dataset.samples:
        samples_by_class.setdefault(label, []).append((path, label))

    selected = []
    rng = random.Random(RANDOM_SEED)
    for label, samples in sorted(samples_by_class.items()):
        samples = list(samples)
        rng.shuffle(samples)
        selected.extend(samples[:max_per_class])

    rng.shuffle(selected)
    dataset.samples = selected
    dataset.imgs = selected
    dataset.targets = [label for _, label in selected]

    counts = {
        dataset.classes[label]: sum(1 for _, y in selected if y == label)
        for label in sorted(samples_by_class)
    }
    print(f"CPU-fast train subset counts: {counts}")
    return dataset

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Datasets & Loaders
# ---------------------------------------------------------------------------

train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
val_dataset   = datasets.ImageFolder(DATASET_DIR / "val",   transform=eval_transform)
test_dataset  = datasets.ImageFolder(DATASET_DIR / "test",  transform=eval_transform)

known_bad_paths = load_known_bad_paths()

train_dataset, skipped_train = filter_corrupted_samples(train_dataset, "train", known_bad_paths)
val_dataset, skipped_val = filter_corrupted_samples(val_dataset, "val", known_bad_paths)
test_dataset, skipped_test = filter_corrupted_samples(test_dataset, "test", known_bad_paths)

train_dataset = limit_balanced_training_samples(train_dataset, MAX_TRAIN_PER_CLASS)

skipped_images = skipped_train + skipped_val + skipped_test
SKIPPED_IMAGES_REPORT.write_text(json.dumps(skipped_images, indent=2), encoding="utf-8")
print(f"Skipped image report saved to {SKIPPED_IMAGES_REPORT}")

if not train_dataset.samples or not val_dataset.samples or not test_dataset.samples:
    raise RuntimeError("One or more dataset splits are empty after filtering corrupted images.")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# class_names: e.g. {0: "fractured", 1: "not_fractured"}
class_names = train_dataset.classes
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
NUM_CLASSES  = len(class_names)

print(f"Classes: {train_dataset.class_to_idx}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(
    "CPU-fast settings: "
    f"image_size={IMAGE_SIZE}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, "
    f"num_workers={NUM_WORKERS}, pin_memory={PIN_MEMORY}, "
    f"max_train_per_class={MAX_TRAIN_PER_CLASS}, unfreeze_all={UNFREEZE_ALL}"
)

if os.getenv("AI_PHYSIO_SMOKE_TEST") == "1":
    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        images, labels = next(iter(loader))
        print(
            f"{split_name} smoke batch: "
            f"images={tuple(images.shape)} labels={tuple(labels.shape)}"
        )
    print("Smoke test passed. Set AI_PHYSIO_SMOKE_TEST=0 or unset it to train.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Model - DenseNet121 with fine-tuning
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1
    model = models.densenet121(weights=weights)
    print("Loaded ImageNet pretrained DenseNet121 weights.")
except Exception as exc:
    print(f"Could not load pretrained weights ({exc}). Falling back to random initialization.")
    model = models.densenet121(weights=None)

if UNFREEZE_ALL:
    for param in model.parameters():
        param.requires_grad = True
else:
    # Freeze backbone by default; selectively unfreeze final feature block below.
    for param in model.parameters():
        param.requires_grad = False

    if FINE_TUNE_LAST_BLOCK:
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True
        print("Fine-tuning DenseNet denseblock4 + norm5 for better accuracy.")
    else:
        print("Training classifier head only.")

# Replace classifier head
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, NUM_CLASSES),
)

model = model.to(device)

# ---------------------------------------------------------------------------
# Loss, Optimizer, Scheduler
# ---------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
trainable_parameters = [param for param in model.parameters() if param.requires_grad]
optimizer = optim.Adam(trainable_parameters, lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def run_epoch(loader, is_train: bool, epoch: int, total_epochs: int):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    phase = "train" if is_train else "val"
    start_time = time.time()
    log_every = max(1, len(loader) // 5)

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx == 1 or batch_idx == len(loader) or batch_idx % log_every == 0:
                elapsed = time.time() - start_time
                progress = batch_idx / len(loader)
                eta = (elapsed / progress) - elapsed if progress else 0
                running_acc = 100.0 * correct / total
                print(
                    f"Epoch {epoch}/{total_epochs} {phase}: "
                    f"batch {batch_idx}/{len(loader)} "
                    f"({progress * 100:.0f}%) "
                    f"loss={total_loss / total:.4f} "
                    f"acc={running_acc:.2f}% "
                    f"eta={format_seconds(eta)}"
                )

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate_with_report(loader, split_name: str):
    """Run full evaluation with confusion matrix and classification report."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = 100.0 * correct / len(all_labels)
    print(f"\n{split_name} Accuracy: {acc:.2f}%")

    if SKLEARN_AVAILABLE:
        print(f"\n{split_name} Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        cm = confusion_matrix(all_labels, all_preds)
        print(f"{split_name} Confusion Matrix:")
        print(cm)
    else:
        print("(Install scikit-learn for a full classification report: pip install scikit-learn)")

    return acc

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

best_val_loss = float("inf")
best_epoch    = -1
patience      = 5
epochs_no_improve = 0

print("\n" + "="*60)
print("Starting training")
print("="*60)
training_start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train_loss, train_acc = run_epoch(train_loader, is_train=True, epoch=epoch, total_epochs=EPOCHS)
    val_loss,   val_acc   = run_epoch(val_loader,   is_train=False, epoch=epoch, total_epochs=EPOCHS)

    scheduler.step(val_loss)
    elapsed_total = time.time() - training_start_time
    epoch_time = time.time() - epoch_start_time
    estimated_remaining = epoch_time * (EPOCHS - epoch)

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}]  "
        f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
        f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  |  "
        f"Epoch Time: {format_seconds(epoch_time)}  "
        f"Total: {format_seconds(elapsed_total)}  "
        f"Remaining: {format_seconds(estimated_remaining)}"
    )

    # Save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
        }, SAVE_MODEL)
        print(f"  Best model saved (epoch {epoch}, val_loss={val_loss:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"  Early stopping counter: {epochs_no_improve} out of {patience}")
        if epochs_no_improve >= patience:
            print("  Early stopping triggered!")
            break

print(f"\nTraining complete. Best epoch: {best_epoch} | Best val loss: {best_val_loss:.4f}")

# ---------------------------------------------------------------------------
# Save class mapping
# ---------------------------------------------------------------------------

with open(SAVE_CLASSES, "w") as f:
    json.dump(idx_to_class, f, indent=2)
print(f"Class mapping saved to {SAVE_CLASSES}")

# ---------------------------------------------------------------------------
# Test set evaluation (loads best checkpoint)
# ---------------------------------------------------------------------------

print("\nLoading best checkpoint for test evaluation...")
checkpoint = torch.load(SAVE_MODEL, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

evaluate_with_report(val_loader,  "Validation")
evaluate_with_report(test_loader, "Test")

print("\n" + "="*60)
print("DISCLAIMER: This model is for educational purposes only.")
print("It is NOT a medical diagnostic tool.")
print("="*60)
