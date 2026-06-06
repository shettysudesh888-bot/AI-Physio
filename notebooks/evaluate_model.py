"""
Evaluate the trained AI Physio fracture classifier.

Run from project root:
    python notebooks/evaluate_model.py

Outputs:
    evaluation_metrics.json
    confusion_matrix_validation.png
    confusion_matrix_test.png
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "backend" / "fracture_model.pth"
CLASSES_PATH = PROJECT_ROOT / "backend" / "class_names.json"
SKIPPED_IMAGES_REPORT = PROJECT_ROOT / "dataset_skipped_images.json"
METRICS_PATH = PROJECT_ROOT / "evaluation_metrics.json"

IMAGE_SIZE = 160
BATCH_SIZE = 64
NUM_WORKERS = 0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ImageFile.LOAD_TRUNCATED_IMAGES = False


def load_class_names() -> list[str]:
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Class mapping not found: {CLASSES_PATH}")

    raw = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return [raw[str(index)] for index in range(len(raw))]


def load_known_bad_paths() -> set[str]:
    if not SKIPPED_IMAGES_REPORT.exists():
        return set()

    skipped = json.loads(SKIPPED_IMAGES_REPORT.read_text(encoding="utf-8"))
    return {
        str(Path(item["path"]).resolve())
        for item in skipped
        if isinstance(item, dict) and item.get("path")
    }


def filter_bad_samples(dataset: datasets.ImageFolder, known_bad_paths: set[str]):
    dataset.samples = [
        (path, label)
        for path, label in dataset.samples
        if str(Path(path).resolve()) not in known_bad_paths
    ]
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]
    return dataset


def make_loader(split: str) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(DATASET_DIR / split, transform=transform)
    dataset = filter_bad_samples(dataset, load_known_bad_paths())
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )


def build_model(num_classes: int):
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


def evaluate_split(model, loader: DataLoader, class_names: list[str], split_name: str):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs[:, 1].tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "r2_score_note": "R2 is a regression metric and is not appropriate for binary classification.",
        "classification_report": classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "support": len(all_labels),
    }

    display = ConfusionMatrixDisplay.from_predictions(
        all_labels,
        all_preds,
        display_labels=class_names,
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title(f"{split_name.title()} Confusion Matrix")
    plt.tight_layout()
    output_path = PROJECT_ROOT / f"confusion_matrix_{split_name}.png"
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"\n{split_name.title()} Metrics")
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision_weighted'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall_weighted'] * 100:.2f}%")
    print(f"F1-score:  {metrics['f1_weighted'] * 100:.2f}%")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Confusion matrix saved to {output_path}")

    return metrics


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    class_names = load_class_names()
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    results = {
        "model_path": str(MODEL_PATH),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "validation": evaluate_split(model, make_loader("val"), class_names, "validation"),
        "test": evaluate_split(model, make_loader("test"), class_names, "test"),
    }

    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nMetrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
