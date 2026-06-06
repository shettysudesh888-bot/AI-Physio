"""
backend/predict.py
------------------
Image preprocessing and AI model inference.

Phase 1 (mock):  Works out of the box - no model file needed.
Phase 2 (real):  Run notebooks/train_model.py first to produce
                 backend/fracture_model.pth and backend/class_names.json,
                 then the real model is loaded automatically.

DISCLAIMER: For educational purposes only. Not a medical diagnostic tool.
"""

import json
import os
import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR     = Path(__file__).resolve().parent
MODEL_PATH   = BASE_DIR / "fracture_model.pth"
CLASSES_PATH = BASE_DIR / "class_names.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE = (160, 160)
MIN_CONFIDENCE = 0.70
MIN_PROBABILITY_MARGIN = 0.15

# ImageNet normalization (must match training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Map model output labels to recommendation condition keys.
# Keys must match EXERCISE_MAP in recommendation.py
LABEL_TO_CONDITION = {
    "fractured":     "fracture",
    "not_fractured": "normal",
    # Extend here when you add more classes
}


def _load_grayscale(image_path: str) -> np.ndarray:
    """Load an image as grayscale for quality checks and preprocessing."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        pil_img = Image.open(image_path).convert("L")
        img = np.array(pil_img)
    return img


def _image_quality(image_path: str) -> dict:
    """
    Compute lightweight quality signals before inference.

    These checks do not diagnose whether the image is an X-ray. They only flag
    common input problems that can make model output less reliable.
    """
    img = _load_grayscale(image_path)
    height, width = img.shape[:2]
    brightness = float(img.mean())
    contrast = float(img.std())
    blur_score = float(cv2.Laplacian(img, cv2.CV_64F).var())

    warnings = []
    if min(width, height) < 180:
        warnings.append("Image resolution is low; small fracture details may be missed.")
    if brightness < 25:
        warnings.append("Image appears very dark.")
    elif brightness > 235:
        warnings.append("Image appears very bright.")
    if contrast < 18:
        warnings.append("Image contrast is low.")
    if blur_score < 35:
        warnings.append("Image may be blurry or poorly focused.")

    return {
        "status": "review" if warnings else "ok",
        "warnings": warnings,
        "metrics": {
            "width": int(width),
            "height": int(height),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "blur_score": round(blur_score, 2),
        },
        "note": "Quality checks are screening signals only and do not confirm image type or diagnosis.",
    }


def _prediction_decision(confidence: float, class_scores: dict, quality: dict) -> dict:
    scores = sorted((float(score) for score in class_scores.values()), reverse=True)
    margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]

    reasons = []
    if confidence < MIN_CONFIDENCE:
        reasons.append(f"Confidence is below {int(MIN_CONFIDENCE * 100)}%.")
    if margin < MIN_PROBABILITY_MARGIN:
        reasons.append("Top class scores are too close together.")
    if quality.get("status") == "review":
        reasons.extend(quality.get("warnings") or [])

    uncertain = bool(reasons)
    return {
        "status": "uncertain" if uncertain else "confident",
        "is_uncertain": uncertain,
        "probability_margin": round(float(margin), 4),
        "reasons": reasons,
        "recommended_action": (
            "Clinical review recommended before relying on this result."
            if uncertain
            else "Result passed basic confidence and image-quality checks."
        ),
    }

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

_model      = None
_idx_to_cls = None  # {0: "fractured", 1: "not_fractured"}


def _load_model():
    """
    Load DenseNet121 from checkpoint.
    Returns (model, idx_to_cls) or (None, None) if checkpoint is missing.
    """
    global _model, _idx_to_cls

    if _model is not None:
        return _model, _idx_to_cls

    if not MODEL_PATH.exists():
        return None, None

    import torch
    import torch.nn as nn
    from torchvision import models

    # Load class mapping
    if CLASSES_PATH.exists():
        with open(CLASSES_PATH) as f:
            raw = json.load(f)
        # json keys are strings; convert to int
        idx_to_cls = {int(k): v for k, v in raw.items()}
    else:
        # Fallback: assume alphabetical order (ImageFolder default)
        idx_to_cls = {0: "fractured", 1: "not_fractured"}

    num_classes = len(idx_to_cls)

    # Rebuild model architecture (must match training)
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model = model.to(device)

    _model      = model
    _idx_to_cls = idx_to_cls

    print(f"[predict] Model loaded from {MODEL_PATH} | Classes: {idx_to_cls}")
    return _model, _idx_to_cls


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image_path: str) -> "np.ndarray":
    """
    Load and preprocess an X-ray image for DenseNet121.

    Pipeline:
    - Load as grayscale (X-rays are single-channel)
    - Resize to 224x224
    - CLAHE contrast enhancement
    - Convert to 3-channel RGB (DenseNet expects 3 channels)
    - Normalize with ImageNet mean/std
    - Shape: (1, 3, H, W), batch of 1
    """
    img = _load_grayscale(image_path)

    # Resize
    img = cv2.resize(img, IMAGE_SIZE)

    # CLAHE improves local contrast typical in X-rays.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    # Convert to float32 in [0, 1]
    img = img.astype(np.float32) / 255.0

    # Stack to 3 channels (H, W) -> (H, W, 3)
    img_rgb = np.stack([img, img, img], axis=-1)

    # Apply ImageNet normalization: (value - mean) / std per channel
    img_rgb = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD

    # (H, W, 3) -> (1, 3, H, W)
    img_rgb = img_rgb.transpose(2, 0, 1)[np.newaxis, :]

    return img_rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# Mock predictor (Phase 1 fallback)
# ---------------------------------------------------------------------------

_MOCK_CLASSES = ["fractured", "not_fractured"]


def _mock_predict(image_array: np.ndarray) -> dict:
    """
    Deterministic mock used when no trained model is available.
    Seed is derived from image statistics so results are reproducible.
    """
    seed  = int(image_array.mean() * 1000) % len(_MOCK_CLASSES)
    label = _MOCK_CLASSES[seed]

    rng    = np.random.default_rng(seed)
    conf   = round(float(rng.uniform(0.60, 0.90)), 4)
    scores = rng.dirichlet(np.ones(len(_MOCK_CLASSES))).tolist()

    class_scores = {c: round(float(s), 4) for c, s in zip(_MOCK_CLASSES, scores)}
    condition    = LABEL_TO_CONDITION.get(label, "normal")

    return {
        "label":        label,
        "condition":    condition,
        "confidence":   conf,
        "class_scores": class_scores,
        "model_mode":   "mock",
        "disclaimer":   "MOCK prediction - run train_model.py to enable real inference.",
    }


# ---------------------------------------------------------------------------
# Real predictor (Phase 2)
# ---------------------------------------------------------------------------

def _real_predict(model, idx_to_cls: dict, image_array: np.ndarray) -> dict:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    tensor = torch.from_numpy(image_array).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().tolist()

    if isinstance(probs, float):
        probs = [probs]

    best_idx  = int(np.argmax(probs))
    label     = idx_to_cls.get(best_idx, "unknown")
    condition = LABEL_TO_CONDITION.get(label, "normal")

    class_scores = {
        idx_to_cls.get(i, str(i)): round(float(p), 4)
        for i, p in enumerate(probs)
    }

    return {
        "label":        label,
        "condition":    condition,
        "confidence":   round(float(probs[best_idx]), 4),
        "class_scores": class_scores,
        "model_mode":   "densenet121",
        "disclaimer":   "Educational use only. Not a medical diagnostic tool.",
        "explainability": {
            "method": "Grad-CAM",
            "target_class": label,
        },
        "_target_idx": best_idx,
    }


def _gradcam_overlay(image_path: str, model, target_idx: int) -> str | None:
    """
    Build a Grad-CAM overlay as a PNG data URL.

    This is an educational visual explanation: it highlights image regions that
    most influenced the selected class score. It is not clinical localization.
    """
    import torch
    import torch.nn.functional as F

    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations, gradients
        activations = output

        def save_gradient(grad):
            nonlocal gradients
            gradients = grad.detach()

        output.register_hook(save_gradient)

    target_layer = model.features.norm5
    forward_handle = target_layer.register_forward_hook(forward_hook)

    try:
        image_array = _preprocess(image_path)
        device = next(model.parameters()).device
        tensor = torch.from_numpy(image_array).to(device)

        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        score = logits[:, target_idx].sum()
        score.backward()

        if activations is None or gradients is None:
            return None

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations.detach()).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max <= cam_min:
            return None

        cam = (cam - cam_min) / (cam_max - cam_min)

        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            original = np.array(Image.open(image_path).convert("L"))

        height, width = original.shape[:2]
        heatmap = cv2.resize(cam, (width, height))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original_rgb, 0.62, heatmap, 0.38, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        buffer = BytesIO()
        Image.fromarray(overlay_rgb).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    finally:
        forward_handle.remove()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_condition(image_path: str) -> dict:
    """
    Main entry point for prediction.

    Returns a dict with:
        label        -- raw model class ("fractured" / "not_fractured")
        condition    -- mapped key for recommendation engine ("fracture" / "normal")
        confidence   -- probability of predicted class
        class_scores -- per-class probabilities
        model_mode   -- "densenet121" or "mock"
        disclaimer   -- educational use notice
        image_path   -- path of analysed image
    """
    try:
        quality = _image_quality(image_path)
        image_array = _preprocess(image_path)
        model, idx_to_cls = _load_model()

        if model is None:
            result = _mock_predict(image_array)
        else:
            result = _real_predict(model, idx_to_cls, image_array)
            target_idx = result.pop("_target_idx", None)
            if target_idx is not None:
                try:
                    overlay = _gradcam_overlay(image_path, model, target_idx)
                    if overlay:
                        result["explainability"]["overlay_image"] = overlay
                        result["explainability"]["note"] = (
                            "Educational heatmap showing regions that influenced the model score. "
                            "Not a clinical localization result."
                        )
                except Exception as exc:
                    result["explainability"]["error"] = f"Heatmap unavailable: {exc}"

        result["image_quality"] = quality
        result["decision"] = _prediction_decision(
            float(result.get("confidence") or 0),
            result.get("class_scores") or {},
            quality,
        )
        result["image_path"] = image_path
        return result

    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
