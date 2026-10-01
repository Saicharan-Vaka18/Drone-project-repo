import os
from pathlib import Path

from dotenv import load_dotenv
from ultralytics import YOLO


# ============================================================
# AeroTrash Detect - YOLO Model Manager
# ============================================================

# Backend directory
BASE = Path(__file__).resolve().parent


# ============================================================
# Load .env file
# ============================================================

load_dotenv(BASE / ".env")


# ============================================================
# Configuration
# ============================================================

# Get model path from .env
#
# If YOLO_MODEL_PATH is not present, use:
# backend/models/my_model.pt
#
MODEL_PATH = Path(
    os.getenv(
        "YOLO_MODEL_PATH",
        str(BASE / "models" / "my_model.pt")
    )
).expanduser()


# ------------------------------------------------------------
# Confidence threshold
# ------------------------------------------------------------
#
# Your testing showed:
#
#   0.10 -> 9 detections
#   0.25 -> 2 detections
#   0.55 -> 0 detections
#
# Therefore we use 0.25 as the default.
#
MODEL_CONFIDENCE = float(
    os.getenv("YOLO_CONFIDENCE", "0.25")
)


# ============================================================
# Global model variable
# ============================================================

MODEL = None


# ============================================================
# Load YOLO model
# ============================================================

def load_model():
    """
    Load the YOLO model only once.

    Returns:
        YOLO: Loaded Ultralytics YOLO model.
    """

    global MODEL

    # Already loaded
    if MODEL is not None:
        return MODEL

    # Check whether model file exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found at:\n{MODEL_PATH}\n\n"
            "Check YOLO_MODEL_PATH in backend/.env"
        )

    print("=" * 60)
    print("AeroTrash Detect - Loading YOLO Model")
    print("=" * 60)
    print(f"Model path : {MODEL_PATH}")
    print(f"Confidence : {MODEL_CONFIDENCE}")

    try:
        MODEL = YOLO(str(MODEL_PATH))

        print("MODEL LOADED SUCCESSFULLY")
        print(f"Model classes: {MODEL.names}")
        print("=" * 60)

        return MODEL

    except Exception as exc:
        MODEL = None

        print("=" * 60)
        print("ERROR LOADING YOLO MODEL")
        print(str(exc))
        print("=" * 60)

        raise RuntimeError(
            f"Could not load YOLO model: {exc}"
        ) from exc


# ============================================================
# Get model
# ============================================================

def get_model():
    """
    Return the loaded YOLO model.

    If the model hasn't been loaded yet,
    load it automatically.
    """

    return load_model()


# ============================================================
# Model status
# ============================================================

def model_status():
    """
    Return information about the YOLO model.
    """

    try:
        model = load_model()

        return {
            "name": MODEL_PATH.name,
            "path": str(MODEL_PATH),
            "exists": MODEL_PATH.exists(),
            "loaded": model is not None,
            "confidence": MODEL_CONFIDENCE,
            "classes": model.names,
        }

    except Exception as exc:

        return {
            "name": MODEL_PATH.name,
            "path": str(MODEL_PATH),
            "exists": MODEL_PATH.exists(),
            "loaded": False,
            "confidence": MODEL_CONFIDENCE,
            "error": str(exc),
        }


# ============================================================
# YOLO prediction
# ============================================================

def predict(frame, confidence=None):
    """
    Run YOLO prediction on an OpenCV image/frame.

    Args:
        frame:
            OpenCV image as a NumPy array.

        confidence:
            Optional confidence threshold.
            If not supplied, MODEL_CONFIDENCE is used.

    Returns:
        Ultralytics prediction results.
    """

    model = load_model()

    # Use configured confidence if no custom value is supplied
    if confidence is None:
        confidence = MODEL_CONFIDENCE

    # Make sure confidence is a valid number
    confidence = float(confidence)

    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(
            f"Confidence must be between 0.0 and 1.0. "
            f"Received: {confidence}"
        )

    results = model.predict(
        frame,
        conf=confidence,
        verbose=False
    )

    return results


# ============================================================
# Extract detections
# ============================================================

def get_detections(frame, confidence=None):
    """
    Run YOLO and convert detections into
    a simple Python list.

    Returns:
        List of dictionaries containing:

        - label
        - confidence
        - class_id
        - box
    """

    results = predict(
        frame,
        confidence=confidence
    )

    detections = []

    for result in results:

        boxes = result.boxes

        # No detections
        if boxes is None:
            continue

        names = result.names

        for box in boxes:

            # Detection confidence
            conf = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            # Class name
            label = str(
                names.get(class_id, class_id)
            )

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "class_id": class_id,
                    "box": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                }
            )

    return detections


# ============================================================
# Simple test when running model.py directly
# ============================================================

if __name__ == "__main__":

    print()
    print("AeroTrash Detect YOLO Model Test")
    print("-" * 40)

    print(f"Model path : {MODEL_PATH}")
    print(f"Exists     : {MODEL_PATH.exists()}")
    print(f"Confidence : {MODEL_CONFIDENCE}")

    try:

        model = load_model()

        print()
        print("STATUS: MODEL READY")
        print(f"Classes: {model.names}")

    except Exception as exc:

        print()
        print("STATUS: MODEL FAILED")
        print(f"Error: {exc}")