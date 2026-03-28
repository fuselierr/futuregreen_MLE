"""
single_image.py
---------------
Process a single image through the full pipeline:
  1. Check input resolution  (MIN_INPUT_RESOLUTION)
  2. Run YOLO – require exactly 1 valid detection
  3. Crop to bounding box and check crop resolution  (MIN_CROP_RESOLUTION)

Usage:
  python single_image.py <image_path> [output_path]

  - If the image passes all 3 checks the cropped image is saved.
      - output_path defaults to  <stem>_cropped.jpg  next to the source.
  - On any failure the script prints a descriptive error and exits with code 1.
"""

import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
YOLO_MODEL = SCRIPT_DIR / "yolov8n-waste-12cls-best.pt"

IOU      = 0.50
AGNOSTIC = False

FILTER_CLASSES = ["paper", "cardboard", "metal", "plastic", "organic", "trash", "glass"]

CLASS_THRESHOLDS = {
    "cardboard":   0.25,
    "metal":       0.32,
    "paper":       0.25,
    "plastic":     0.30,
    "biological":  0.35,
    "trash":       0.20,
    "battery":     0.20,
    "clothes":     0.20,
    "shoes":       0.20,
    "brown-glass": 0.25,
    "green-glass": 0.25,
    "white-glass": 0.25,
}

CROP_PADDING = 10

# Resolution thresholds
MIN_INPUT_RESOLUTION = (250, 250)   # (min_width, min_height) for the raw input image
MIN_CROP_RESOLUTION  = (112, 112)   # (min_width, min_height) for the cropped result
# ──────────────────────────────────────────────────────────────────────────────


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fail(message: str) -> None:
    """Print an error message and exit with a non-zero status."""
    print(f"[FAIL] {message}", file=sys.stderr)
    sys.exit(1)


def convert_to_jpg(img_path: Path) -> Path:
    """
    Convert the image at img_path to JPEG (RGB, quality 95).
    If the file is already a JPEG, returns the original path unchanged.
    Otherwise saves a converted copy as <stem>.jpg beside the source and
    returns that new path.
    """
    if img_path.suffix.lower() in (".jpg", ".jpeg"):
        return img_path

    out_path = img_path.with_suffix(".jpg")
    img = Image.open(img_path).convert("RGB")
    img.save(str(out_path), format="JPEG", quality=95)
    print(f"[INFO] Converted {img_path.name} → {out_path.name}")
    return out_path


def check_input_resolution(img_path: Path) -> tuple[int, int]:
    """
    Open the image and verify it meets MIN_INPUT_RESOLUTION.
    Returns (width, height) on success, calls fail() on failure.
    """
    img = Image.open(img_path)
    w, h = img.size
    min_w, min_h = MIN_INPUT_RESOLUTION
    if w < min_w or h < min_h:
        fail(
            f"Input resolution {w}x{h} is below the minimum "
            f"{min_w}x{min_h}."
        )
    return w, h


def run_yolo(img_path: Path, model: YOLO):
    """
    Run YOLO on img_path.
    Returns the single valid bounding-box xyxy list on success,
    calls fail() if the detection count is not exactly 1.
    """
    results = model(str(img_path), iou=IOU, agnostic_nms=AGNOSTIC, verbose=False)
    result  = results[0]

    allowed_classes = {c.lower() for c in FILTER_CLASSES}
    matched_boxes   = []

    if result.boxes is not None:
        for cls_id, conf, xyxy in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.xyxy.tolist(),
        ):
            cls_name  = model.names.get(int(cls_id), str(cls_id)).lower()
            threshold = CLASS_THRESHOLDS.get(cls_name, 0.25)
            if cls_name in allowed_classes and conf >= threshold:
                matched_boxes.append(xyxy)

    count = len(matched_boxes)
    if count != 1:
        fail(
            f"YOLO detected {count} valid object(s) "
            f"(expected exactly 1)."
        )

    return matched_boxes[0]


def crop_image(img_path: Path, xyxy, padding: int) -> Image.Image:
    """
    Crop img_path to the bounding box (with padding clamped to image bounds).
    Returns the cropped PIL Image.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return img.crop((x1, y1, x2, y2))


def check_bbox_resolution(xyxy) -> None:
    """
    Verify the raw bounding box (before padding or cropping) meets
    MIN_CROP_RESOLUTION.  Calls fail() on failure.
    """
    x1, y1, x2, y2 = map(int, xyxy)
    bw, bh = x2 - x1, y2 - y1
    min_w, min_h = MIN_CROP_RESOLUTION
    if bw < min_w or bh < min_h:
        fail(
            f"Bounding-box size {bw}x{bh} is below the minimum "
            f"{min_w}x{min_h} (checked before padding)."
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    img_path = Path(sys.argv[1])

    if not img_path.is_file():
        fail(f"File not found: {img_path}")

    # ── Step 0: Convert to JPEG ───────────────────────────────────────────────
    img_path = convert_to_jpg(img_path)

    # Determine output path
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    else:
        out_path = img_path.with_name(img_path.stem + "_cropped.jpg")

    # ── Step 1: Check input resolution ────────────────────────────────────────
    iw, ih = check_input_resolution(img_path)
    print(f"[OK]   Input resolution: {iw}x{ih}")

    # ── Step 2: Run YOLO ──────────────────────────────────────────────────────
    print(f"[INFO] Loading YOLO model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)

    xyxy = run_yolo(img_path, model)
    print(f"[OK]   YOLO: exactly 1 valid detection found.")

    # ── Step 3: Check raw bbox resolution, then crop ──────────────────────────
    check_bbox_resolution(xyxy)
    x1, y1, x2, y2 = map(int, xyxy)
    print(f"[OK]   Bounding-box size (raw): {x2 - x1}x{y2 - y1}")
    cropped = crop_image(img_path, xyxy, CROP_PADDING)
    cw, ch = cropped.size
    print(f"[OK]   Crop size (with padding): {cw}x{ch}")

    # ── All checks passed: save the cropped image ──────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(str(out_path))
    print(f"[DONE] Cropped image saved to: {out_path}")


if __name__ == "__main__":
    main()
