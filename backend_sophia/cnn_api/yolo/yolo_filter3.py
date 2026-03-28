import os
import re
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────
SOURCE_DIR = Path(r"data\processed\consolidated_raws")
TARGET_DIR = Path(r"data\processed\dataset_vers3.2")

SCRIPT_DIR = Path(__file__).parent
YOLO_MODEL = SCRIPT_DIR / "yolov8n-waste-12cls-best.pt"
IOU        = 0.50
AGNOSTIC   = False

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

IMAGE_EXTS   = {".jpg"}
CROP_PADDING = 10

CLEAR_TARGET_BEFORE_RUN = False

# Resolution filter – if True, crops smaller than MIN_RESOLUTION are rejected.
# Set FILTER_MIN_RESOLUTION = False to disable this check entirely.
FILTER_MIN_RESOLUTION = True
MIN_RESOLUTION        = (50, 50)   # (min_width, min_height) in pixels
# ──────────────────────────────────────────────────────────────────────────────


def scan_existing_counts(target_path: Path) -> dict[str, int]:
    """
    If a previous run partially populated the target directory, find the
    highest existing index for each class so we don't overwrite anything.
    Returns {classname: highest_index_seen}.
    """
    counts: dict[str, int] = {}
    if not target_path.exists():
        return counts

    for class_dir in target_path.iterdir():
        if not class_dir.is_dir() or class_dir.name == "rejected":
            continue
        class_name = class_dir.name
        max_num = 0
        pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.jpg$", re.IGNORECASE)
        for f in class_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                max_num = max(max_num, int(m.group(1)))
        counts[class_name] = max_num
    return counts


def count_target_objects(result, model_names: dict, test_classes: list, class_thresholds: dict) -> int:
    """Count bounding boxes that match a target class and pass its confidence threshold."""
    if result.boxes is None or len(result.boxes) == 0:
        return 0

    count = 0
    allowed_classes = {c.lower() for c in test_classes}

    for cls_id, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        cls_name = model_names.get(int(cls_id), str(cls_id)).lower()
        if cls_name in allowed_classes:
            threshold = class_thresholds.get(cls_name, 0.25)
            if conf >= threshold:
                count += 1

    return count


def get_single_target_box(result, model_names: dict, test_classes: list, class_thresholds: dict):
    """Return the xyxy of the sole valid detection, or None if != 1 valid box."""
    if result.boxes is None or len(result.boxes) == 0:
        return None

    allowed_classes = {c.lower() for c in test_classes}
    matched_boxes = []

    for cls_id, conf, xyxy in zip(
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.xyxy.tolist(),
    ):
        cls_name = model_names.get(int(cls_id), str(cls_id)).lower()
        if cls_name in allowed_classes:
            threshold = class_thresholds.get(cls_name, 0.25)
            if conf >= threshold:
                matched_boxes.append(xyxy)

    return matched_boxes[0] if len(matched_boxes) == 1 else None


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def check_crop_resolution(img_path: Path, xyxy, padding: int, min_res: tuple):
    """
    Compute the bounding-box crop size (with padding) and compare against min_res.
    Returns (passes: bool, crop_w: int, crop_h: int).
    """
    img = Image.open(img_path)
    w, h = img.size
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    crop_w = x2 - x1
    crop_h = y2 - y1
    passes = crop_w >= min_res[0] and crop_h >= min_res[1]
    return passes, crop_w, crop_h


def crop_and_save_image(img_path: Path, xyxy, save_path: Path, padding: int = 0):
    """Crop img_path to xyxy (with padding clamped to image bounds) and save."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    img.crop((x1, y1, x2, y2)).save(save_path)


def main():
    cwd = Path(os.getcwd())
    source_path  = cwd / SOURCE_DIR
    target_path  = cwd / TARGET_DIR
    rejected_path = target_path / "rejected"

    if not source_path.exists():
        print(f"[ERROR] Source directory not found: {source_path}")
        return

    # ── Prepare output ────────────────────────────────────────────────────────
    if CLEAR_TARGET_BEFORE_RUN and target_path.exists():
        shutil.rmtree(target_path)

    target_path.mkdir(parents=True, exist_ok=True)
    rejected_path.mkdir(parents=True, exist_ok=True)

    # Pre-scan to resume without collision
    existing = scan_existing_counts(target_path)

    accepted_counters: dict[str, int] = {}

    # Highest existing rejected index
    rej_pattern = re.compile(r"^rejected_(\d+)\.jpg$", re.IGNORECASE)
    max_rej = 0
    for f in rejected_path.iterdir():
        m = rej_pattern.match(f.name)
        if m:
            max_rej = max(max_rej, int(m.group(1)))
    rejected_counter = max_rej + 1

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading YOLO model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)
    print("Model task:", getattr(model, "task", "unknown"))
    print("Model names:", model.names)
    print(f"[INFO] Filter classes: {FILTER_CLASSES}")

    # ── Collect class subdirectories ──────────────────────────────────────────
    class_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    if not class_dirs:
        print("[ERROR] No class subdirectories found in source directory.")
        return

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = {
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "rejected_multi": 0,
        "rejected_size": 0,
        "errors": 0
    }
    class_stats: dict[str, dict] = {}

    # ── Process each class ────────────────────────────────────────────────────
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = sorted([p for p in class_dir.iterdir() if is_image_file(p)])

        if not images:
            print(f"[SKIP] {class_name}: no .jpg images found.")
            continue

        # Init counter, resuming from any previously accepted images
        if class_name not in accepted_counters:
            accepted_counters[class_name] = existing.get(class_name, 0) + 1

        class_stats[class_name] = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "rejected_multi": 0,
            "rejected_size": 0,
            "errors": 0
        }
        class_out_dir = target_path / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n── Processing class: {class_name} ({len(images)} images) ──")

        for img_path in images:
            stats["total"]                      += 1
            class_stats[class_name]["total"]    += 1

            try:
                results = model(
                    str(img_path),
                    iou=IOU,
                    agnostic_nms=AGNOSTIC,
                    verbose=False,
                )
                result = results[0]

                object_count = count_target_objects(
                    result, model.names, FILTER_CLASSES, CLASS_THRESHOLDS
                )

                if object_count != 1:
                    # ── REJECT ────────────────────────────────────────────────
                    new_name = f"rejected_{rejected_counter}.jpg"
                    dest     = rejected_path / new_name
                    shutil.copy2(str(img_path), str(dest))
                    print(f"  [REJECT] {img_path.name} → {new_name}  (objects={object_count})")
                    rejected_counter                          += 1
                    stats["rejected"]                         += 1
                    stats["rejected_multi"]                   += 1
                    class_stats[class_name]["rejected"]       += 1
                    class_stats[class_name]["rejected_multi"] += 1

                else:
                    # ── ACCEPT: crop to bounding box, save in class folder ─────
                    single_box = get_single_target_box(
                        result, model.names, FILTER_CLASSES, CLASS_THRESHOLDS
                    )

                    if single_box is None:
                        # Parsed box disagrees with count — treat as rejected
                        new_name = f"rejected_{rejected_counter}.jpg"
                        dest     = rejected_path / new_name
                        shutil.copy2(str(img_path), str(dest))
                        print(f"  [REJECT] {img_path.name} → {new_name}  (count=1 but box parse failed)")
                        rejected_counter                    += 1
                        stats["rejected"]                   += 1
                        class_stats[class_name]["rejected"] += 1
                    else:
                        # ── Optional resolution filter ─────────────────────────
                        if FILTER_MIN_RESOLUTION:
                            res_ok, crop_w, crop_h = check_crop_resolution(
                                img_path, single_box, CROP_PADDING, MIN_RESOLUTION
                            )
                            if not res_ok:
                                new_name = f"rejected_{rejected_counter}.jpg"
                                dest     = rejected_path / new_name
                                shutil.copy2(str(img_path), str(dest))
                                print(
                                    f"  [REJECT] {img_path.name} → {new_name}"
                                    f"  (crop {crop_w}x{crop_h} < {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]})"
                                )
                                rejected_counter                         += 1
                                stats["rejected"]                        += 1
                                stats["rejected_size"]                   += 1
                                class_stats[class_name]["rejected"]      += 1
                                class_stats[class_name]["rejected_size"] += 1
                                continue

                        new_name = f"{class_name}_{accepted_counters[class_name]}.jpg"
                        dest     = class_out_dir / new_name
                        crop_and_save_image(
                            img_path=img_path,
                            xyxy=single_box,
                            save_path=dest,
                            padding=CROP_PADDING,
                        )
                        print(f"  [ACCEPT] {img_path.name} → {new_name}  (cropped)")
                        accepted_counters[class_name]           += 1
                        stats["accepted"]                       += 1
                        class_stats[class_name]["accepted"]     += 1

            except Exception as exc:
                print(f"  [ERROR]  {img_path.name}: {exc}")
                stats["errors"]                     += 1
                class_stats[class_name]["errors"]   += 1

    # ── Final Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"{'Total processed':<25}: {stats['total']}")
    print(f"{'Total accepted':<25}: {stats['accepted']}")
    print(f"{'Total rejected':<25}: {stats['rejected']}")
    print(f"{'  ├─ Multi-object':<25}: {stats['rejected_multi']}")
    print(f"{'  └─ Size/Resolution':<25}: {stats['rejected_size']}")
    print(f"{'Total errors':<25}: {stats['errors']}")
    print()
    print(f"{'Class':<15} {'Total':<8} {'Accepted':<10} {'Rejected':<10} {'Rej(Multi)':<12} {'Rej(Size)':<12} {'Errors':<8}")
    print("-" * 80)
    for cls, s in class_stats.items():
        print(f"{cls:<15} {s['total']:<8} {s['accepted']:<10} {s['rejected']:<10} {s['rejected_multi']:<12} {s['rejected_size']:<12} {s['errors']:<8}")
    print("=" * 80)


if __name__ == "__main__":
    main()