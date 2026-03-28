from pathlib import Path
from ultralytics import YOLO
from PIL import Image

SOURCE_DIR = Path(r"rejected/multiple/cardboard")
CROP_TARGET_DIR = Path(r"cropped/cpdCardboard")

YOLO_MODEL = "yolov8n-waste-12cls-best.pt"
IOU = 0.35
AGNOSTIC    = False
DEVICE = "cpu"

MIN_OBJECTS = 2
MAX_OBJECTS = 8
MIN_CROP_SIZE = 50
CROP_PADDING = 5
IMAGE_EXTS = {".jpg"}

FILTER_CLASSES = ["cardboard"]
CLASS_THRESHOLDS = {
    "cardboard": 0.25,
    "metal": 0.32,
    "paper": 0.25,
    "plastic": 0.30,
    "biological": 0.35,
    "trash": 0.20,
    "battery": 0.20,
    "clothes": 0.20,
    "shoes": 0.20,
    "brown-glass": 0.25,
    "green-glass": 0.25,
    "white-glass": 0.25
}

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def get_unique_target_path(target_dir: Path, filename: str) -> Path:
    target_path = target_dir / filename
    if not target_path.exists():
        return target_path

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1

    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = target_dir / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def get_all_target_boxes(result, model_names, test_classes, class_thresholds):
    if result.boxes is None or len(result.boxes) == 0:
        return []

    cls_ids = result.boxes.cls.tolist()
    confs = result.boxes.conf.tolist()
    xyxys = result.boxes.xyxy.tolist()

    allowed_classes = {c.lower() for c in test_classes}
    matched_boxes = []

    for cls_id, conf, xyxy in zip(cls_ids, confs, xyxys):
        cls_name = model_names.get(int(cls_id), str(cls_id)).lower()

        if cls_name in allowed_classes:
            threshold = class_thresholds.get(cls_name, 0.25)
            if conf >= threshold:
                matched_boxes.append(xyxy)

    return matched_boxes


def crop_and_save_image(img_path: Path, xyxy, save_path: Path, padding: int = 0) -> bool:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    x1, y1, x2, y2 = map(int, xyxy)

    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
        return False

    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(save_path)
    return True


def main():
    if not SOURCE_DIR.exists():
        print(f"[ERROR] Folder not found: {SOURCE_DIR}")
        return

    CROP_TARGET_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [p for p in SOURCE_DIR.iterdir() if is_image_file(p)],
        key=lambda p: p.name.lower()
    )

    if not image_paths:
        print("[INFO] No images found.")
        return

    print("Loading model...")
    model = YOLO(YOLO_MODEL)

    total_crops = 0
    skipped_images = 0

    for img_path in image_paths:
        try:
            results = model(
                str(img_path),
                verbose=False,
                device=DEVICE,
                iou=IOU,
                agnostic_nms=AGNOSTIC,
            )

            result = results[0]

            boxes = get_all_target_boxes(
                result,
                model.names,
                FILTER_CLASSES,
                CLASS_THRESHOLDS
            )

            object_count = len(boxes)

            if not (MIN_OBJECTS <= object_count <= MAX_OBJECTS):
                print(f"[SKIP] {img_path.name} | count={object_count}")
                skipped_images += 1
                continue

            saved_this_image = 0

            for i, box in enumerate(boxes, start=1):
                save_name = f"{img_path.stem}_obj{i}{img_path.suffix}"
                save_path = get_unique_target_path(CROP_TARGET_DIR, save_name)

                success = crop_and_save_image(
                    img_path=img_path,
                    xyxy=box,
                    save_path=save_path,
                    padding=CROP_PADDING
                )

                if success:
                    saved_this_image += 1
                    total_crops += 1
                    print(f"[CROP] {img_path.name} -> {save_name}")
                else:
                    print(f"[SKIP CROP] {img_path.name} -> obj{i} smaller than {MIN_CROP_SIZE}")

            print(f"[DONE] {img_path.name} | objects={object_count} | saved={saved_this_image}")

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print("\nFinished.")
    print(f"Total crops saved: {total_crops}")
    print(f"Skipped images: {skipped_images}")


if __name__ == "__main__":
    main()