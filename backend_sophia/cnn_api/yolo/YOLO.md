# YOLO Waste Classification Module - Specification

## Overview

This module provides YOLOv8-based object detection and image processing utilities for waste classification and dataset preparation. It uses a pre-trained YOLOv8 nano model (`yolov8n-waste-12cls-best.pt`) to detect and classify waste items across 12 classes with configurable confidence thresholds.

**Model**: YOLOv8 Nano - Waste Classification (12 classes)
**Framework**: Ultralytics YOLO
**Purpose**: Image filtering, object detection, dataset organization, and cropping for waste classification pipelines

---

## Model Information

### Supported Classes (12 classes)

1. `cardboard` - Cardboard boxes and packaging
2. `metal` - Metal cans, containers, etc.
3. `paper` - Paper sheets, documents, etc.
4. `plastic` - Plastic bottles, bags, etc.
5. `biological` / `organic` - Organic/compostable waste
6. `trash` - General trash/non-recyclable items
7. `battery` - Batteries
8. `clothes` - Textile/clothing items
9. `shoes` - Footwear
10. `brown-glass` - Brown glass items
11. `green-glass` - Green glass items
12. `white-glass` - Clear/white glass items

### Model Configuration

- **Framework**: Ultralytics YOLOv8
- **Model Size**: Nano (lightweight, fast inference)
- **IOU Threshold**: 0.50 (default, configurable)
- **NMS**: Agnostic NMS disabled by default (class-aware NMS)
- **Device**: CPU/CUDA (auto-detected)

### Confidence Thresholds

Per-class confidence thresholds ensure quality detections:

```python
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
```

---

## Scripts and Components

### 1. `single_image.py`

**Purpose**: Process a single image through a complete validation pipeline with YOLO detection and cropping.

**Pipeline**:
1. **Input Resolution Check** - Verify image meets minimum dimensions (250×250 default)
2. **YOLO Detection** - Run object detection, require exactly 1 valid detection
3. **Bounding Box Validation** - Verify detected object meets minimum size (112×112 default)
4. **Cropping** - Crop image to bounding box with padding (10px default)

**Usage**:
```bash
python single_image.py <image_path> [output_path]
```

**Arguments**:
- `image_path` (required): Path to input image file
- `output_path` (optional): Path to save cropped image. Defaults to `<stem>_cropped.jpg`

**Configuration**:
```python
MIN_INPUT_RESOLUTION = (250, 250)   # Minimum input image dimensions
MIN_CROP_RESOLUTION  = (112, 112)   # Minimum bounding box dimensions
CROP_PADDING = 10                    # Padding around bounding box (pixels)
FILTER_CLASSES = [...]               # Classes to detect
```

**Exit Codes**:
- `0` - Success
- `1` - Failure (image too small, wrong detection count, etc.)

**Output**:
- Cropped image saved to specified path
- Console messages for debugging (OK, FAIL, INFO, DONE)

---

### 2. `yolo_filter2.py`

**Purpose**: Filter images based on object count, moving/copying images with incorrect detection counts to a target directory.

**Functionality**:
- Scans source directory for images
- Runs YOLO detection on each image
- Counts valid objects matching filter criteria
- Moves/copies images with object_count ≠ 1 to target directory
- Reports summary statistics

**Configuration**:
```python
SOURCE_DIR = Path(r"TEST/paper_tt")              # Input directory
TARGET_DIR = Path(r"TEST/rejected/multiple")     # Output directory for rejected images
FILTER_CLASSES = ["paper"]                       # Classes to filter for
COPY_INSTEAD_OF_MOVE = False                     # Copy or move files
CLEAR_TARGET_BEFORE_RUN = True                   # Clear target before processing
IMAGE_EXTS = {".jpg"}                            # Supported image extensions
```

**Output**:
- Moves/copies images with incorrect object counts
- Console report: moved count, kept count, error count
- Unique naming with counter to prevent overwrites

**Statistics**:
```
Moved/Copied to multiple: N
Kept: N
Errors: N
```

---

### 3. `yolo_filter3.py`

**Purpose**: Comprehensive dataset filtering and organization with multi-class support, resolution filtering, and organized output structure.

**Features**:
- Processes images organized by class subdirectories
- Accepts only images with exactly 1 valid detection
- Optional resolution filtering for cropped images
- Automatically organizes accepted images into class-based output structure
- Moves rejected images to a `rejected/` folder with metadata
- Resumes processing from previous partial runs
- Detailed per-class and overall statistics

**Configuration**:
```python
SOURCE_DIR = Path(r"data\processed\consolidated_raws")
TARGET_DIR = Path(r"data\processed\dataset_vers3.2")
YOLO_MODEL = SCRIPT_DIR / "yolov8n-waste-12cls-best.pt"
IOU = 0.50
AGNOSTIC = False
FILTER_CLASSES = ["paper", "cardboard", "metal", ...]
CROP_PADDING = 10
FILTER_MIN_RESOLUTION = True                     # Enable/disable size filtering
MIN_RESOLUTION = (50, 50)                        # Minimum crop dimensions
CLEAR_TARGET_BEFORE_RUN = False                  # Preserve previous progress
```

**Input Structure**:
```
SOURCE_DIR/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image3.jpg
│   └── ...
└── ...
```

**Output Structure**:
```
TARGET_DIR/
├── class1/
│   ├── class1_1.jpg (cropped)
│   ├── class1_2.jpg
│   └── ...
├── class2/
│   ├── class2_1.jpg
│   └── ...
└── rejected/
    ├── rejected_1.jpg (multi-object)
    ├── rejected_2.jpg (size too small)
    └── ...
```

**Rejection Reasons**:
- **Multi-object** - Object count ≠ 1
- **Size/Resolution** - Cropped image below MIN_RESOLUTION threshold

**Statistics Report**:
```
PROCESSING COMPLETE
─────────────────────────────────────────
Total processed: N
Total accepted: N
Total rejected: N
  ├─ Multi-object: N
  └─ Size/Resolution: N
Total errors: N

Class          Total    Accepted   Rejected  Rej(Multi) Rej(Size)  Errors
────────────────────────────────────────────────────────────────────────
class1         100      80         20        15         5          0
class2         50       45         5         3          2          0
...
```

---

### 4. `yolo_multiCrop.py`

**Purpose**: Extract and crop multiple objects from images with multi-object detection.

**Features**:
- Detects multiple objects in images
- Filters by object count (MIN_OBJECTS to MAX_OBJECTS)
- Crops and saves each detected object separately
- Applies minimum crop size filtering
- Unique naming with object indices

**Configuration**:
```python
SOURCE_DIR = Path(r"rejected/multiple/cardboard")
CROP_TARGET_DIR = Path(r"cropped/cpdCardboard")
YOLO_MODEL = "yolov8n-waste-12cls-best.pt"
IOU = 0.35                           # More lenient for multiple objects
AGNOSTIC = False
DEVICE = "cpu"
MIN_OBJECTS = 2                      # Minimum objects to accept
MAX_OBJECTS = 8                      # Maximum objects to accept
MIN_CROP_SIZE = 50                   # Minimum crop dimensions
CROP_PADDING = 5                     # Padding around each crop
FILTER_CLASSES = ["cardboard"]
```

**Input**:
- Images containing multiple objects of target class

**Output**:
- Individual cropped images: `<original_name>_obj1.jpg`, `<original_name>_obj2.jpg`, etc.
- Cropped images saved to CROP_TARGET_DIR with unique naming

**Statistics**:
```
Total crops saved: N
Skipped images: N (outside object count range or crop size too small)
```

---

## Shared Utilities and Functions

### Helper Functions Used Across Scripts

#### `is_image_file(path: Path) -> bool`
Check if a path is an image file with supported extension (`.jpg`)

#### `count_target_objects(result, model_names, test_classes, class_thresholds) -> int`
Count valid detections matching filter criteria and confidence thresholds

#### `crop_image(img_path: Path, xyxy, padding: int) -> Image.Image`
Crop image to bounding box with padding, clamped to image bounds

#### `get_unique_target_path(target_dir: Path, filename: str) -> Path`
Generate unique output path with counter suffix to prevent overwrites

#### `check_crop_resolution(img_path, xyxy, padding, min_res) -> (bool, int, int)`
Validate cropped image meets minimum resolution requirements

---

## Dependencies

```python
from ultralytics import YOLO          # YOLOv8 detection
from PIL import Image                 # Image processing (PIL/Pillow)
from pathlib import Path              # File path handling
import shutil                         # File operations (move, copy)
import os, re                         # OS and regex utilities
```

**System Requirements**:
- Python 3.8+
- ultralytics >= 8.0.0
- Pillow >= 8.0.0
- numpy (dependency of ultralytics)
- torch/torchvision (dependency of ultralytics)

---

## Usage Examples

### Example 1: Single Image Processing

```bash
# Process single image, save crop with default name
python single_image.py /path/to/image.jpg

# Process and save to specific location
python single_image.py /path/to/image.jpg /output/cropped_image.jpg
```

### Example 2: Filter Images by Object Count

```bash
# Edit configuration in yolo_filter2.py
SOURCE_DIR = Path("my_images")
TARGET_DIR = Path("rejected_images")
FILTER_CLASSES = ["plastic"]

# Run filter
python yolo_filter2.py
```

### Example 3: Organize Dataset

```bash
# Configure yolo_filter3.py
SOURCE_DIR = Path("raw_dataset")
TARGET_DIR = Path("organized_dataset")
FILTER_MIN_RESOLUTION = True
MIN_RESOLUTION = (50, 50)

# Run dataset organization
python yolo_filter3.py
```

### Example 4: Multi-Object Cropping

```bash
# Configure yolo_multiCrop.py
SOURCE_DIR = Path("multi_object_images")
CROP_TARGET_DIR = Path("individual_crops")
MIN_OBJECTS = 2
MAX_OBJECTS = 5

# Extract individual objects
python yolo_multiCrop.py
```

---

## Performance Considerations

### Resolution Thresholds

- **Input Image**: Minimum 250×250 pixels recommended for good detection quality
- **Bounding Box**: Detected objects should be at least 112×112 pixels
- **Crop Output**: Can filter down to 50×50 pixels for training datasets

### Processing Speed

- **YOLOv8 Nano** is optimized for speed (suitable for real-time applications)
- CPU processing: ~100-500ms per image (depends on image size)
- GPU processing: ~10-50ms per image (with CUDA-capable GPU)

### Memory Usage

- Minimal memory footprint for YOLOv8 Nano
- Batch processing not implemented (processes one image at a time)
- Can process thousands of images sequentially without memory issues

---

## Configuration Best Practices

### For Quality Dataset Creation
```python
MIN_INPUT_RESOLUTION = (300, 300)      # Ensure decent input quality
MIN_CROP_RESOLUTION = (100, 100)       # Useful objects in crops
FILTER_MIN_RESOLUTION = True            # Enforce size requirements
IOU = 0.50                              # Standard NMS threshold
```

### For Strict Single-Object Detection
```python
CLASS_THRESHOLDS = {
    "plastic": 0.35,
    "metal": 0.35,
    # Increase thresholds for strict filtering
}
CROP_PADDING = 5                        # Minimal padding for clean crops
```

### For Multi-Object Processing
```python
MIN_OBJECTS = 2
MAX_OBJECTS = 5                         # Limit complexity
IOU = 0.35                              # More lenient NMS
CROP_PADDING = 10                       # Extra padding for separated objects
```

---

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `File not found` | Invalid image path | Verify path exists and is correct |
| `Expected exactly 1 detection` | No detection or multiple objects | Check image quality, adjust confidence thresholds |
| `Resolution below minimum` | Image/crop too small | Use higher resolution input images |
| `Cannot import YOLO` | Ultralytics not installed | `pip install ultralytics` |
| `Model file not found` | yolov8n-waste-12cls-best.pt missing | Ensure model file is in script directory |

### Debugging

Add verbose output:
```python
results = model(str(img_path), verbose=True)  # Show YOLO inference details
```

Check detection results:
```python
result = results[0]
if result.boxes:
    for box in result.boxes:
        print(f"Class: {model.names[int(box.cls)]}, Conf: {box.conf:.2f}")
```

---

## Integration with CNN API

This YOLO module can be integrated with the CNN API (`/predict/`) to create a two-stage pipeline:

1. **YOLO Stage**: Object detection and localization (yolo_filter3.py)
2. **CNN Stage**: Fine-grained classification on cropped images

**Proposed Workflow**:
1. User uploads image
2. YOLO detects and crops object(s)
3. Cropped image(s) sent to CNN for classification
4. Combined results returned to user

---

## Model File

**Filename**: `yolov8n-waste-12cls-best.pt`
- **Size**: Pre-trained YOLOv8 Nano checkpoint
- **Location**: `/Users/sophialuo/github/futuregreen_MLE/backend_sophia/cnn_api/yolo/`
- **Format**: PyTorch model (.pt)
- **Auto-downloads**: On first run if not found (Ultralytics caches models)

---

## Future Enhancements

1. **Batch Processing**: Process multiple images in parallel
2. **API Endpoint**: REST API wrapper for scripts
3. **Real-time Inference**: Streaming video/camera input support
4. **Model Fine-tuning**: Custom training scripts for specific waste categories
5. **Visualization**: Generate bounding box visualizations
6. **Logging**: Structured logging for production environments
7. **Configuration Files**: YAML-based configuration for easier parameter management

---

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics)
- [Waste Classification Dataset Paper](placeholder)

---

## License and Attribution

YOLOv8 model trained on custom waste classification dataset.
Uses Ultralytics YOLOv8 framework (AGPL-3.0).

---

**Last Updated**: March 28, 2026
**Version**: 1.0
