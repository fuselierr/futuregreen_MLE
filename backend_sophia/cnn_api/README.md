# CNN Image Prediction API

## Overview
This is a Django REST API for processing trash classification using a trained TensorFlow/Keras CNN model. The API processes single RGB images, resizes them to 150x150 pixels, normalizes them, and returns predictions without storing history. All errors are logged to a file for debugging.

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run migrations:
```bash
python manage.py migrate
```

3. Start the development server:
```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000/api/`

## API Endpoints

### Single Image Prediction
**POST** `/api/predict/`

Submit a single image for prediction.

#### Request Format:
```json
{
  "image_name": "trash_image.jpg",
  "image_data": [[[R, G, B], [R, G, B], ...], ...],
  "image_width": 640,
  "image_height": 480
}
```

**Parameters:**
- `image_name` (string): Name of the image file
- `image_data` (3D array): Nested list representing image pixels in RGB format [height][width][RGB]
  - Must be a 3-dimensional array with exactly 3 RGB channels
  - Each pixel should have 3 values: Red, Green, Blue (0-255)
  - Shape must be: `[height][width][3]`
- `image_width` (integer): Original image width in pixels
- `image_height` (integer): Original image height in pixels

#### Validation Rules:
- Image data must be exactly 3-dimensional
- Each pixel must have exactly 3 RGB channels
- Image name cannot be empty
- Image width and height must be positive integers

#### Example cURL Request:
```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "image_name": "trash.jpg",
    "image_data": [[[100, 150, 200], [110, 160, 210], ...], ...],
    "image_width": 640,
    "image_height": 480
  }'
```

#### Response:
```json
{
  "image_name": "trash_image.jpg",
  "prediction_result": "cardboard",
  "confidence": 0.9823
}
```

#### Error Response:
```json
{
  "error": "image_data must have exactly 3 RGB channels"
}
```

### Health Check
**GET** `/api/predict/check_health/`

Check the health status of the API and verify if the CNN model is loaded.

#### Response (Model Loaded):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "CNN model is loaded and ready for predictions"
}
```
Status Code: `200 OK`

#### Response (Model Not Loaded):
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "message": "CNN model failed to load"
}
```
Status Code: `503 Service Unavailable`

#### Example cURL Request:
```bash
curl -X GET http://localhost:8000/api/predict/check_health/
```

## Image Processing

The API automatically performs the following preprocessing steps:

1. **Validation**: Verifies image is 3D with 3 RGB channels
2. **Conversion**: Converts input nested list to numpy array
3. **Dimension Check**: Verifies image dimensions match RGB format (height × width × 3)
4. **Resizing**: Resizes image to 150×150 pixels using OpenCV (cv2.INTER_LINEAR)
5. **Normalization**: Converts pixel values from [0, 255] to [0, 1]
6. **Batch Dimension**: Adds batch dimension (1, 150, 150, 3) for model input

## CNN Model

**Model Name**: TrashCNN_es_v1.1.keras
**Framework**: TensorFlow/Keras
**Input Shape**: (1, 150, 150, 3)
**Output Classes**: 
- cardboard
- glass
- metal
- paper
- plastic
- trash

## Error Handling & Logging

### Logging
All API activities and errors are logged to `logs/api.log`:
- Model loading events
- Prediction requests and results
- Validation errors
- Processing errors
- Exceptions with full stack traces

Example log entry:
```
2026-01-31 10:30:45,123 - predict.views - INFO - Received prediction request
2026-01-31 10:30:45,456 - predict.views - INFO - Processing image: trash.jpg (640x480)
2026-01-31 10:30:46,789 - predict.views - INFO - Prediction successful for trash.jpg: cardboard (confidence: 0.9823)
```

### Common Error Cases

**Invalid image dimensions:**
```json
{
  "error": "Expected image shape (480, 640, 3), got (480, 640, 4)"
}
```

**Invalid RGB channels:**
```json
{
  "error": "image_data must have exactly 3 RGB channels"
}
```

**Image not 3-dimensional:**
```json
{
  "error": "image_data must be a 3-dimensional array"
}
```

## Key Features

- ✅ **Single Image Processing**: Processes one image per request
- ✅ **Comprehensive Logging**: All errors logged to `logs/api.log`
- ✅ **RGB Image Validation**: Ensures 3-dimensional array with 3 channels
- ✅ **No History Tracking**: Predictions not stored in database
- ✅ **Automatic Resizing**: Images resized to 150×150
- ✅ **Normalization**: Pixel values automatically normalized to [0, 1]
- ✅ **Confidence Scores**: Returns prediction confidence for each result
- ✅ **Comprehensive Error Handling**: Detailed validation and error messages

## Usage Example (Python)

```python
import requests
import numpy as np

# Create sample RGB image (640x480)
image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# Prepare prediction request
payload = {
    "image_name": "test_image.jpg",
    "image_data": image.tolist(),
    "image_width": 640,
    "image_height": 480
}

# Make prediction
response = requests.post("http://localhost:8000/api/predict/", json=payload)
result = response.json()

print(f"Prediction: {result['prediction_result']}")
print(f"Confidence: {result['confidence']}")
```

## Dependencies

- Django 6.0.1
- djangorestframework 3.14.0
- TensorFlow 2.15.0
- OpenCV (cv2) 4.8.1.78
- NumPy 1.24.3

See `requirements.txt` for complete list.

## Notes

- Predictions are **not stored** in the database
- Only **one image per request** is supported
- Each request is processed independently
- The model is loaded once and cached in memory for performance
- All errors and warnings are logged to `logs/api.log`
- Suitable for real-time prediction use cases
