# CNN Image Prediction API

## Overview
This is a Django REST API for processing trash classification using a trained TensorFlow/Keras CNN model. The API processes RGB images (provided as Base64-encoded strings), resizes them to 150x150 pixels, normalizes them, and returns predictions without storing history. All errors are logged to a file for debugging.

## Features
- ✅ Base64-encoded image input support
- ✅ Multiple model selection (benchmark or main model)
- ✅ Automatic image preprocessing (resize to 150x150, normalization)
- ✅ Real-time CNN predictions with confidence scores
- ✅ Health check endpoint with model status
- ✅ Model information endpoint with architecture details
- ✅ User feedback collection and storage
- ✅ Comprehensive input validation with detailed error messages
- ✅ Full request/response logging to file
- ✅ No prediction history stored in database

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

### 1. Single Image Prediction
**POST** `/api/predict/`

Submit a Base64-encoded image for trash classification prediction.

#### Request Format:
```json
{
  "image_source": "w",
  "image_name": "trash_image.jpg",
  "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "image_width": 640,
  "image_height": 480
}
```

**Parameters:**
- `image_source` (string, required): Model source to use for prediction
  - `"w"`: Web/benchmark model (benchmark_model.keras)
  - `"m"`: Main model (TrashCNN_es_v1.1.keras)
- `image_name` (string): Name of the image file (non-empty)
- `image_data` (string): Base64-encoded image data
  - The image will be automatically decoded and converted to RGB format
  - Supports common image formats (PNG, JPG, BMP, etc.)
  - RGBA images are automatically converted to RGB
- `image_width` (integer): Original image width in pixels
- `image_height` (integer): Original image height in pixels

#### Image Processing Pipeline:
1. Decodes Base64 string to image bytes
2. Converts image to RGB format (handles RGBA, grayscale, etc.)
3. Validates image dimensions match the declared width/height
4. Resizes image to 150x150 pixels using bilinear interpolation
5. Normalizes pixel values from [0, 255] to [0, 1]
6. Adds batch dimension for model input
7. Performs CNN prediction

#### Supported Classes:
`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`

#### Validation Rules:
- Image name must be non-empty
- Image width and height must be positive integers
- Base64-encoded image data must be valid and decodable
- Decoded image dimensions must match declared width/height

#### Example Python Request:
```python
import requests
import base64

# Load and encode image
with open('trash.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    "image_source": "w",  # Use web model
    "image_name": "trash.jpg",
    "image_data": image_base64,
    "image_width": 640,
    "image_height": 480
}

response = requests.post('http://localhost:8000/api/predict/', json=payload)
print(response.json())
```

#### Example cURL Request:
```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "image_source": "w",
    "image_name": "trash.jpg",
    "image_data": "'$(base64 -i trash.jpg)'",
    "image_width": 640,
    "image_height": 480
  }'
```

#### Success Response (200 OK):
```json
{
  "image_name": "trash_image.jpg",
  "prediction_result": "cardboard",
  "confidence": 0.9823
}
```

#### Error Response (400 Bad Request):
```json
{
  "error": "Failed to decode Base64 image: [error details]"
}
```

### 2. Health Check
**GET** `/api/predict/check_health/`

Check the health status of the API and verify if the CNN model is loaded.

#### Response (Model Loaded - 200 OK):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "CNN model is loaded and ready for predictions"
}
```

#### Response (Model Not Loaded - 503 Service Unavailable):
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "message": "CNN model failed to load"
}
```

#### Example cURL Request:
```bash
curl -X GET http://localhost:8000/api/predict/check_health/
```

#### Example Python Request:
```python
import requests

response = requests.get('http://localhost:8000/api/predict/check_health/')
print(response.json())
```

### 3. Model Information
**GET** `/api/predict/model_info/`

Retrieve detailed information about the trained CNN model including architecture and parameter counts.

#### Response (200 OK):
```json
{
  "model_name": "TrashCNN_es_v1.1",
  "input_shape": [150, 150, 3],
  "output_shape": [6],
  "total_layers": 14,
  "total_params": 2152486,
  "trainable_params": 2152486,
  "non_trainable_params": 0
}
```

**Response Fields:**
- `model_name` (string): Name of the loaded model file
- `input_shape` (array): Input layer shape (height, width, channels) - always [150, 150, 3]
- `output_shape` (array): Output layer shape - [6] for 6 trash classes
- `total_layers` (integer): Total number of layers in the model
- `total_params` (integer): Total number of model parameters
- `trainable_params` (integer): Number of trainable parameters
- `non_trainable_params` (integer): Number of non-trainable/frozen parameters

#### Error Response (503 Service Unavailable):
```json
{
  "error": "CNN model is not loaded"
}
```

#### Example cURL Request:
```bash
curl -X GET http://localhost:8000/api/predict/model_info/
```

#### Example Python Request:
```python
import requests

response = requests.get('http://localhost:8000/api/predict/model_info/')
model_info = response.json()
print(f"Model: {model_info['model_name']}")
print(f"Layers: {model_info['total_layers']}")
print(f"Parameters: {model_info['total_params']}")
```

### 4. User Feedback
**POST** `/api/predict/user_feedback/`

Submit user feedback on model predictions to help improve the model. Feedback is stored in the SQLite database.

#### Request Format:
```json
{
  "model_prediction": "plastic",
  "user_prediction": "paper",
  "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
}
```

**Parameters:**
- `model_prediction` (string): The predicted trash type from the model (non-empty)
- `user_prediction` (string): The actual/corrected trash type according to the user (non-empty)
- `image_data` (string): Base64-encoded image data used for prediction

#### Validation Rules:
- Model prediction must be non-empty
- User prediction must be non-empty
- Image data must be valid Base64-encoded string

#### Success Response (201 Created):
```json
{
  "success": true,
  "message": "User feedback stored successfully",
  "feedback_id": 1
}
```

#### Error Response (400 Bad Request):
```json
{
  "error": "Validation error: [error details]",
  "success": false
}
```

#### Error Response (500 Internal Server Error):
```json
{
  "error": "Failed to store user feedback: [error details]",
  "success": false
}
```

#### Example Python Request:
```python
import requests
import base64

# Load and encode image
with open('trash.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    "model_prediction": "plastic",
    "user_prediction": "paper",
    "image_data": image_base64
}

response = requests.post('http://localhost:8000/api/predict/user_feedback/', json=payload)
print(response.json())
```

#### Example cURL Request:
```bash
curl -X POST http://localhost:8000/api/predict/user_feedback/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_prediction": "plastic",
    "user_prediction": "paper",
    "image_data": "'$(base64 -i trash.jpg)'"
  }'
```

#### Database Storage:
User feedback is stored in the SQLite database with the following fields:
- `id`: Unique feedback identifier
- `model_prediction`: Model's predicted trash type
- `user_prediction`: User's corrected trash type
- `image_data`: Base64-encoded image for reference
- `created_at`: Timestamp of when feedback was submitted

You can view stored feedback via Django admin at `/admin/predict/userfeedback/`

## Image Processing Pipeline

The API automatically performs the following preprocessing steps on Base64-encoded input:

1. **Base64 Decoding**: Converts Base64 string to image bytes
2. **Format Detection**: Automatically detects image format (PNG, JPG, BMP, etc.)
3. **RGB Conversion**: Converts any color space (RGBA, grayscale, etc.) to RGB
4. **Dimension Validation**: Verifies decoded image matches declared width/height
5. **Resizing**: Resizes image to 150×150 pixels using OpenCV bilinear interpolation
6. **Normalization**: Converts pixel values from [0, 255] to [0, 1] range
7. **Batch Dimension**: Adds batch dimension (1, 150, 150, 3) for model input
8. **Prediction**: Performs CNN inference and returns results

## CNN Model Details

**Model Name**: TrashCNN_es_v1.1.keras
**Framework**: TensorFlow/Keras 2.20.0
**Input Shape**: (batch_size, 150, 150, 3)
**Output Shape**: (batch_size, 6)

**Supported Classification Classes** (6 classes):
1. `cardboard`
2. `glass`
3. `metal`
4. `paper`
5. `plastic`
6. `trash`

## Error Handling & Logging

### Logging
All API activities and errors are automatically logged to `logs/api.log`:
- Model loading/initialization events
- Each prediction request and results
- Validation errors with details
- Image preprocessing errors
- Model inference errors
- Exceptions with full stack traces

Log file location: `logs/api.log`

Example log entries:
```
2026-03-06 10:30:45,123 - predict.views - INFO - CNN model loaded successfully from /path/to/TrashCNN_es_v1.1.keras
2026-03-06 10:30:50,456 - predict.views - INFO - Received prediction request
2026-03-06 10:30:50,789 - predict.views - INFO - Processing image: trash.jpg (640x480)
2026-03-06 10:30:51,012 - predict.views - DEBUG - Base64 image decoded successfully. Shape: (480, 640, 3)
2026-03-06 10:30:51,234 - predict.views - INFO - Prediction successful for trash.jpg: cardboard (confidence: 0.9823)
```

### Common Error Cases

**Invalid Base64 encoding:**
```json
{
  "error": "Failed to decode Base64 image: Incorrect padding"
}
```

**Mismatched image dimensions:**
```json
{
  "error": "Image preprocessing failed: Expected image shape (480, 640, 3), got (384, 512, 3)"
}
```

**Missing required field:**
```json
{
  "error": "Validation error: {'image_data': [ErrorDetail(string='This field is required.', code='required')]}"
}
```

**Empty image name:**
```json
{
  "error": "Validation error: {'image_name': [ErrorDetail(string='This field may not be blank.', code='blank')]}"
}
```

**Model not loaded:**
```json
{
  "error": "CNN model is not loaded"
}
```

## Key Features

- ✅ **Base64 Image Support**: Accepts Base64-encoded image data
- ✅ **Multiple Model Selection**: Choose between benchmark and main model
- ✅ **Automatic Format Conversion**: Handles PNG, JPG, BMP, etc.
- ✅ **Automatic Color Space Handling**: Converts RGBA, grayscale, etc. to RGB
- ✅ **Single Image Processing**: Processes one image per request
- ✅ **Real-time Predictions**: No history stored in database
- ✅ **Confidence Scores**: Returns prediction confidence for each result
- ✅ **Automatic Resizing**: Images resized to 150×150 automatically
- ✅ **Normalization**: Pixel values automatically normalized to [0, 1]
- ✅ **User Feedback Collection**: Store user corrections in database
- ✅ **Comprehensive Logging**: All activities logged to `logs/api.log`
- ✅ **Health Monitoring**: Endpoints to check API and model status
- ✅ **Model Introspection**: Endpoint to retrieve model architecture details
- ✅ **Detailed Error Messages**: Helpful validation and processing error messages
- ✅ **In-Memory Caching**: Model loaded once and cached for performance

## Usage Examples

#### Python Example - Health Check
```python
import requests

# Check API health
response = requests.get("http://localhost:8000/api/predict/check_health/")
health = response.json()

if health['model_loaded']:
    print("✓ API is healthy and model is loaded")
else:
    print("✗ API is unhealthy - model not loaded")
```

#### Python Example - Model Info
```python
import requests

# Get model information
response = requests.get("http://localhost:8000/api/predict/model_info/")
model_info = response.json()

print(f"Model: {model_info['model_name']}")
print(f"Input Shape: {model_info['input_shape']}")
print(f"Total Parameters: {model_info['total_params']:,}")
print(f"Trainable Parameters: {model_info['trainable_params']:,}")
```

#### Python Example - User Feedback
```python
import requests
import base64

# Load and encode image
with open('trash.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Submit feedback
payload = {
    "model_prediction": "plastic",
    "user_prediction": "paper",
    "image_data": image_base64
}

response = requests.post("http://localhost:8000/api/predict/user_feedback/", json=payload)
feedback = response.json()

if feedback['success']:
    print(f"✓ Feedback stored with ID: {feedback['feedback_id']}")
else:
    print(f"✗ Failed to store feedback: {feedback['error']}")
```

## Testing

A comprehensive test suite is provided in `test_api.py` that covers:
- Health check endpoint
- Single image prediction with valid input (both model sources)
- Input validation (wrong dimensions, wrong channels, non-3D arrays)
- Empty image name validation
- Model info endpoint
- User feedback endpoint

### Running Tests
```bash
python test_api.py
```

The test suite will output detailed test results for all endpoints.

## Dependencies

- Django 6.0.1
- djangorestframework 3.14.0
- TensorFlow 2.20.0
- Keras 3.10.0+
- OpenCV (cv2) 4.9.0+
- NumPy (latest <2.0)
- Pillow (for image handling)

See `requirements.txt` for complete list.

## Notes

- Predictions are **not stored** in the database
- User feedback **is stored** in the SQLite database for model improvement
- Only **one image per request** is supported
- Image data must be provided as **Base64-encoded string**
- The model is loaded once at startup and cached in memory
- All activities are logged to `logs/api.log` for debugging
- The API is suitable for real-time trash classification use cases
- Returns confidence scores (0.0 to 1.0) for predictions
- Supports images of any size - automatically resized to 150×150
- Multiple models available: benchmark model (`w`) and main model (`m`)
