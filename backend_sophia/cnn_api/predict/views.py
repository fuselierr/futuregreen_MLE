import numpy as np
import cv2
import logging
import base64
import io
from pathlib import Path

from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from PIL import Image

from .serializers import (
    PredictionInputSerializer,
    PredictionOutputSerializer,
    HealthCheckSerializer,
    ModelInfoSerializer,
)

# Configure logging
logger = logging.getLogger(__name__)
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "api.log"

# Configure file handler for logging
file_handler = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


class PredictionViewSet(viewsets.ViewSet):
    """
    API ViewSet for CNN image predictions
    
    Processes RGB images using the TrashCNN model:
    - Resizes images to 150x150
    - Normalizes pixel values
    - Makes predictions without storing history
    """
    
    parser_classes = (JSONParser,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.model_name = None
        self._load_model()

    def _load_model(self):
        """Load the trained CNN model"""
        if self.model is None:
            from tensorflow.keras.models import load_model
            model_path = Path(__file__).resolve().parent.parent / "TrashCNN_es_v1.1.keras"
            try:
                self.model = load_model(str(model_path))
                self.model_name = model_path.stem  # Extract filename without extension
                logger.info(f"CNN model loaded successfully from {model_path}")
            except Exception as e:
                error_msg = f"Failed to load CNN model from {model_path}: {str(e)}"
                logger.error(error_msg)
                self.model = None
                self.model_name = None

    def _decode_base64_image(self, base64_string):
        """
        Decode a Base64-encoded image string to a numpy array in RGB format
        
        Args:
            base64_string: Base64-encoded image data as string
            
        Returns:
            Numpy array with image data in RGB format (height, width, 3)
        """
        try:
            # Decode base64 string to bytes
            image_bytes = base64.b64decode(base64_string)
            
            # Read image using PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL Image to numpy array
            image_array = np.array(image, dtype=np.uint8)
            
            logger.debug(f"Base64 image decoded successfully. Shape: {image_array.shape}")
            
            return image_array
            
        except Exception as e:
            error_msg = f"Failed to decode Base64 image: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def create(self, request, *args, **kwargs):
        """
        Process image and return prediction without storing in database
        
        Expected input:
        {
            "image_name": "trash_image.jpg",
            "image_data": [[[R, G, B], [R, G, B], ...], ...],
            "image_width": 640,
            "image_height": 480
        }
        """
        logger.info("Received prediction request")
        serializer = PredictionInputSerializer(data=request.data)
        
        try:
            serializer.is_valid(raise_exception=True)
        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(error_msg)
            return Response(
                {"error": error_msg},
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        try:
            # Extract validated data
            image_data = serializer.validated_data.get("image_data")
            image_width = serializer.validated_data.get("image_width")
            image_height = serializer.validated_data.get("image_height")
            image_name = serializer.validated_data.get("image_name")
            
            logger.info(f"Processing image: {image_name} ({image_width}x{image_height})")
            
            # Decode Base64 image to numpy array in RGB format
            image_array = self._decode_base64_image(image_data)            

            # Prepare image: resize, normalize, and fix dimensions
            processed_image = self._preprocess_image(image_array, image_width, image_height)
            prediction_result, confidence = self._predict(processed_image)
            
            logger.info(
                f"Prediction successful for {image_name}: "
                f"{prediction_result} (confidence: {confidence:.4f})"
            )
            
            # Return result without storing
            output_data = {
                "image_name": image_name,
                "prediction_result": prediction_result,
                "confidence": float(confidence),
            }
            
            output_serializer = PredictionOutputSerializer(output_data)
            return Response(output_serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Response(
                {"error": error_msg},
                status=status.HTTP_400_BAD_REQUEST,
            )

    def _preprocess_image(self, image_array, width, height):
        """
        Preprocess image: convert to numpy array, resize to 150x150, and normalize
        
        Args:
            image_array: numpy array with shape (height, width, 3)
            width: original image width
            height: original image height
            
        Returns:
            Preprocessed image array ready for model input
        """
        try:
            # Validate dimensions
            if image_array.shape != (height, width, 3):
                raise ValueError(
                    f"Expected image shape ({height}, {width}, 3), got {image_array.shape}"
                )
            
            logger.debug(f"Image converted to numpy array with shape {image_array.shape}")
            
            # Resize to 150x150 using cv2
            resized_image = cv2.resize(image_array, (150, 150), interpolation=cv2.INTER_LINEAR)
            
            # Normalize: convert from [0, 255] to [0, 1]
            normalized_image = resized_image.astype(np.float32) / 255.0
            
            # Add batch dimension: (150, 150, 3) -> (1, 150, 150, 3)
            batched_image = np.expand_dims(normalized_image, axis=0)
            
            logger.debug(f"Image preprocessed: resized to 150x150, normalized, and batched")
            
            return batched_image
            
        except Exception as e:
            error_msg = f"Image preprocessing failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _predict(self, image_array):
        """
        Make prediction using the CNN model
        
        Args:
            image_array: preprocessed image data with batch dimension
            
        Returns:
            Tuple of (prediction_label, confidence_score)
        """
        if self.model is None:
            self._load_model()
        
        try:
            predictions = self.model.predict(image_array, verbose=0)
            confidence = float(np.max(predictions))
            predicted_class = int(np.argmax(predictions))
            
            # Class labels - adjust based on your model's training classes
            class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
            prediction_label = (
                class_labels[predicted_class] 
                if predicted_class < len(class_labels) 
                else f"class_{predicted_class}"
            )
            
            logger.debug(f"Model prediction: class={prediction_label}, confidence={confidence:.4f}")
            
            return prediction_label, confidence
        except Exception as e:
            error_msg = f"Model prediction failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


    @action(detail=False, methods=["get"])
    def check_health(self, request):
        """
        Health check endpoint to verify if the CNN model is loaded
        
        Returns:
            - status: "healthy" if model is loaded, "unhealthy" otherwise
            - model_loaded: boolean indicating if CNN model is loaded
            - message: descriptive message about the health status
        """
        try:
            if self.model is None:
                self._load_model()
            
            if self.model is not None:
                health_data = {
                    "status": "healthy",
                    "model_loaded": True,
                    "message": "CNN model is loaded and ready for predictions"
                }
                logger.info("Health check: API is healthy")
                return Response(health_data, status=status.HTTP_200_OK)
            else:
                health_data = {
                    "status": "unhealthy",
                    "model_loaded": False,
                    "message": "CNN model failed to load"
                }
                logger.warning("Health check: API is unhealthy - model not loaded")
                return Response(health_data, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(error_msg)
            health_data = {
                "status": "unhealthy",
                "model_loaded": False,
                "message": error_msg
            }
            return Response(health_data, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    @action(detail=False, methods=["get"])
    def model_info(self, request):
        """
        Get model parameters endpoint to retrieve information about the trained CNN model
        
        Returns:
            - model_name: name of the model
            - input_shape: shape of the model's input layer
            - output_shape: shape of the model's output layer
            - total_layers: total number of layers in the model
            - total_params: total number of parameters in the model
            - trainable_params: number of trainable parameters
            - non_trainable_params: number of non-trainable parameters
        """
        try:
            if self.model is None:
                self._load_model()
            
            if self.model is None:
                error_msg = "CNN model is not loaded"
                logger.error(error_msg)
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            
            # Extract model information
            model_name = self.model_name 
            
            # Get input shape (excluding batch dimension)
            input_shape = list(self.model.input_shape[1:]) if hasattr(self.model, 'input_shape') else []
            
            # Get output shape (excluding batch dimension)
            output_shape = list(self.model.output_shape[1:]) if hasattr(self.model, 'output_shape') else []
            
            # Count layers
            total_layers = len(self.model.layers)
            
            # Get parameter counts
            total_params = int(self.model.count_params())
            
            # Calculate trainable and non-trainable parameters
            trainable_params = int(
                sum(
                    np.prod(w.shape) 
                    for w in self.model.trainable_weights
                )
            )
            non_trainable_params = total_params - trainable_params
            
            model_data = {
                "model_name": model_name,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "total_layers": total_layers,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "non_trainable_params": non_trainable_params,
            }
            
            logger.info(f"Model info retrieved: {model_name}")
            
            model_info_serializer = ModelInfoSerializer(model_data)
            return Response(model_info_serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            error_msg = f"Failed to retrieve model info: {str(e)}"
            logger.error(error_msg)
            return Response(
                {"error": error_msg},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
