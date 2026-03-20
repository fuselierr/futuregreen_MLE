from rest_framework import serializers
from .models import Feedback


# Serializer for CNN prediction input, output, health check response, and model info response. Validates input data and formats output data for the CNN prediction API endpoints.
class PredictionInputSerializer(serializers.Serializer):
    """Serializer for CNN prediction input"""
    
    image_source = serializers.CharField(
        max_length=1,
        help_text="Source of the image: 'w' for URL, 'm' for direct upload"
    )
    image_name = serializers.CharField(
        max_length=255,
        help_text="Name of the image file"
    )
    image_data = serializers.CharField(
        help_text="Base64-encoded image data"
    )
    image_width = serializers.IntegerField(
        help_text="Original image width in pixels",
        min_value=1
    )
    image_height = serializers.IntegerField(
        help_text="Original image height in pixels",
        min_value=1
    )


    def validate_image_name(self, value):
        """Validate image name is not empty"""
        if not value.strip():
            raise serializers.ValidationError("image_name cannot be empty")
        return value

    def validate_image_data(self, value):
        """Validate image data is a non-empty base64 string"""
        if not value.strip():
            raise serializers.ValidationError("image_data cannot be empty")
        return value


# Serializer for CNN prediction output, including predicted class label and confidence score. Formats the output data for the API response and ensures it adheres to the expected structure.
class PredictionOutputSerializer(serializers.Serializer):
    """Serializer for prediction output"""
    image_name = serializers.CharField()
    prediction_result = serializers.CharField()
    confidence = serializers.FloatField()


# Serializer for health check endpoint response, including status, model loaded flag, and a descriptive message. Used to format the response data for the health check API endpoint and provide useful information about the API's operational status.
class HealthCheckSerializer(serializers.Serializer):
    """Serializer for health check endpoint response"""
    status = serializers.CharField()
    model_loaded = serializers.BooleanField()
    message = serializers.CharField()


# Serializer for model info endpoint response, including model name, input shape, output shape, total layers, total parameters, trainable parameters, and non-trainable parameters. Used to format the response data for the model info API endpoint and provide detailed information about the CNN model architecture and parameters.
class ModelInfoSerializer(serializers.Serializer):
    """Serializer for model info endpoint response"""
    model_name = serializers.CharField()
    input_shape = serializers.ListField(child=serializers.IntegerField())
    output_shape = serializers.ListField(child=serializers.IntegerField())
    total_layers = serializers.IntegerField()
    total_params = serializers.IntegerField()
    trainable_params = serializers.IntegerField()
    non_trainable_params = serializers.IntegerField()


# Serializer for user feedback, including star rating and optional feedback text. Validates and stores user reviews in the database.
class FeedbackSerializer(serializers.ModelSerializer):
    """Serializer for user feedback/review submission"""
    class Meta:
        model = Feedback
        fields = ['id', 'rating', 'feedback', 'timestamp']
        read_only_fields = ['id', 'timestamp']
    
    def validate_rating(self, value):
        """Validate rating is between 1 and 5"""
        if not 1 <= value <= 5:
            raise serializers.ValidationError("Rating must be between 1 and 5")
        return value
