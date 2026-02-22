from rest_framework import serializers


class PredictionInputSerializer(serializers.Serializer):
    """Serializer for CNN prediction input"""
    
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


class PredictionOutputSerializer(serializers.Serializer):
    """Serializer for prediction output"""
    image_name = serializers.CharField()
    prediction_result = serializers.CharField()
    confidence = serializers.FloatField()


class HealthCheckSerializer(serializers.Serializer):
    """Serializer for health check endpoint response"""
    status = serializers.CharField()
    model_loaded = serializers.BooleanField()
    message = serializers.CharField()


class ModelInfoSerializer(serializers.Serializer):
    """Serializer for model info endpoint response"""
    model_name = serializers.CharField()
    input_shape = serializers.ListField(child=serializers.IntegerField())
    output_shape = serializers.ListField(child=serializers.IntegerField())
    total_layers = serializers.IntegerField()
    total_params = serializers.IntegerField()
    trainable_params = serializers.IntegerField()
    non_trainable_params = serializers.IntegerField()
