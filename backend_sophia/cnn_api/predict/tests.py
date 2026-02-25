from django.test import TestCase
from rest_framework.test import APIClient
import json
import numpy as np


class PredictionAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_predict_with_valid_data(self):
        """Test prediction with valid RGB image data"""
        # Create dummy 10x10 RGB image data
        pixel_data = [[[100, 150, 200] for _ in range(10)] for _ in range(10)]
        
        response = self.client.post(
            "/api/predict/",
            data=json.dumps({
                "image_name": "test_image.jpg",
                "image_data": pixel_data,
                "image_width": 10,
                "image_height": 10
            }),
            content_type="application/json"
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction_result", response.data)
        self.assertIn("confidence", response.data)
        self.assertIn("image_name", response.data)
