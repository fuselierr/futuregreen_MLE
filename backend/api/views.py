from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response 

# Create your views here.

# process_image
# Endpoint type: POST
# Description: Receives an image, processes it using the DS model, and returns the results.

# upload_image
# Endpoint type: POST
# Description: Allows users to upload images to the server for processing.

# get_result
# Endpoint type: GET
# Description: Retrieves the most likely waste type and its confidence score based on the processed image.

# get_result_matrix
# Endpoint type: GET
# Description: Provides a confusion matrix or similar performance metrics for the DS model.

# submit_feedback
# Endpoint type: POST
# Description: Allows users to submit feedback on the accuracy of the model's predictions, which can be used for further training and improvement.
# ask frontend for more clarification about this

# Additional features (waiting on frontend). You can message Alan about this