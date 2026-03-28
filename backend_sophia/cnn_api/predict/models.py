from django.db import models


class UserFeedback(models.Model):
    """Model to store user feedback on model predictions"""
    
    model_prediction = models.CharField(
        max_length=255,
        help_text="The model's predicted type of trash"
    )
    user_prediction = models.CharField(
        max_length=255,
        help_text="The user's predicted type of trash"
    )
    image_data = models.TextField(
        help_text="Base64-encoded image data"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Feedback: Model={self.model_prediction}, User={self.user_prediction}"
    
    class Meta:
        verbose_name = "User Feedback"
        verbose_name_plural = "User Feedbacks"
        ordering = ["-created_at"]


class Feedback(models.Model):
    rating = models.IntegerField()
    feedback = models.TextField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Rating: {self.rating}, Feedback: {self.feedback[:20]}..."
