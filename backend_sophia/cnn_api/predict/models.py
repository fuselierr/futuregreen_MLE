from django.db import models

class Feedback(models.Model):
    rating = models.IntegerField()
    feedback = models.TextField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Rating: {self.rating}, Feedback: {self.feedback[:20]}..."