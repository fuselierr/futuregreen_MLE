from django.db import models

# No models needed - predictions are not tracked in database
# Feedback thing 
class FeedbackLimit(models.Model):
    date = models.DateField(unique=True)
    count = models.PositiveIntegerField(default=0)

    def __clash_protected_str__(self):
        return f"{self.date}: {self.count} submissions"