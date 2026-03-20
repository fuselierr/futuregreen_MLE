from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import PredictionViewSet, submit_review

router = DefaultRouter()
router.register(r"predict", PredictionViewSet, basename="predict")

urlpatterns = [
    path("", include(router.urls)),
    path("reviews/", submit_review, name="submit_review"),
]
