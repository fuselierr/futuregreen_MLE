from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import PredictionViewSet, submit_review

router = DefaultRouter()
router.register(r"predict", views.PredictionViewSet, basename="predict")

urlpatterns = [
    path("", include(router.urls)),
<<<<<<< HEAD
    path('update_server/', views.webhook, name='webhook'),
]
    path("reviews/", submit_review, name="submit_review"),
]
