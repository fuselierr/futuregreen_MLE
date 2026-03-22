from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r"predict", views.PredictionViewSet, basename="predict")

urlpatterns = [
    path("", include(router.urls)),
    path('update_server/', views.webhook, name='webhook'),
]