from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import PatientViewSet, ProcessedImageViewSet, PatientImageView

router = DefaultRouter()
router.register('', PatientViewSet)
router.register('processed-images', ProcessedImageViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('image/<str:patient_id>/', PatientImageView.as_view(), name='patient-image'),
] 