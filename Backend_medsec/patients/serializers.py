from rest_framework import serializers
from .models import Patient, ProcessedImage, CroppedRegion
from django.urls import reverse
import base64

class CroppedRegionSerializer(serializers.ModelSerializer):
    class Meta:
        model = CroppedRegion
        fields = ['id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2']

class ProcessedImageSerializer(serializers.ModelSerializer):
    cropped_regions = CroppedRegionSerializer(many=True, read_only=True)
    
    class Meta:
        model = ProcessedImage
        fields = ['id', 'blurred_image', 'grid_image', 'restored_image', 'cropped_regions', 'created_at']

class PatientSerializer(serializers.ModelSerializer):
    # Instead of showing all processed images, we'll add the most recent one in the view
    # processed_image will be added manually in the view response
    
    class Meta:
        model = Patient
        fields = ['id', 'name', 'age', 'note', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']
        
    def validate_age(self, value):
        if value < 0 or value > 120:
            raise serializers.ValidationError("Age must be between 0 and 120 years")
        return value 