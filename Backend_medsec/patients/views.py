from django.shortcuts import render, get_object_or_404
from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from django.http import HttpResponse, HttpResponseForbidden, FileResponse
from django.core.files.base import ContentFile
from .models import Patient, ProcessedImage, CroppedRegion
from .serializers import PatientSerializer, ProcessedImageSerializer
from authentication.permissions import IsDoctorUser, IsLabUser
from .utils import process_image, restore_from_cropped
import logging
import cv2
import os
import base64
import numpy as np
from django.conf import settings
from io import BytesIO
from django.urls import reverse
from django.core.cache import cache
import json

# Set up logging
logger = logging.getLogger(__name__)

class PatientViewSet(viewsets.ModelViewSet):
    """
    ViewSet for patient data management:
    - Create patient with single image (POST): Only Lab users
    - View patient details (GET): Only Doctor users
    """
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
    parser_classes = (MultiPartParser, FormParser)
    
    def get_permissions(self):
        """
        Custom permissions:
        - For create (POST): Only Lab users can create patients
        - For retrieve (GET): Only Doctor users can view patient details
        - All other actions are disabled
        """
        if self.action == 'create':
            permission_classes = [permissions.IsAuthenticated & IsLabUser]
        elif self.action == 'retrieve':
            permission_classes = [permissions.IsAuthenticated & IsDoctorUser]
        else:
            # Disable list, update and delete
            permission_classes = [permissions.IsAuthenticated]
        return [permission() for permission in permission_classes]
    
    def list(self, request, *args, **kwargs):
        """
        Disable listing all patients. Only individual patient records can be accessed.
        """
        raise PermissionDenied("Listing all patients is not allowed. Please access individual patient records.")
    
    def update(self, request, *args, **kwargs):
        """Disable update functionality"""
        raise PermissionDenied("Updating patients is not available.")
    
    def partial_update(self, request, *args, **kwargs):
        """Disable partial update functionality"""
        raise PermissionDenied("Updating patients is not available.")
        
    def destroy(self, request, *args, **kwargs):
        """Disable delete functionality"""
        raise PermissionDenied("Deleting patients is not available.")
    
    def create(self, request, *args, **kwargs):
        """
        Create a new patient with a single image processing (Lab users only)
        """
        # Log incoming data (without sensitive info)
        logger.info(f"Creating new patient with data keys: {list(request.data.keys())}")
        
        # Get the uploaded image from request before serializer handling
        # The field should be named 'image' in the form
        uploaded_image = request.FILES.get('image', None)
        
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Save patient first to get ID
        patient = serializer.save()
        logger.info(f"Patient created with ID: {patient.id}")
        
        # Process the single uploaded image if provided
        if uploaded_image:
            try:
                logger.info(f"Processing uploaded image for patient {patient.id}")
                # Pass the current user for DH key-based encryption
                processed_image = process_image(uploaded_image, patient, user=request.user)
                
                if processed_image:
                    logger.info(f"Image processed successfully for patient {patient.id}")
                    
                    # No longer generating and saving restored images at creation time
                    # The restored image will be generated on-demand when a doctor requests it
                    logger.info(f"Restored image will be generated on-demand when requested by a doctor")
                else:
                    logger.warning(f"No image processed for patient {patient.id}")
            except Exception as e:
                logger.error(f"Error processing image for patient {patient.id}: {str(e)}")
                # Continue without failing the request
        else:
            logger.warning(f"No image provided for patient {patient.id}")
        
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
        
    def retrieve(self, request, *args, **kwargs):
        """
        Doctor API endpoint to get patient data with restored image URL.
        
        Process:
        1. Retrieve patient data
        2. Generate a direct URL for the restored image
        3. Return the patient data with the image URL in a single response
        """
        # Check doctor permission
        if not IsDoctorUser().has_permission(request, self):
            raise PermissionDenied("Only Doctor users can view patient details.")
        
        # Get the patient data
        instance = self.get_object()
        
        # Create response with patient fields
        response_data = {
            'id': instance.id,
            'name': instance.name,
            'age': instance.age,
            'note': instance.note
        }
        
        # Get the patient's processed image
        processed_images = ProcessedImage.objects.filter(patient=instance)
        
        if processed_images.exists():
            # Generate direct image URL for this patient
            image_url = request.build_absolute_uri(
                reverse('patient-image', kwargs={'patient_id': instance.id})
            )
            response_data['image_url'] = image_url
        else:
            response_data['error'] = "No processed image found for this patient"
        
        return Response(response_data)

class ProcessedImageViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for viewing processed images
    """
    queryset = ProcessedImage.objects.all()
    permission_classes = [IsAuthenticated]
    serializer_class = ProcessedImageSerializer
    
    def get_queryset(self):
        """
        Filter processed images by patient if patient_id is provided
        """
        queryset = ProcessedImage.objects.all()
        patient_id = self.request.query_params.get('patient_id')
        
        if patient_id:
            queryset = queryset.filter(patient__id=patient_id)
            
        return queryset

class RestoreImageView(APIView):
    """
    API endpoint for restoring original images by merging cropped regions back into blurred images.
    Images are generated on-demand and never saved to disk.
    """
    permission_classes = [IsAuthenticated & IsDoctorUser]
    
    def get(self, request, processed_image_id):
        """
        Restore an image by merging cropped regions into blurred image.
        The restoration happens in memory and is never saved to disk.
        
        Returns the restored image as JPEG or PNG response.
        """
        try:
            # Check if enhancement is requested
            enhance = request.query_params.get('enhance', 'true').lower() == 'true'
            
            # Check if format is specified (default to png)
            img_format = request.query_params.get('format', 'png').lower()
            # Check if PNG quality is specified (0-9, higher is better compression but slower)
            png_compression = int(request.query_params.get('compression', '3'))
            # Check if user wants jpeg and its quality setting
            jpeg_quality = int(request.query_params.get('quality', '95'))
            
            # Get the processed image
            try:
                processed_image = ProcessedImage.objects.get(id=processed_image_id)
            except ProcessedImage.DoesNotExist:
                return Response(
                    {"error": f"Processed image with ID {processed_image_id} not found"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Check if user has permission to access this patient's data
            if IsDoctorUser().has_permission(request, self):
                # Restore the image on-demand
                logger.info(f"Restoring image {processed_image_id} with enhancement={enhance}")
                # Pass the user object for proper decryption
                restored_img, filename = restore_from_cropped(processed_image_id, enhance, user=request.user)
                
                if img_format == 'jpg' or img_format == 'jpeg':
                    # Use high-quality JPEG encoding for users who prefer it
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    content_type = 'image/jpeg'
                    _, img_encoded = cv2.imencode('.jpg', restored_img, encode_param)
                else:
                    # Use PNG for highest quality and lossless encoding
                    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression]
                    content_type = 'image/png'
                    _, img_encoded = cv2.imencode('.png', restored_img, encode_param)
                
                # Return the image without saving it
                response = HttpResponse(img_encoded.tobytes(), content_type=content_type)
                
                # Set appropriate filename based on format
                if img_format == 'jpg' or img_format == 'jpeg':
                    response['Content-Disposition'] = f'inline; filename="{filename.replace(".jpg", ".jpg")}"'
                else:
                    response['Content-Disposition'] = f'inline; filename="{filename.replace(".jpg", ".png")}"'
                
                # Add cache headers for better performance
                response['Cache-Control'] = 'private, max-age=3600'  # Cache for 1 hour on client
                
                logger.info(f"Successfully restored image {processed_image_id} in {img_format} format")
                return response
            else:
                return Response(
                    {"error": "You do not have permission to access this resource"}, 
                    status=status.HTTP_403_FORBIDDEN
                )
                
        except Exception as e:
            logger.error(f"Error restoring image {processed_image_id}: {str(e)}")
            return Response(
                {"error": f"Error restoring image: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class DecryptCroppedImageView(APIView):
    """
    API endpoint for decrypting and serving cropped region images
    """
    permission_classes = [IsAuthenticated & IsDoctorUser]
    
    def get(self, request, cropped_region_id):
        """
        Decrypt and serve a cropped region image
        """
        try:
            # Get the cropped region
            cropped_region = get_object_or_404(CroppedRegion, id=cropped_region_id)
            
            # Check permissions - only allow doctors to access
            if not IsDoctorUser().has_permission(request, self):
                return HttpResponseForbidden("Only doctors can view decrypted images")
            
            # Get decrypted image data - pass the current user for DH key-based decryption
            decrypted_data = cropped_region.get_decrypted_image(user=request.user)
            if decrypted_data is None:
                return HttpResponse("Could not decrypt image", status=500)
            
            # Determine content type based on image format
            content_type = f'image/{cropped_region.image_format.lower()}'
            
            # Return the image as a response
            response = HttpResponse(decrypted_data, content_type=content_type)
            filename = cropped_region.original_filename
            response['Content-Disposition'] = f'inline; filename="{filename}"'
            
            return response
            
        except Exception as e:
            logger.error(f"Error decrypting image {cropped_region_id}: {str(e)}")
            return HttpResponse(f"Error: {str(e)}", status=500)

# Add a new view for serving the image directly
class PatientImageView(APIView):
    """API endpoint to get a specific patient's processed image"""
    permission_classes = [IsAuthenticated & IsDoctorUser]
    
    def get(self, request, patient_id):
        """Get a patient image in its restored form"""
        try:
            # Check if there's a specific processed image ID requested
            processed_image_id = request.query_params.get('processed_image_id')
            
            # Check if we should use E2E encryption (might be in either of these formats)
            use_e2e = request.query_params.get('use_e2e', 'true').lower() == 'true'
            
            # Get the enhancement parameter (default to false for exact preservation)
            enhance = request.query_params.get('enhance', 'false').lower() == 'true'
            
            # Get the format parameter (default to png)
            img_format = request.query_params.get('format', 'png').lower()
            if img_format not in ['png', 'jpg', 'jpeg']:
                img_format = 'png'
            
            # Get the patient
            patient = Patient.objects.get(id=patient_id)
            
            # Get the processed image
            if processed_image_id:
                processed_image = ProcessedImage.objects.get(id=processed_image_id, patient=patient)
            else:
                # Get the most recent processed image
                processed_image = ProcessedImage.objects.filter(patient=patient).order_by('-created_at').first()
                
            if not processed_image:
                return Response({"error": "No processed image found for this patient"}, status=404)
            
            # Always pass the current user for decryption unless explicitly disabled
            # This ensures proper E2E decryption with user-specific keys
            user_for_decryption = request.user if use_e2e else None
            
            # Log the decryption request details
            logger.info(f"Image request - Patient: {patient_id}, User: {request.user.id}, E2E: {use_e2e}")
            
            # Restore the image from blurred + crops
            restored_img, filename, similarity = restore_from_cropped(
                processed_image.id, 
                enhance=enhance,
                user=user_for_decryption
            )
            
            # Generate an appropriate filename - use processed image ID to avoid confusion
            output_filename = f"patient_{patient_id}_image_{processed_image.id}.{img_format}"
            
            # Convert the image to bytes
            success, buffer = cv2.imencode(f'.{img_format}', restored_img)
            if not success:
                return Response({"error": "Failed to encode image"}, status=500)
            
            # Create response with the image data
            response = HttpResponse(
                buffer.tobytes(),
                content_type=f"image/{img_format}"
            )
            
            # Add content disposition header to name the file
            response['Content-Disposition'] = f'inline; filename="{output_filename}"'
            
            # Add similarity metrics in response headers if available
            if similarity:
                # Convert similarity dict to JSON string
                similarity_json = json.dumps(similarity)
                # Add as a custom HTTP header
                response['X-Image-Similarity'] = similarity_json
            
            return response
            
        except Patient.DoesNotExist:
            return Response({"error": "Patient not found"}, status=404)
        except ProcessedImage.DoesNotExist:
            return Response({"error": "Processed image not found"}, status=404) 
        except Exception as e:
            # Log the error for debugging
            logger.error(f"Error in PatientImageView: {str(e)}")
            return Response({"error": str(e)}, status=500)
