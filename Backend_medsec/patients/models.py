from django.db import models
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
import os
from django.core.files.base import ContentFile
from django.utils.safestring import mark_safe
from .utils import decrypt_image, load_encryption_keys_from_file
import base64
import logging
from django.core.cache import cache

# Create your models here.

class EncryptedImageField(models.BinaryField):
    """
    A custom field that stores images as encrypted binary data
    """
    description = "Encrypted image field"
    
    def __init__(self, *args, **kwargs):
        # Set a high max_length to accommodate encrypted images
        kwargs['max_length'] = 10485760  # 10MB
        super().__init__(*args, **kwargs)

class Patient(models.Model):
    id = models.CharField(primary_key=True, max_length=50, verbose_name="Patient ID")
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    note = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.id})"
    
    class Meta:
        ordering = ['-created_at']

class ProcessedImage(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='processed_images')
    blurred_image = models.ImageField(upload_to='patient_images/blurred/')
    grid_image = models.ImageField(upload_to='patient_images/grid/')
    restored_image = models.ImageField(upload_to='patient_images/restored/', blank=True, null=True)
    enhanced = models.BooleanField(default=False, help_text="Indicates if the restored image is enhanced")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    original_entropy = models.FloatField(null=True, blank=True)
    encrypted_entropy = models.FloatField(null=True, blank=True)
    encryption_time = models.FloatField(null=True, blank=True, help_text="Encryption time in milliseconds")
    decryption_time = models.FloatField(null=True, blank=True, help_text="Decryption time in milliseconds")
    
    def __str__(self):
        return f"Processed Image for {self.patient.name} ({self.created_at})"

class ImageFingerprint(models.Model):
    """
    Stores a simplified fingerprint of the original image for similarity comparison
    """
    processed_image = models.OneToOneField(ProcessedImage, on_delete=models.CASCADE, related_name='fingerprint')
    color_histogram = models.BinaryField(help_text="Serialized color histogram data")
    avg_hash = models.CharField(max_length=64, help_text="Average hash of the image")
    phash = models.CharField(max_length=64, help_text="Perceptual hash of the image")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Fingerprint for {self.processed_image}"

class CroppedRegion(models.Model):
    """
    Model to store cropped sensitive regions from processed images.
    These regions are encrypted for security.
    """
    processed_image = models.ForeignKey(ProcessedImage, on_delete=models.CASCADE, related_name='cropped_regions')
    class_name = models.CharField(max_length=50)
    confidence = models.FloatField()
    x1 = models.IntegerField()
    y1 = models.IntegerField()
    x2 = models.IntegerField()
    y2 = models.IntegerField()
    cropped_image_data = models.BinaryField()
    original_filename = models.CharField(max_length=255)
    image_format = models.CharField(max_length=10, default='JPEG')
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Cache for decrypted images to avoid repeated decryption
    _decrypted_cache = {}
    
    def get_decrypted_image(self, user=None, use_static_key=False):
        """
        Decrypt and return the cropped image data.
        
        This method has been simplified to always use the static key for decryption.
        User parameter is kept for backward compatibility but is ignored.
        
        Args:
            user: Optional user object (ignored, using static key only)
            use_static_key: Deprecated parameter, kept for backward compatibility
        
        Returns:
            bytes: Decrypted image data
        """
        from patients.utils import decrypt_image
        import logging
        import time
        
        # Set up logging
        logger = logging.getLogger(__name__)
        
        # Use a single cache key format for all decryptions
        cache_key = f"decrypted_region_{self.id}"
        
        logger.info(f"Attempting to decrypt region {self.id} for class {self.class_name}")
        
        # First check local cache (in-memory)
        if cache_key in self._decrypted_cache:
            logger.info(f"Using in-memory cached decrypted data for region {self.id}")
            return self._decrypted_cache[cache_key]
        
        # Then check Django's cache (persistent between restarts)
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.info(f"Using persistent cached decrypted data for region {self.id}")
            # Store in local cache for faster future access
            self._decrypted_cache[cache_key] = cached_data
            return cached_data
        
        try:
            if not self.cropped_image_data:
                logger.error(f"No encrypted data available for region {self.id}")
                return None
                
            # Get the encrypted data size for logging
            encrypted_size = len(self.cropped_image_data)
            logger.info(f"Decrypting region {self.id}, encrypted size: {encrypted_size} bytes")
            
            # Simple decryption using the static key
            start_time = time.time()
            decrypted_data, decryption_time_ms = decrypt_image(self.cropped_image_data)
            
            # Final check if decryption was successful
            if decrypted_data is None:
                logger.error(f"Decryption failed for region {self.id}")
                return None
                
            logger.info(f"Successfully decrypted region {self.id}, decrypted size: {len(decrypted_data)} bytes in {decryption_time_ms:.2f}ms")
            
            # Store decryption time in the parent ProcessedImage
            processed_image = self.processed_image
            if not processed_image.decryption_time:
                processed_image.decryption_time = decryption_time_ms
            else:
                # Average with existing time using a weighted approach
                # This gives more weight to recent decryptions while maintaining history
                processed_image.decryption_time = (processed_image.decryption_time * 0.7) + (decryption_time_ms * 0.3)
                
            # Save the updated decryption time
            processed_image.save(update_fields=['decryption_time'])
            
            # Store in both caches - local and Django's cache
            self._decrypted_cache[cache_key] = decrypted_data
            
            # Store in Django's cache with 1 hour timeout
            cache.set(cache_key, decrypted_data, 3600) 
            
            # Clean local cache if it gets too large (limit to 50 entries)
            if len(self._decrypted_cache) > 50:
                # Remove oldest items
                keys = list(self._decrypted_cache.keys())
                for key in keys[:10]:  # Remove 10 oldest
                    del self._decrypted_cache[key]
            
            return decrypted_data
        except Exception as e:
            logger.error(f"Error decrypting image for region {self.id}: {str(e)}")
            # Log stack trace for debugging
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def __str__(self):
        return f"Cropped Region {self.id} from {self.processed_image.patient.name if self.processed_image.patient else 'Unknown'}"

# Signal to delete files when models are deleted
@receiver(post_delete, sender=ProcessedImage)
def delete_processed_images(sender, instance, **kwargs):
    """Delete image files when ProcessedImage is deleted"""
    try:
        if instance.blurred_image:
            if os.path.isfile(instance.blurred_image.path):
                os.remove(instance.blurred_image.path)
        if instance.grid_image:
            if os.path.isfile(instance.grid_image.path):
                os.remove(instance.grid_image.path)
        if instance.restored_image:
            if os.path.isfile(instance.restored_image.path):
                os.remove(instance.restored_image.path)
    except Exception as e:
        # Log but don't crash if file deletion fails
        print(f"Error deleting processed images: {e}")
