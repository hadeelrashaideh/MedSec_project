from django.contrib import admin
from .models import Patient, ProcessedImage, CroppedRegion, ImageFingerprint
from django.utils.html import format_html
import base64
import binascii
from django.core.cache import cache
import logging
import os
import json
from django.contrib.auth import get_user_model
from .utils import load_encryption_keys_from_file, save_encryption_key

logger = logging.getLogger(__name__)

class CroppedRegionInline(admin.TabularInline):
    model = CroppedRegion
    extra = 0
    readonly_fields = ['encrypted_data_preview', 'class_name', 'confidence', 'coordinates']
    fields = ['encrypted_data_preview', 'class_name', 'confidence', 'coordinates']
    
    def coordinates(self, obj):
        return format_html('({}, {}) to ({}, {})', obj.x1, obj.y1, obj.x2, obj.y2)
    coordinates.short_description = 'Coordinates'
    
    def encrypted_data_preview(self, obj):
        if not obj.cropped_image_data:
            return "No encrypted data"
        
        # Show encrypted data visualization
        data_len = len(obj.cropped_image_data)
        # Get first 32 bytes for preview (including IV)
        preview_hex = binascii.hexlify(obj.cropped_image_data[:32]).decode('ascii')
        formatted_hex = ' '.join([preview_hex[i:i+2] for i in range(0, len(preview_hex), 2)])
        
        # Apply formatting with CSS to show it's encrypted data
        return format_html(
            '<div style="font-family: monospace; background-color: #f0f0f0; padding: 10px; border: 1px solid #ddd;">'
            '<p><strong>Encrypted Data ({} bytes)</strong></p>'
            '<p>IV + Beginning of encrypted data:</p>'
            '<p style="color: #0066cc;">{} ...</p>'
            '<p style="color: #cc0000; font-style: italic;">* Data is AES-256 encrypted</p>'
            '</div>',
            data_len, formatted_hex
        )
    encrypted_data_preview.short_description = 'Encrypted Data'
    
    def has_add_permission(self, request, obj=None):
        return False

class ProcessedImageInline(admin.StackedInline):
    model = ProcessedImage
    extra = 0
    readonly_fields = ['blurred_preview', 'grid_preview']
    fields = ['blurred_preview', 'grid_preview']
    
    def blurred_preview(self, obj):
        if obj.blurred_image:
            return format_html('<img src="{}" width="300" height="auto" />', obj.blurred_image.url)
        return "No image"
    blurred_preview.short_description = 'Blurred Image'
    
    def grid_preview(self, obj):
        if obj.grid_image:
            return format_html('<img src="{}" width="300" height="auto" />', obj.grid_image.url)
        return "No image"
    grid_preview.short_description = 'Grid View'
    
    def has_add_permission(self, request, obj=None):
        return False

@admin.register(ProcessedImage)
class ProcessedImageAdmin(admin.ModelAdmin):
    list_display = ['patient', 'created_at', 'image_previews']
    list_filter = ['created_at', 'patient']
    readonly_fields = ['patient', 'blurred_image', 'grid_image', 'restored_image', 'enhanced', 
                      'created_at', 'updated_at', 'original_entropy', 'encrypted_entropy',
                      'encryption_time', 'decryption_time', 'blurred_preview', 'grid_preview', 
                      'fingerprint_encryption_info', 'similarity_metrics']
    inlines = [CroppedRegionInline]
    
    def has_add_permission(self, request):
        return False
        
    def has_change_permission(self, request, obj=None):
        return False
        
    def has_delete_permission(self, request, obj=None):
        return False
    
    def image_previews(self, obj):
        if obj.grid_image:
            return format_html('<img src="{}" width="200" height="auto" />', obj.grid_image.url)
        return "No image"
    image_previews.short_description = 'Image Overview'
    
    def blurred_preview(self, obj):
        if obj.blurred_image:
            return format_html('<img src="{}" width="500" height="auto" />', obj.blurred_image.url)
        return "No image"
    blurred_preview.short_description = 'Blurred Image'
    
    def grid_preview(self, obj):
        if obj.grid_image:
            return format_html('<img src="{}" width="500" height="auto" />', obj.grid_image.url)
        return "No image"
    grid_preview.short_description = 'Grid View'
    
    def fingerprint_encryption_info(self, obj):
        """Display fingerprint visualization and encryption info"""
        try:
            # Get fingerprint if available
            has_fingerprint = hasattr(obj, 'fingerprint')
            
            # Get encryption data from regions
            regions = obj.cropped_regions.all()
            num_regions = regions.count()
            encrypted_samples = []
            
            # Collect sample encryption data (up to 2 regions)
            if num_regions > 0:
                for region in regions[:2]:
                    if region.cropped_image_data:
                        import binascii
                        data_len = len(region.cropped_image_data)
                        # Get first 16 bytes (IV) and sample of the actual encrypted data
                        iv_hex = binascii.hexlify(region.cropped_image_data[:16]).decode('ascii')
                        sample_hex = binascii.hexlify(region.cropped_image_data[16:48]).decode('ascii')
                        encrypted_samples.append({
                            'class': region.class_name,
                            'size': data_len,
                            'iv': ' '.join([iv_hex[i:i+2] for i in range(0, len(iv_hex), 2)]),
                            'data': ' '.join([sample_hex[i:i+2] for i in range(0, len(sample_hex), 2)])
                        })
            
            # Build the HTML
            html = """
            <div style="max-width: 800px; margin: 15px 0;">
                <h3 style="border-bottom: 1px solid #ddd; padding-bottom: 8px; color: #333;">Fingerprint & Encryption Details</h3>
                
                <div style="display: flex; flex-wrap: wrap;">
                    <!-- Fingerprint Column -->
                    <div style="flex: 1; min-width: 300px; padding-right: 20px;">
                        <div style="background-color: #f9f9f9; border-radius: 5px; border: 1px solid #e0e0e0; padding: 15px; margin-bottom: 15px;">
                            <h4 style="margin-top: 0; color: #333;">Image Fingerprint</h4>
            """
            
            # Add fingerprint visualization if available
            if has_fingerprint:
                # Get hash values
                avg_hash = obj.fingerprint.avg_hash
                phash = obj.fingerprint.phash
                
                # For visualization, convert hex hash to a binary grid
                def hash_to_grid(hash_str):
                    # Convert hex to binary
                    bin_str = ""
                    for c in hash_str:
                        bin_c = bin(int(c, 16))[2:].zfill(4)
                        bin_str += bin_c
                    return bin_str
                
                # Generate binary grids for visualization
                avg_hash_bin = hash_to_grid(avg_hash[:16])  # Use first 16 chars = 64 bits
                phash_bin = hash_to_grid(phash[:16])  # Use first 16 chars = 64 bits
                
                # Create 8x8 grid visualization for avg_hash
                avg_grid = '<div style="display: grid; grid-template-columns: repeat(8, 1fr); gap: 1px; width: 120px; margin-bottom: 10px;">'
                for i in range(0, 64, 1):
                    bit = avg_hash_bin[i] if i < len(avg_hash_bin) else '0'
                    color = "#000" if bit == '1' else "#fff"
                    avg_grid += f'<div style="background-color: {color}; width: 100%; padding-bottom: 100%;"></div>'
                avg_grid += '</div>'
                
                # Create 8x8 grid visualization for phash
                phash_grid = '<div style="display: grid; grid-template-columns: repeat(8, 1fr); gap: 1px; width: 120px; margin-bottom: 10px;">'
                for i in range(0, 64, 1):
                    bit = phash_bin[i] if i < len(phash_bin) else '0'
                    color = "#000" if bit == '1' else "#fff"
                    phash_grid += f'<div style="background-color: {color}; width: 100%; padding-bottom: 100%;"></div>'
                phash_grid += '</div>'
                
                html += f"""
                            <div style="display: flex; margin-bottom: 15px;">
                                <div style="margin-right: 20px;">
                                    <h5 style="margin-top: 0; margin-bottom: 5px; font-size: 14px; color: #555;">Average Hash</h5>
                                    <div style="background-color: white; border: 1px solid #ddd; padding: 10px; border-radius: 4px;">
                                        {avg_grid}
                                    </div>
                                </div>
                                <div>
                                    <h5 style="margin-top: 0; margin-bottom: 5px; font-size: 14px; color: #555;">Perceptual Hash</h5>
                                    <div style="background-color: white; border: 1px solid #ddd; padding: 10px; border-radius: 4px;">
                                        {phash_grid}
                                    </div>
                                </div>
                            </div>
                            
                            <div>
                                <h5 style="margin-top: 0; margin-bottom: 5px; font-size: 14px; color: #555;">Hash Values</h5>
                                <div style="font-family: monospace; font-size: 12px; background-color: white; padding: 10px; border-radius: 4px; border: 1px solid #ddd;">
                                    <div style="margin-bottom: 5px;"><span style="color: #666;">avg_hash:</span> <span style="color: #0066cc;">{avg_hash}</span></div>
                                    <div><span style="color: #666;">phash:</span> <span style="color: #0066cc;">{phash}</span></div>
                                </div>
                            </div>
                """
            else:
                html += """
                            <div style="padding: 20px; text-align: center; color: #777; background-color: #f0f0f0; border-radius: 4px;">
                                No fingerprint data available for this image
                            </div>
                """
                
            html += """
                        </div>
                    </div>
                    
                    <!-- Encryption Column -->
                    <div style="flex: 1; min-width: 300px;">
                        <div style="background-color: #f9f9f9; border-radius: 5px; border: 1px solid #e0e0e0; padding: 15px; margin-bottom: 15px;">
                            <h4 style="margin-top: 0; color: #333;">Encryption Details</h4>
            """
            
            # Add encryption info
            if encrypted_samples:
                # Add visualization of AES encryption on actual cropped regions
                if num_regions > 0:
                    html += """
                            <div style="margin-bottom: 20px;">
                                <h5 style="margin-top: 0; margin-bottom: 8px; font-size: 14px; color: #555;">AES Encryption of Sensitive Regions</h5>
                                <div style="display: flex; flex-direction: column; gap: 15px;">
                    """
                    
                    # Process up to 2 regions to show encryption
                    for region in regions[:2]:
                        if region.cropped_image_data:
                            # Extract some actual bytes to create a realistic visualization
                            import binascii
                            
                            # Get encrypted data for visualization
                            encrypted_bytes = region.cropped_image_data
                            byte_sample = encrypted_bytes[16:48]  # Skip IV, get sample of actual encrypted data
                            hex_bytes = binascii.hexlify(byte_sample).decode('ascii')
                            
                            # Create an array of pixel values from the hex bytes
                            pixels = []
                            for i in range(0, len(hex_bytes), 2):
                                if i+1 < len(hex_bytes):
                                    pixel_val = int(hex_bytes[i:i+2], 16)
                                    pixels.append(pixel_val)
                            
                            html += f"""
                                    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
                                        <div style="font-weight: bold; margin-bottom: 8px; display: flex; justify-content: space-between;">
                                            <span>Region: {region.class_name}</span>
                                            <span style="color: #777; font-size: 12px;">Confidence: {region.confidence:.2f}</span>
                                        </div>
                                        <div style="display: flex; align-items: center; gap: 15px; overflow-x: auto; padding-bottom: 10px;">
                                            <!-- Cropped region -->
                                            <div style="text-align: center; min-width: 160px;">
                                                <div style="width: 160px; height: 160px; border: 1px solid #ddd; background-color: white; display: flex; align-items: center; justify-content: center; position: relative;">
                            """
                            
                            # Now get and show the actual cropped region instead of the full image with overlay
                            decrypted_data = None
                            try:
                                decrypted_data = region.get_decrypted_image(user=getattr(self, 'request', None) and self.request.user)
                            except:
                                pass
                                
                            if decrypted_data:
                                # Convert bytes to base64 for display
                                import base64
                                b64_img = base64.b64encode(decrypted_data).decode('utf-8')
                                img_format = region.image_format.lower() if region.image_format else 'jpeg'
                                img_src = f"data:image/{img_format};base64,{b64_img}"
                                html += f"""
                                                    <img src="{img_src}" style="max-width: 100%; max-height: 100%; object-fit: contain;" />
                                """
                            else:
                                html += """
                                                    <div style="color: #777; font-style: italic;">Cannot display region</div>
                                """
                            
                            html += """
                                                </div>
                                                <div style="font-size: 11px; margin-top: 5px;">Sensitive Region</div>
                                            </div>
                                            
                                            <!-- Arrow -->
                                            <div style="display: flex; align-items: center;">
                                                <div style="width: 30px; height: 2px; background-color: #888; position: relative;">
                                                    <div style="width: 0; height: 0; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-left: 8px solid #888; position: absolute; right: -8px; top: -4px;"></div>
                                                </div>
                                            </div>
                                            
                                            <!-- AES Encrypted Region -->
                                            <div style="text-align: center; min-width: 160px;">
                                                <div style="width: 160px; height: 160px; border: 1px solid #ddd; background-color: black; position: relative; overflow: hidden;">
                            """
                            
                            # Create a realistic visualization of encrypted data
                            # First add a static/noise background
                            num_pixels = 2000
                            import random
                            random.seed(f"{region.id}_{region.class_name}")  # Stable visualization for each region
                            
                            for i in range(num_pixels):
                                x = random.randint(0, 160)
                                y = random.randint(0, 160)
                                size = random.randint(1, 3)
                                
                                # Use actual encrypted bytes for coloring when possible
                                if pixels and i < len(pixels):
                                    val = pixels[i]
                                    r = (val * 17) % 255
                                    g = (val * 23) % 255
                                    b = (val * 31) % 255
                                    opacity = 0.7 + (val % 30) / 100
                                else:
                                    r = random.randint(0, 255)
                                    g = random.randint(0, 255)
                                    b = random.randint(0, 255)
                                    opacity = random.uniform(0.5, 0.9)
                                
                                html += f"""
                                                    <div style="position: absolute; left: {x}px; top: {y}px; width: {size}px; height: {size}px; background-color: rgba({r},{g},{b},{opacity});"></div>
                                """
                            
                            # Add occasional binary "bits" overlay to simulate encryption
                            for i in range(40):
                                x = random.randint(10, 150)
                                y = random.randint(10, 150)
                                bit = random.choice(["0", "1"])
                                opacity = random.uniform(0.3, 0.7)
                                
                                html += f"""
                                                    <div style="position: absolute; left: {x}px; top: {y}px; font-family: monospace; font-size: 8px; color: rgba(0, 255, 0, {opacity});">{bit}</div>
                                """
                            
                            # Add hexadecimal representation of first few bytes overlaid
                            rows = 3
                            cols = 4
                            hex_display = ""
                            for r in range(rows):
                                for c in range(cols):
                                    idx = r * cols + c
                                    if idx < len(hex_bytes) // 2:
                                        byte_idx = idx * 2
                                        hex_val = hex_bytes[byte_idx:byte_idx+2]
                                        hex_display += f"""
                                                    <div style="position: absolute; left: {30 + c * 30}px; top: {60 + r * 14}px; font-family: monospace; font-size: 10px; color: rgba(0, 255, 0, 0.6);">{hex_val}</div>
                                        """
                            
                            html += hex_display
                            
                            html += """
                                                </div>
                                                <div style="font-size: 11px; margin-top: 5px;">AES-256-CBC Encrypted</div>
                                            </div>
                                            
                                            <!-- Arrow -->
                                            <div style="display: flex; align-items: center;">
                                                <div style="width: 30px; height: 2px; background-color: #888; position: relative;">
                                                    <div style="width: 0; height: 0; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-left: 8px solid #888; position: absolute; right: -8px; top: -4px;"></div>
                                                </div>
                                            </div>
                                            
                                            <!-- Binary/Hex representation -->
                                            <div style="text-align: center; min-width: 160px; max-width: 200px;">
                                                <div style="width: 160px; height: 160px; border: 1px solid #ddd; background-color: #121212; position: relative; overflow: auto; padding: 5px; font-family: monospace; font-size: 10px; color: #00ff00; text-align: left;">
                            """
                            
                            # Show binary and hex representation of actual bytes
                            binary_html = ""
                            hex_rows = []
                            byte_count = min(64, len(encrypted_bytes))
                            
                            for i in range(0, byte_count, 8):
                                end = min(i + 8, byte_count)
                                row_bytes = encrypted_bytes[i:end]
                                hex_row = binascii.hexlify(row_bytes).decode('ascii')
                                hex_formatted = ' '.join([hex_row[j:j+2] for j in range(0, len(hex_row), 2)])
                                hex_rows.append(hex_formatted)
                                
                            binary_html += """<div style="margin-bottom: 8px; font-weight: bold;">Encrypted Data:</div>"""
                            
                            for i, hex_row in enumerate(hex_rows):
                                offset_style = "color: #888" if i == 0 else ""
                                row_style = "color: #ff0000" if i == 0 else "" # IV in red
                                binary_html += f"""<div style="margin-bottom: 2px;"><span style="{offset_style}">{i*8:04x}</span>: <span style="{row_style}">{hex_row}</span></div>"""
                            
                            binary_html += """<div style="margin-top: 8px; color: #888; font-size: 9px;">First 16 bytes: Initialization Vector (IV)<br>Remaining: Encrypted region data</div>"""
                            
                            html += binary_html
                            
                            html += """
                                                </div>
                                                <div style="font-size: 11px; margin-top: 5px;">Binary Representation</div>
                                            </div>
                                        </div>
                                    </div>
                            """
                    
                    html += """
                                </div>
                                <div style="font-size: 10px; margin-top: 10px; text-align: center;">
                                    <p>AES-256-CBC encryption with random IVs ensures that even identical content produces completely different encrypted output</p>
                                </div>
                            </div>
                    """
                
                html += """
                            <div style="margin-bottom: 15px;">
                                <h5 style="margin-top: 0; margin-bottom: 8px; font-size: 14px; color: #555;">Encryption Specifications</h5>
                                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                                    <tr style="background-color: #f0f0f0;">
                                        <th style="text-align: left; padding: 5px; border: 1px solid #ddd;">Parameter</th>
                                        <th style="text-align: left; padding: 5px; border: 1px solid #ddd;">Value</th>
                                    </tr>
                                    <tr>
                                        <td style="padding: 5px; border: 1px solid #ddd;">Algorithm</td>
                                        <td style="padding: 5px; border: 1px solid #ddd;">AES-256-CBC</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 5px; border: 1px solid #ddd;">Key Size</td>
                                        <td style="padding: 5px; border: 1px solid #ddd;">256 bits</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 5px; border: 1px solid #ddd;">IV Size</td>
                                        <td style="padding: 5px; border: 1px solid #ddd;">128 bits (16 bytes)</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 5px; border: 1px solid #ddd;">Padding</td>
                                        <td style="padding: 5px; border: 1px solid #ddd;">PKCS#7</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 5px; border: 1px solid #ddd;">Total Encrypted Regions</td>
                                        <td style="text-align: right; padding: 5px; border: 1px solid #ddd;">{}</td>
                                    </tr>
                                </table>
                            </div>
                """.format(num_regions)
                
                # Add sample encrypted data (first two regions)
                html += """
                            <h5 style="margin-top: 15px; margin-bottom: 8px; font-size: 14px; color: #555;">Encrypted Region Samples</h5>
                """
                
                for sample in encrypted_samples:
                    html += f"""
                            <div style="margin-bottom: 15px; background-color: white; border: 1px solid #ddd; padding: 10px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span style="font-weight: bold; color: #333;">{sample['class']}</span>
                                    <span style="color: #777; font-size: 12px;">{sample['size']} bytes</span>
                                </div>
                                <div style="font-family: monospace; font-size: 11px; margin-bottom: 5px;">
                                    <div style="color: #666; margin-bottom: 3px;">IV (16 bytes):</div>
                                    <div style="background-color: #f8f8ff; padding: 5px; color: #cc0000; overflow-wrap: break-word;">{sample['iv']}</div>
                                </div>
                                <div style="font-family: monospace; font-size: 11px;">
                                    <div style="color: #666; margin-bottom: 3px;">Data (sample):</div>
                                    <div style="background-color: #f8f8ff; padding: 5px; color: #0066cc; overflow-wrap: break-word;">{sample['data']}...</div>
                                </div>
                            </div>
                    """
            else:
                html += """
                            <div style="padding: 20px; text-align: center; color: #777; background-color: #f0f0f0; border-radius: 4px;">
                                No encrypted data available for this image
                            </div>
                """
            
            html += """
                        </div>
                    </div>
                </div>
            </div>
            """
            
            return format_html(html)
            
        except Exception as e:
            return f"Error displaying fingerprint and encryption info: {str(e)}"
    
    fingerprint_encryption_info.short_description = 'Fingerprint & Encryption'
    
    def similarity_metrics(self, obj):
        """Calculate and display similarity, entropy and confidence"""
        try:
            import cv2
            import numpy as np
            from scipy.stats import entropy
            from .utils import restore_from_cropped, calculate_original_image_entropy
            
            # Get regions and confidence
            regions = obj.cropped_regions.all()
            num_regions = regions.count()
            
            if num_regions > 0:
                avg_confidence = sum(region.confidence for region in regions) / num_regions
                conf_color = "#28a745" if avg_confidence > 0.7 else "#ffc107" if avg_confidence > 0.5 else "#dc3545"
            else:
                avg_confidence = 0
                conf_color = "#dc3545"
            
            # Get stored entropy values using the new calculation function
            orig_entropy_data = calculate_original_image_entropy(obj.id)
            orig_entropy = orig_entropy_data["scaled_1_8"]
            orig_raw_entropy = orig_entropy_data["raw"]
            entropy_source = orig_entropy_data.get("source", "unknown")
            
            # Get encrypted entropy (or use default if not available)
            encrypted_entropy = obj.encrypted_entropy if obj.encrypted_entropy is not None else 7.5
            
            # Set colors based on entropy ranges
            orig_entropy_color = "#28a745" if orig_raw_entropy < 5.0 else "#ffc107" if orig_raw_entropy < 6.0 else "#dc3545"
            encrypted_entropy_color = "#28a745" if encrypted_entropy < 4 else "#ffc107" if encrypted_entropy < 6 else "#dc3545"
            
            # Get similarity data from restored image
            try:
                # Only calculate if fingerprint exists
                if hasattr(obj, 'fingerprint'):
                    # Force fresh calculation - Pass the current admin user for proper decryption
                    user = getattr(self, 'request', None) and self.request.user
                    if not user:
                        logger.warning("No user available for decryption in admin panel")
                    
                    restored_img, _, similarity_data = restore_from_cropped(obj.id, enhance=False, user=user)
                    
                    # Post-restoration metrics
                    similarity = similarity_data.get("similarity", 0)
                    if similarity > 0.99:
                        similarity_color = "#dc3545"  # Red for suspicious perfect scores
                        perfect_warning = ''
                    else:
                        similarity_color = "#28a745" if similarity > 0.85 else "#ffc107" if similarity > 0.7 else "#dc3545"
                        perfect_warning = ""
                    
                    quality = similarity_data.get("quality", "Unknown")
                    
                    # Pre-restoration metrics
                    pre_similarity = similarity_data.get("pre_similarity", 0)
                    pre_similarity_color = "#28a745" if pre_similarity > 0.85 else "#ffc107" if pre_similarity > 0.7 else "#dc3545"
                    
                    # Improvement
                    improvement = similarity_data.get("improvement", 0)
                    improvement_color = "#28a745" if improvement > 0.3 else "#ffc107" if improvement > 0.1 else "#666666"
                else:
                    similarity = 0
                    pre_similarity = 0
                    improvement = 0
                    similarity_color = "#dc3545"
                    pre_similarity_color = "#dc3545"
                    improvement_color = "#666666"
                    quality = "No fingerprint data"
                    perfect_warning = ""
            except Exception as e:
                similarity = 0
                pre_similarity = 0
                improvement = 0
                similarity_color = "#dc3545"
                pre_similarity_color = "#dc3545"
                improvement_color = "#666666"
                quality = f"Error: {str(e)}"
                perfect_warning = ""
            
            html = f"""
            <div style="max-width: 600px; background-color: #f9f9f9; border-radius: 5px; padding: 15px; margin-top: 10px; border: 1px solid #ddd;">
                <h3 style="margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 8px;">
                    Image Analysis
                    <span style="float: right; font-size: 14px; background-color: {similarity_color}; color: white; padding: 3px 8px; border-radius: 4px;">
                        {quality}
                    </span>
                </h3>
                
                <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                    <div style="flex: 2; margin-right: 10px;">
                        <h4 style="margin-top: 0; font-size: 14px; color: #333;">Similarity Comparison</h4>
                        <table style="width: 100%; border-collapse: collapse; margin-top: 5px;">
                            <tr>
                                <td style="width: 35%; padding: 3px; font-size: 12px;">Before Restoration:</td>
                                <td style="width: 65%; padding: 3px;">
                                    <div style="height: 16px; background-color: #e0e0e0; border-radius: 8px; overflow: hidden;">
                                        <div style="height: 100%; width: {pre_similarity * 100}%; background-color: {pre_similarity_color};"></div>
                                    </div>
                                </td>
                                <td style="width: 10%; padding: 3px; font-size: 12px; text-align: right; font-weight: bold;">{pre_similarity:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px; font-size: 12px;">After Restoration:</td>
                                <td style="padding: 3px;">
                                    <div style="height: 16px; background-color: #e0e0e0; border-radius: 8px; overflow: hidden;">
                                        <div style="height: 100%; width: {similarity * 100}%; background-color: {similarity_color};"></div>
                                    </div>
                                </td>
                                <td style="padding: 3px; font-size: 12px; text-align: right; font-weight: bold;">{similarity:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px; font-size: 12px;">Improvement:</td>
                                <td style="padding: 3px;">
                                    <div style="height: 16px; background-color: #e0e0e0; border-radius: 8px; overflow: hidden;">
                                        <div style="height: 100%; width: {improvement * 100 * 2}%; background-color: {improvement_color};"></div>
                                    </div>
                                </td>
                                <td style="padding: 3px; font-size: 12px; text-align: right; font-weight: bold;">+{improvement:.2f}</td>
                            </tr>
                        </table>
                        {perfect_warning}
                    </div>
                    
                    <div style="flex: 1; margin-right: 10px;">
                        <h4 style="margin-top: 0; font-size: 14px; color: #333;">YOLO Confidence</h4>
                        <div style="height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; margin-bottom: 5px;">
                            <div style="height: 100%; width: {avg_confidence * 100}%; background-color: {conf_color};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 12px;">
                            <span>Low</span>
                            <span style="font-weight: bold;">{avg_confidence:.2f}</span>
                            <span>High</span>
                        </div>
                    </div>
                    
                    <div style="flex: 1;">
                        <h4 style="margin-top: 0; font-size: 14px; color: #333;">Entropy (1-8 Scale)</h4>
                        
                        <div style="margin-bottom: 8px;">
                            <div style="font-size: 11px; margin-bottom: 2px;">Original Image:</div>
                            <div style="height: 16px; background-color: #e0e0e0; border-radius: 8px; overflow: hidden; margin-bottom: 3px;">
                                <div style="height: 100%; width: {((orig_entropy - 1) / 7) * 100}%; background-color: {orig_entropy_color};"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 10px;">
                                <span>1</span>
                                <span style="font-weight: bold;">{orig_entropy:.2f} (scaled) / {orig_raw_entropy:.2f} bits <span style="color: #777; font-size: 11px;">({entropy_source})</span></span>
                                <span>8</span>
                            </div>
                        </div>
                        
                        <div>
                            <div style="font-size: 11px; margin-bottom: 2px;">Encrypted Parts:</div>
                            <div style="height: 16px; background-color: #e0e0e0; border-radius: 8px; overflow: hidden; margin-bottom: 3px;">
                                <div style="height: 100%; width: {((encrypted_entropy - 1) / 7) * 100}%; background-color: {encrypted_entropy_color};"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 10px;">
                                <span>1</span>
                                <span style="font-weight: bold;">{encrypted_entropy:.2f}</span>
                                <span>8</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="background-color: white; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
                    <h4 style="margin-top: 0; font-size: 14px; color: #333;">Detection Summary</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="background-color: #f0f0f0;">
                            <th style="text-align: left; padding: 5px; border: 1px solid #ddd;">Metric</th>
                            <th style="text-align: right; padding: 5px; border: 1px solid #ddd;">Value</th>
                        </tr>
                        <tr>
                            <td style="padding: 5px; border: 1px solid #ddd;">Detected Regions</td>
                            <td style="text-align: right; padding: 5px; border: 1px solid #ddd;">{num_regions}</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px; border: 1px solid #ddd;">Confidence</td>
                            <td style="text-align: right; padding: 5px; border: 1px solid #ddd;">{avg_confidence:.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px; border: 1px solid #ddd;">Original Image Entropy</td>
                            <td style="text-align: right; padding: 5px; border: 1px solid #ddd;">{orig_entropy:.2f} (scaled) / {orig_raw_entropy:.2f} bits <span style="color: #777; font-size: 11px;">({entropy_source})</span></td>
                        </tr>
                        <tr>
                            <td style="padding: 5px; border: 1px solid #ddd;">Encrypted Parts Entropy</td>
                            <td style="text-align: right; padding: 5px; border: 1px solid #ddd;">{encrypted_entropy:.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px; border: 1px solid #ddd;">Pre-Restoration Similarity</td>
                            <td style="text-align: right; padding: 5px; border: 1px solid #ddd;">{pre_similarity:.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px; border: 1px solid #ddd;">Post-Restoration Similarity</td>
                            <td style="text-align: right; padding: 5px; border: 1px solid #ddd;">{similarity:.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px; border: 1px solid #ddd;">Improvement</td>
                            <td style="text-align: right; padding: 5px; border: 1px solid #ddd; color: {improvement_color};">+{improvement:.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px; border: 1px solid #ddd;">Encryption Time</td>
                            <td style="text-align: right; padding: 5px; border: 1px solid #ddd;">{obj.encryption_time or 0:.2f} ms</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px; border: 1px solid #ddd;">Decryption Time</td>
                            <td style="text-align: right; padding: 5px; border: 1px solid #ddd;">{obj.decryption_time or 0:.2f} ms</td>
                        </tr>
                    </table>
                </div>
            </div>
            """
            return format_html(html)
            
        except Exception as e:
            return f"Error calculating metrics: {str(e)}"
    
    similarity_metrics.short_description = 'Image Analysis'

    @admin.action(description="Clear image cache and force recalculation")
    def clear_image_cache(self, request, queryset):
        for processed_image in queryset:
            # Clear all cache entries for this image
            cache_keys = [
                f"restored_image:{processed_image.id}:exact",
                f"decrypted_region_{processed_image.id}"
            ]
            
            # Clear for variations of the key (with timestamps)
            for i in range(300):  # Check a range of potential cache keys
                cache_keys.append(f"restored_image:{processed_image.id}:exact:{i}")
            
            for key in cache_keys:
                cache.delete(key)
                
            # Clear region caches
            for region in processed_image.cropped_regions.all():
                cache_key = f"decrypted_region_{region.id}"
                cache.delete(cache_key)
                
                # Also clear from the model's private cache
                if hasattr(region, '_decrypted_cache') and cache_key in region._decrypted_cache:
                    del region._decrypted_cache[cache_key]
        
        # Removed success message

    @admin.action(description="Recalculate entropy values to 4-7 range")
    def recalculate_entropy(self, request, queryset):
        from .utils import recalculate_image_entropy
        
        total = queryset.count()
        updated = 0
        skipped = 0
        errors = 0
        
        for image in queryset:
            results = recalculate_image_entropy(image.id)
            
            if results.get("status") == "error":
                errors += 1
                continue
                
            updated += results["updated_images"]
            skipped += results["skipped_images"]
            errors += results["errors"]
        
        # Removed success message

    @admin.action(description="Check and import encryption keys from file")
    def check_encryption_keys(self, request, queryset):
        User = get_user_model()
        
        # Count users with DH keys in cache
        users = User.objects.all()
        users_with_keys = 0
        
        for user in users:
            cache_key = f"encryption_key_{user.id}"
            if cache.get(cache_key):
                users_with_keys += 1
        
        # Load keys from file
        file_keys = load_encryption_keys_from_file()
        
        # Check if we need to import keys
        if users_with_keys == 0 and len(file_keys) > 0:
            # Import keys from file to cache
            imported = 0
            
            for user_id_str, key_value in file_keys.items():
                try:
                    user_id = int(user_id_str)
                    cache_key = f"encryption_key_{user_id}"
                    cache.set(cache_key, key_value, timeout=60*60*24*30)  # 30 days
                    imported += 1
                except Exception as e:
                    ""
        elif users_with_keys > 0 and len(file_keys) == 0:
            # Export keys from cache to file
            exported = 0
            
            for user in users:
                cache_key = f"encryption_key_{user.id}"
                key_value = cache.get(cache_key)
                if key_value:
                    if save_encryption_key(user.id, key_value):
                        exported += 1
        elif users_with_keys > 0 and len(file_keys) > 0:
            # Both cache and file have keys, compare and sync
            cache_to_file = 0
            file_to_cache = 0
            
            # First, update file with any new cache keys
            for user in users:
                user_id_str = str(user.id)
                cache_key = f"encryption_key_{user.id}"
                key_value = cache.get(cache_key)
                
                if key_value and (user_id_str not in file_keys or file_keys[user_id_str] != key_value):
                    if save_encryption_key(user.id, key_value):
                        cache_to_file += 1
            
            # Then, update cache with any file keys not in cache
            for user_id_str, key_value in file_keys.items():
                try:
                    user_id = int(user_id_str)
                    cache_key = f"encryption_key_{user_id}"
                    if not cache.get(cache_key):
                        cache.set(cache_key, key_value, timeout=60*60*24*30)  # 30 days
                        file_to_cache += 1
                except Exception as e:
                    ""

    def changelist_view(self, request, extra_context=None):
        """Override changelist view to add encryption key warnings"""
        from django.core.cache import cache
        from django.contrib.auth import get_user_model
        
        # Get encryption key status
        User = get_user_model()
        users = User.objects.all()
        users_with_keys = 0
        
        for user in users:
            cache_key = f"encryption_key_{user.id}"
            if cache.get(cache_key):
                users_with_keys += 1
        
        # Check for file-based keys
        file_keys = load_encryption_keys_from_file()
        file_keys_count = len(file_keys)
        
        return super().changelist_view(request, extra_context)

    actions = [clear_image_cache, recalculate_entropy, check_encryption_keys]

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'age', 'created_at']
    search_fields = ['id', 'name']
    list_filter = ['created_at']
    inlines = [ProcessedImageInline]

@admin.register(CroppedRegion)
class CroppedRegionAdmin(admin.ModelAdmin):
    list_display = ['processed_image', 'class_name', 'confidence', 'coordinates', 'encryption_status']
    list_filter = ['class_name', 'processed_image__patient']
    search_fields = ['class_name', 'processed_image__patient__name']
    readonly_fields = ['encrypted_data_preview', 'coordinates', 'encryption_status', 'encryption_details', 'similarity_analysis']
    
    def coordinates(self, obj):
        return format_html('({}, {}) to ({}, {})', obj.x1, obj.y1, obj.x2, obj.y2)
    coordinates.short_description = 'Coordinates'
    
    def encryption_status(self, obj):
        if obj.cropped_image_data:
            return format_html('<span style="color: green; font-weight: bold;">✓ Encrypted ({} bytes)</span>', 
                              len(obj.cropped_image_data))
        return format_html('<span style="color: red;">Not encrypted</span>')
    encryption_status.short_description = 'Encryption Status'
    
    def similarity_analysis(self, obj):
        """Calculate and display entropy and similarity for this region"""
        try:
            # Get the decrypted image data
            decrypted_data = obj.get_decrypted_image(user=getattr(self, 'request', None) and self.request.user)
            if not decrypted_data:
                return format_html(
                    '<div style="color: #dc3545; padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px;">'
                    '<strong>Decryption Failed:</strong> Cannot calculate similarity. Please ensure you have the correct decryption key.'
                    '</div>'
                )
            
            # Convert to OpenCV format
            import numpy as np
            import cv2
            from scipy.stats import entropy
            from io import BytesIO
            
            # Decode image
            img_array = np.frombuffer(decrypted_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                return format_html(
                    '<div style="color: #dc3545; padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px;">'
                    '<strong>Image Decoding Failed:</strong> The decrypted data could not be decoded as an image.'
                    '</div>'
                )
            
            # Calculate entropy
            img_entropy = entropy(img.flatten())
            
            # Calculate confidence color (higher is better)
            conf_color = "#28a745" if obj.confidence > 0.7 else "#ffc107" if obj.confidence > 0.5 else "#dc3545"
            
            # Calculate entropy color
            entropy_color = "#28a745" if img_entropy < 4 else "#ffc107" if img_entropy < 6 else "#dc3545"
            
            # Create the visualization
            html = f"""
            <div style="max-width: 500px; background-color: #f9f9f9; border-radius: 5px; padding: 15px; margin-top: 10px; border: 1px solid #ddd;">
                <h3 style="margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 8px;">Region Analysis</h3>
                
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <div style="flex: 1; margin-right: 10px;">
                        <h4 style="margin-top: 0; font-size: 14px; color: #333;">YOLO Confidence</h4>
                        <div style="height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
                            <div style="height: 100%; width: {obj.confidence * 100}%; background-color: {conf_color};"></div>
                        </div>
                        <div style="text-align: center; font-weight: bold; color: {conf_color};">{obj.confidence:.2f}</div>
                    </div>
                    
                    <div style="flex: 1;">
                        <h4 style="margin-top: 0; font-size: 14px; color: #333;">Entropy</h4>
                        <div style="height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
                            <div style="height: 100%; width: {min(100, img_entropy * 12.5)}%; background-color: {entropy_color};"></div>
                        </div>
                        <div style="text-align: center; font-weight: bold; color: {entropy_color};">{img_entropy:.2f}</div>
                    </div>
                </div>
                
                <div style="margin-top: 10px; color: #666; font-size: 12px;">
                    <p>
                        <strong>Entropy:</strong> {img_entropy:.4f} (information content of region)<br>
                        <strong>Class:</strong> {obj.class_name} with {obj.confidence:.2f} confidence<br>
                        <strong>Size:</strong> {obj.x2 - obj.x1} × {obj.y2 - obj.y1} pixels
                    </p>
                </div>
            </div>
            """
            return format_html(html)
            
        except Exception as e:
            return f"Error calculating similarity: {str(e)}"
    
    similarity_analysis.short_description = 'Similarity & Entropy Analysis'
    
    def encryption_details(self, obj):
        if not obj.cropped_image_data:
            return "No encrypted data"
        
        # First 16 bytes are the IV
        iv = obj.cropped_image_data[:16]
        iv_hex = binascii.hexlify(iv).decode('ascii')
        formatted_iv = ' '.join([iv_hex[i:i+2] for i in range(0, len(iv_hex), 2)])
        
        # Following bytes are the encrypted data
        data_len = len(obj.cropped_image_data) - 16
        
        # Display sample of encrypted data (first 32 bytes after IV)
        sample_size = min(32, data_len)
        sample = obj.cropped_image_data[16:16+sample_size]
        sample_hex = binascii.hexlify(sample).decode('ascii')
        formatted_sample = ' '.join([sample_hex[i:i+2] for i in range(0, len(sample_hex), 2)])
        
        return format_html(
            '<div style="font-family: monospace; background-color: #f0f0f0; padding: 10px; border: 1px solid #ddd;">'
            '<p><strong>AES-256-CBC Encryption Details</strong></p>'
            '<p>Original filename: <span style="color: #006600;">{}</span></p>'
            '<p>Image format: <span style="color: #006600;">{}</span></p>'
            '<p>Total encrypted size: <span style="color: #006600;">{} bytes</span></p>'
            '<p>Initialization Vector (16 bytes):</p>'
            '<p style="color: #0066cc;">{}</p>'
            '<p>Encrypted data sample ({} bytes of {} total):</p>'
            '<p style="color: #cc6600;">{} ...</p>'
            '</div>',
            obj.original_filename, obj.image_format, len(obj.cropped_image_data),
            formatted_iv, sample_size, data_len, formatted_sample
        )
    encryption_details.short_description = 'Encryption Details'
    
    def encrypted_data_preview(self, obj):
        if not obj.cropped_image_data:
            return "No encrypted data"
        
        # Show encrypted data visualization
        data_len = len(obj.cropped_image_data)
        # Get first 64 bytes for preview (including IV)
        preview_size = min(64, data_len)
        preview_hex = binascii.hexlify(obj.cropped_image_data[:preview_size]).decode('ascii')
        
        # Format in blocks of 16 bytes (32 hex chars) for readability
        formatted_blocks = []
        for i in range(0, len(preview_hex), 32):
            block = preview_hex[i:i+32]
            formatted_block = ' '.join([block[j:j+2] for j in range(0, len(block), 2)])
            formatted_blocks.append(formatted_block)
        
        # Apply formatting with CSS to show it's encrypted data
        return format_html(
            '<div style="font-family: monospace; background-color: #f0f0f0; padding: 10px; border: 1px solid #ddd;">'
            '<p><strong>Encrypted Image Data ({} bytes)</strong></p>'
            '<p>First {} bytes shown:</p>'
            '{}'
            '<p style="color: #cc0000; font-style: italic;">* Data is AES-256-CBC encrypted</p>'
            '</div>',
            data_len, preview_size,
            format_html('<br>'.join(['<span style="color: {};">{}</span>'.format(
                '#0066cc' if i == 0 else '#cc6600', 
                block
            ) for i, block in enumerate(formatted_blocks)]))
        )
    encrypted_data_preview.short_description = 'Encrypted Data'

@admin.register(ImageFingerprint)
class ImageFingerprintAdmin(admin.ModelAdmin):
    list_display = ['processed_image', 'created_at', 'hash_preview']
    list_filter = ['created_at']
    readonly_fields = ['processed_image', 'avg_hash', 'phash', 'color_histogram', 'created_at']
    
    def has_add_permission(self, request):
        return False
    
    def hash_preview(self, obj):
        try:
            import pickle
            import cv2
            import numpy as np
            from django.utils.html import format_html
            
            # Get a preview of the hashes
            avg_hash = obj.avg_hash[:16] + "..." if len(obj.avg_hash) > 16 else obj.avg_hash
            phash = obj.phash[:16] + "..." if len(obj.phash) > 16 else obj.phash
            
            # Try to get histogram data
            hist_data = pickle.loads(obj.color_histogram)
            has_valid_hist = True
        except:
            has_valid_hist = False
        
        if has_valid_hist:
            result = f"""
            <div style="font-family: monospace; background: #f8f9fa; padding: 10px; border-radius: 4px; border: 1px solid #ddd;">
                <div><strong>avg_hash:</strong> {avg_hash}</div>
                <div><strong>phash:</strong> {phash}</div>
                <div><strong>histogram:</strong> Valid data present</div>
            </div>
            """
        else:
            result = f"""
            <div style="font-family: monospace; background: #f8f9fa; padding: 10px; border-radius: 4px; border: 1px solid #ddd;">
                <div><strong>avg_hash:</strong> {avg_hash}</div>
                <div><strong>phash:</strong> {phash}</div>
                <div style="color: #dc3545;"><strong>histogram:</strong> Invalid or missing data</div>
            </div>
            """
        
        return format_html(result)
    
    hash_preview.short_description = 'Fingerprint Preview'
