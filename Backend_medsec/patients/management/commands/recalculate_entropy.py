from django.core.management.base import BaseCommand
from patients.utils import recalculate_image_entropy, calculate_entropy
from patients.models import ProcessedImage
import time
import numpy as np
import os
import cv2

class Command(BaseCommand):
    help = 'Recalculates entropy values for all images to ensure they properly reflect image differences'

    def add_arguments(self, parser):
        parser.add_argument(
            '--id',
            type=int,
            help='Specific ProcessedImage ID to update',
        )
        
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of images to process in each batch',
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed information about entropy changes',
        )
        
        parser.add_argument(
            '--analyze',
            action='store_true',
            help='Perform detailed analysis of entropy distribution across images',
        )

    def handle(self, *args, **options):
        start_time = time.time()
        
        image_id = options.get('id')
        batch_size = options.get('batch_size')
        verbose = options.get('verbose')
        analyze = options.get('analyze')
        
        self.stdout.write(self.style.SUCCESS(f"Recalculating entropy values with improved normalization"))
        self.stdout.write(f"This version preserves more of the actual differences between images")
        self.stdout.write(f"Original images should have entropy around 6.12-6.36 bits, encrypted regions around 7.30")
        
        if image_id:
            self.stdout.write(self.style.SUCCESS(f'Recalculating entropy for image ID {image_id}...'))
            results = recalculate_image_entropy(image_id)
            
            if results.get("status") == "error":
                self.stdout.write(self.style.ERROR(f'Error: {results["error"]}'))
                return
                
            self.stdout.write(self.style.SUCCESS(f'Updated {results["updated_images"]} images'))
            
            # Print details
            for detail in results["details"]:
                if detail["status"] == "updated":
                    self.stdout.write(f'  Image {detail["id"]}: '
                                     f'Old entropy: {detail["old_entropy"]:.2f}, '
                                     f'New entropy: {detail["new_entropy"]:.2f} '
                                     f'(Raw: {detail["raw_entropy"]:.2f})')
                    self.stdout.write(f'    Randomness: {detail["randomness"]}')
                    self.stdout.write(f'    Unique values: {detail["unique_values"]}/256')
                    
                    # Print region information if available
                    if "regions" in detail and detail["regions"]:
                        self.stdout.write(f'    Regions:')
                        for region in detail["regions"]:
                            self.stdout.write(f'      Region {region["region_id"]}: '
                                             f'Raw entropy: {region["entropy"]:.4f}')
                elif detail["status"] == "error":
                    self.stdout.write(self.style.ERROR(f'  Image {detail["id"]}: Error: {detail["error"]}'))
                else:
                    self.stdout.write(self.style.WARNING(f'  Image {detail["id"]}: Skipped: {detail["reason"]}'))
        else:
            # Process in batches
            total_images = ProcessedImage.objects.count()
            self.stdout.write(self.style.SUCCESS(f'Recalculating entropy for {total_images} images in batches of {batch_size}...'))
            
            # Get all image IDs
            image_ids = list(ProcessedImage.objects.values_list('id', flat=True))
            
            # Process in batches
            total_updated = 0
            total_skipped = 0
            total_errors = 0
            entropy_changes = []
            all_entropies = []
            
            for i in range(0, len(image_ids), batch_size):
                batch_ids = image_ids[i:i+batch_size]
                self.stdout.write(f'Processing batch {i//batch_size + 1}/{(len(image_ids) + batch_size - 1)//batch_size}...')
                
                for img_id in batch_ids:
                    results = recalculate_image_entropy(img_id)
                    
                    if results.get("status") == "error":
                        self.stdout.write(self.style.ERROR(f'Error processing batch: {results["error"]}'))
                        continue
                        
                    total_updated += results["updated_images"]
                    total_skipped += results["skipped_images"]
                    total_errors += results["errors"]
                    
                    # Track entropy changes for analysis
                    for detail in results["details"]:
                        if detail["status"] == "updated":
                            old_val = detail.get("old_entropy", 0)
                            new_val = detail.get("new_entropy", 0)
                            raw_val = detail.get("raw_entropy", 0)
                            
                            # Add to the list of all entropies for distribution analysis
                            if new_val:
                                all_entropies.append((detail["id"], new_val, raw_val))
                            
                            if old_val and new_val:
                                change = abs(new_val - old_val)
                                entropy_changes.append((detail["id"], old_val, new_val, raw_val, change))
                                
                                if verbose and change > 0.1:
                                    self.stdout.write(f'  Image {detail["id"]}: '
                                                     f'Old: {old_val:.2f}, '
                                                     f'New: {new_val:.2f}, '
                                                     f'Raw: {raw_val:.2f}, '
                                                     f'Change: {change:.2f}')
                    
                    # Print progress
                    self.stdout.write(f'  Progress: {i + len(batch_ids)}/{total_images} images processed')
            
            # Analyze and print entropy change statistics
            if entropy_changes:
                changes = [c[4] for c in entropy_changes]
                avg_change = sum(changes) / len(changes)
                max_change = max(changes)
                significant_changes = sum(1 for c in changes if c > 0.1)
                
                # Calculate raw entropy statistics
                raw_values = [c[3] for c in entropy_changes]
                min_raw = min(raw_values)
                max_raw = max(raw_values)
                avg_raw = sum(raw_values) / len(raw_values)
                
                self.stdout.write(self.style.SUCCESS(
                    f'\nEntropy Change Analysis:\n'
                    f'Average change: {avg_change:.2f}\n'
                    f'Maximum change: {max_change:.2f}\n'
                    f'Significant changes (>0.1): {significant_changes} out of {len(changes)} images\n'
                    f'Raw entropy range: {min_raw:.2f} - {max_raw:.2f} (avg: {avg_raw:.2f})'
                ))
                
                # Check if entropy values are now differentiated
                sorted_raw = sorted(raw_values)
                differences = [sorted_raw[i+1] - sorted_raw[i] for i in range(len(sorted_raw)-1)]
                avg_diff = sum(differences) / len(differences) if differences else 0
                
                self.stdout.write(self.style.SUCCESS(
                    f'\nEntropy Differentiation Analysis:\n'
                    f'Average difference between consecutive values: {avg_diff:.4f}\n'
                    f'Number of unique entropy values: {len(set([round(v, 3) for v in raw_values]))}\n'
                    f'Total images: {len(raw_values)}'
                ))
                
                if avg_diff > 0.001:
                    self.stdout.write(self.style.SUCCESS(
                        f'SUCCESS: Entropy values are now differentiated between images'
                    ))
                else:
                    self.stdout.write(self.style.WARNING(
                        f'WARNING: Entropy values may still be too similar between images'
                    ))
                
                # Print examples of largest changes
                if verbose:
                    self.stdout.write("\nLargest entropy changes:")
                    sorted_changes = sorted(entropy_changes, key=lambda x: x[4], reverse=True)
                    for img_id, old_val, new_val, raw_val, change in sorted_changes[:10]:
                        self.stdout.write(f'  Image {img_id}: {old_val:.2f} -> {new_val:.2f} (Raw: {raw_val:.2f}, Change: {change:.2f})')
                
                # Print entropy distribution if requested
                if analyze and all_entropies:
                    self.stdout.write("\nEntropy Distribution Analysis:")
                    
                    # Group by rounded entropy value to see clusters
                    rounded_entropy = {}
                    for img_id, scaled_val, raw_val in all_entropies:
                        rounded = round(raw_val, 2)
                        if rounded not in rounded_entropy:
                            rounded_entropy[rounded] = []
                        rounded_entropy[rounded].append((img_id, scaled_val))
                    
                    # Print distribution
                    self.stdout.write(f"{'Entropy Value':<15} | {'Count':<10} | {'Image IDs (sample)'}")
                    self.stdout.write("-" * 60)
                    
                    for entropy_val in sorted(rounded_entropy.keys()):
                        ids = rounded_entropy[entropy_val]
                        id_sample = [str(img_id) for img_id, _ in ids[:3]]
                        if len(ids) > 3:
                            id_sample.append("...")
                        self.stdout.write(f"{entropy_val:<15.2f} | {len(ids):<10} | {', '.join(id_sample)}")
            
            # Print final summary
            end_time = time.time()
            duration = end_time - start_time
            
            self.stdout.write(self.style.SUCCESS(
                f'Finished recalculating entropy for {total_images} images in {duration:.2f} seconds\n'
                f'Updated: {total_updated}, Skipped: {total_skipped}, Errors: {total_errors}'
            )) 