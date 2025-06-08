import os
import sys
import argparse
import csv
import pydicom
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle

class XRayLabeler:
    def __init__(self, input_dir, output_file):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.dicom_files = []
        self.current_index = 0
        self.annotations = {}
        self.current_rect = None
        self.existing_annotations = {}
        self.load_existing_annotations()
        
    def load_existing_annotations(self):
        """Load existing annotations if output file exists"""
        if self.output_file.exists():
            with open(self.output_file, 'r', newline='') as f:
                reader = csv.reader(f)
                headers = next(reader)
                for row in reader:
                    if len(row) >= 5:
                        filename = row[0]
                        self.existing_annotations[filename] = {
                            'x': float(row[1]),
                            'y': float(row[2]),
                            'width': float(row[3]),
                            'height': float(row[4])
                        }
    
    def scan_directory(self):
        """Scan directory for DICOM, JPG, and PNG files"""
        self.dicom_files = list(self.input_dir.glob('**/*.dcm'))
    
    # If no DICOM files found, try looking for JPG or PNG files
        if not self.dicom_files:
            self.dicom_files = list(self.input_dir.glob('**/*.jpg')) + list(self.input_dir.glob('**/*.png'))
        
            if not self.dicom_files:
                print(f"No DICOM, JPG, or PNG files found in {self.input_dir}")
                sys.exit(1)
            
            print(f"Found {len(self.dicom_files)} JPG or PNG files")
        else:
            print(f"Found {len(self.dicom_files)} DICOM files")

    def start_labeling(self):
        """Start the labeling process"""
        self.scan_directory()
        self.setup_display()
        plt.show()
        
    def setup_display(self):
        """Setup the matplotlib display"""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Add navigation buttons
        self.ax_prev = plt.axes([0.1, 0.01, 0.1, 0.05])
        self.prev_button = plt.Button(self.ax_prev, 'Previous')
        self.prev_button.on_clicked(self.previous_image)
        
        self.ax_next = plt.axes([0.21, 0.01, 0.1, 0.05])
        self.next_button = plt.Button(self.ax_next, 'Next')
        self.next_button.on_clicked(self.next_image)
        
        self.ax_save = plt.axes([0.32, 0.01, 0.1, 0.05])
        self.save_button = plt.Button(self.ax_save, 'Save')
        self.save_button.on_clicked(self.save_annotations)
        
        # Setup rectangle selector
        self.selector = RectangleSelector(
            self.ax, self.on_select, 
            useblit=True,
            button=[1],  # Left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        self.load_current_image()
        
    def on_select(self, eclick, erelease):
        """Callback for rectangle selection"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        current_file = str(self.dicom_files[self.current_index].name)
        self.annotations[current_file] = {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
        
        # Update display
        self.draw_annotations()
        
    def load_current_image(self):
        """Load and display the current image"""
        if not self.dicom_files:
            return
            
        self.ax.clear()
        current_file = self.dicom_files[self.current_index]
        
        try:
            # Try to read as DICOM first
            try:
                ds = pydicom.dcmread(current_file)
                image = ds.pixel_array
            except:
                # If fails, try to read as regular image
                image = cv2.imread(str(current_file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise Exception("Could not read image file")
            
            # Normalize image for display
            if image.max() > 0:
                image = (image / image.max() * 255).astype(np.uint8)
                
            self.ax.imshow(image, cmap='gray')
            self.ax.set_title(f"File: {current_file.name} ({self.current_index + 1}/{len(self.dicom_files)})")
            
            # Load existing annotation if available
            if current_file.name in self.existing_annotations:
                self.annotations[current_file.name] = self.existing_annotations[current_file.name]
                
            self.draw_annotations()
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error reading file {current_file}: {e}")
            self.ax.text(0.5, 0.5, f"Error reading file: {e}", 
                         ha="center", va="center", transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
            
    def draw_annotations(self):
        """Draw existing annotations on the image"""
        for patch in self.ax.patches:
            patch.remove()
            
        current_file = str(self.dicom_files[self.current_index].name)
        if current_file in self.annotations:
            ann = self.annotations[current_file]
            rect = Rectangle(
                (ann['x'], ann['y']), 
                ann['width'], ann['height'],
                linewidth=2, edgecolor='r', facecolor='none'
            )
            self.ax.add_patch(rect)
            self.fig.canvas.draw_idle()
            
    def next_image(self, event=None):
        """Navigate to next image"""
        if self.current_index < len(self.dicom_files) - 1:
            self.current_index += 1
            self.load_current_image()
            
    def previous_image(self, event=None):
        """Navigate to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
            
    def save_annotations(self, event=None):
        """Save annotations to CSV file"""
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'x', 'y', 'width', 'height'])
            
            # Combine existing annotations with new ones
            combined_annotations = {**self.existing_annotations, **self.annotations}
            
            for filename, ann in combined_annotations.items():
                writer.writerow([
                    filename,
                    ann['x'],
                    ann['y'],
                    ann['width'],
                    ann['height']
                ])
                
        print(f"Annotations saved to {self.output_file}")
        
def main():
    parser = argparse.ArgumentParser(description='Label regions in X-ray DICOM images')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing DICOM files')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file for annotations')
    
    args = parser.parse_args()
    
    labeler = XRayLabeler(args.input, args.output)
    labeler.start_labeling()
    
if __name__ == '__main__':
    main() 