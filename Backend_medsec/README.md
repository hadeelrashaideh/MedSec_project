# Medical Image Processing System

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Security Implementation](#security-implementation)
5. [User Roles & Permissions](#user-roles--permissions)
6. [Technical Implementation](#technical-implementation)
7. [Machine Learning Model](#machine-learning-model)
8. [API Reference](#api-reference)
9. [Database Schema](#database-schema)
10. [Libraries & Dependencies](#libraries--dependencies)
11. [Installation & Setup](#installation--setup)
12. [Development Guidelines](#development-guidelines)
13. [Deployment](#deployment)
14. [Troubleshooting](#troubleshooting)
15. [License](#license)

## Overview

The Medical Image Processing System is a comprehensive platform designed for secure handling, processing, and analysis of sensitive medical images. The system employs state-of-the-art object detection, image processing, and encryption technologies to ensure patient data privacy while providing authorized medical professionals with the tools they need for diagnosis and treatment.

This system addresses the critical challenge in medical imaging: balancing accessibility for healthcare professionals with strict privacy requirements for patient data. By automatically detecting and encrypting sensitive regions within medical images, the system allows for secure sharing and storage of medical data.

## Key Features

### Security & Privacy

- **Selective Encryption**: Automatically identifies and encrypts only sensitive regions in medical images
- **AES-256-CBC Encryption**: Military-grade encryption for all sensitive data
- **Diffie-Hellman Key Exchange**: Secure protocol for establishing encryption keys
- **Role-Based Access Control**: Granular permissions based on user roles
- **Audit Logging**: Comprehensive tracking of all system activities

### Image Processing

- **YOLO-based Object Detection**: State-of-the-art AI model for identifying sensitive regions
- **Automatic Region Extraction**: Precise cropping of detected sensitive areas
- **Image Restoration**: On-demand reconstruction of original images for authorized users
- **Format Support**: Handles JPEG, PNG, and other common medical image formats
- **Batch Processing**: Capability to process multiple images efficiently

### Analytics & Reporting

- **Entropy Analysis**: Advanced metrics for assessing image information content
- **Similarity Comparison**: Tools to compare original and processed images
- **Confidence Scoring**: Reliability metrics for detection results
- **Processing Statistics**: Performance metrics for system optimization

### User Experience

- **Intuitive Interfaces**: Separate UIs tailored for lab technicians and doctors
- **Real-time Processing**: Immediate feedback on uploaded images
- **Responsive Design**: Access from various devices and screen sizes
- **Search & Filter**: Powerful tools to locate patient records and images

## System Architecture

The system follows a modern Django-based architecture with the following components:

### Backend Components

1. **Authentication Service**:

   - User management and authentication
   - JWT token handling
   - Diffie-Hellman key exchange implementation
   - Role-based permission enforcement

2. **Patient Management Service**:

   - Patient record CRUD operations
   - Data validation and sanitization
   - Patient history tracking
   - Relationship management between patients and images

3. **Image Processing Pipeline**:

   - Image upload and validation
   - YOLO model integration for object detection
   - Region extraction and processing
   - Encryption/decryption operations
   - Image storage management

4. **Admin Service**:
   - System configuration management
   - User administration
   - Monitoring and reporting tools
   - Audit log access

### Data Flow

1. Lab technician uploads patient image
2. Authentication service validates user permissions
3. Image processing pipeline detects sensitive regions
4. Detected regions are encrypted with AES-256
5. Processed image and encrypted regions are stored
6. Doctor requests access to patient image
7. Authentication service validates doctor's permissions
8. Encrypted regions are decrypted for authorized viewing

## Security Implementation

### Encryption System

- **Algorithm**: AES-256-CBC (Advanced Encryption Standard with 256-bit keys in Cipher Block Chaining mode)
- **Key Management**: Secure key generation and distribution using Diffie-Hellman protocol
- **Initialization Vectors**: Random 16-byte IVs for each encryption operation
- **Padding**: PKCS#7 padding for proper block alignment

### Authentication & Authorization

- **Token-based Authentication**: JWT (JSON Web Tokens) with appropriate expiration
- **Password Security**: Bcrypt hashing with salt for password storage
- **Multi-factor Authentication**: Optional 2FA for enhanced security
- **Session Management**: Secure handling of user sessions with appropriate timeouts

### Data Protection

- **In Transit**: TLS/SSL encryption for all data transmission
- **At Rest**: Encrypted storage for sensitive information
- **Key Security**: Secure key management with proper access controls
- **Data Isolation**: Proper separation between patient data

## User Roles & Permissions

### Lab Technician

- **Permissions**:
  - Upload new patient images
  - Create and update patient records
  - View processing results (blurred and grid images)
  - Access non-sensitive patient information
- **Restrictions**:
  - Cannot view decrypted sensitive regions
  - Cannot modify processed images
  - Limited access to patient history

### Doctor

- **Permissions**:
  - View all patient records
  - Access and decrypt sensitive regions
  - Generate restored images on demand
  - Analyze image metrics and statistics
  - Add medical notes to patient records
- **Restrictions**:
  - Cannot modify original or processed images
  - Limited administrative functions

### Administrator

- **Permissions**:
  - Manage user accounts and permissions
  - Configure system parameters
  - Access audit logs and system metrics
  - Manage encryption settings
- **Restrictions**:
  - Cannot directly access patient data without proper role
  - Actions are logged for accountability

## Technical Implementation

### Image Processing Pipeline

#### 1. Image Upload & Validation

- Supported formats: JPEG, PNG, DICOM
- Size limitations: Up to 20MB per image
- Validation checks: Format verification, dimension constraints, metadata sanitization

#### 2. Object Detection

- **Model**: YOLOv8 (You Only Look Once)
- **Configuration**:
  - Confidence threshold: 0.25 (configurable)
  - Non-maximum suppression IoU threshold: 0.45
  - Classes: Configured for specific sensitive medical regions
- **Performance**:
  - Average inference time: ~200ms per image
  - Precision: >0.92 for supported classes
  - Recall: >0.90 for supported classes

#### 3. Region Processing

- Automatic cropping based on detection bounding boxes
- Padding addition for privacy enhancement
- Format preservation of original image
- Metadata handling and sanitization

#### 4. Encryption Process

- Region-specific encryption with unique IVs
- Original filename and format preservation
- Entropy calculation for quality assurance
- Performance metrics collection

#### 5. Storage & Retrieval

- Efficient storage of blurred images, grid representations, and encrypted regions
- On-demand decryption for authorized users
- Caching system for improved performance
- Backup and recovery mechanisms

### Encryption Flow

#### Key Exchange Protocol

1. Client initiates key exchange request
2. Server generates DH parameters (p, g) and server keypair
3. Server sends public parameters and public key to client
4. Client generates client keypair using server parameters
5. Client sends public key to server
6. Both parties independently compute the shared secret
7. Shared secret is hashed to derive AES-256 encryption key

#### Encryption Process

1. Generate random 16-byte Initialization Vector
2. Pad data using PKCS#7 to match AES block size
3. Encrypt data using AES-256-CBC with derived key and IV
4. Prepend IV to encrypted data for storage
5. Store encryption metadata (format, original filename)

#### Decryption Process

1. Extract IV from first 16 bytes of encrypted data
2. Retrieve encryption key for authorized user
3. Decrypt data using AES-256-CBC
4. Remove padding from decrypted data
5. Verify data integrity through header checks

## Machine Learning Model

### Model Architecture

- **Base Model**: YOLOv8n (nano variant)
- **Framework**: PyTorch
- **Size**: 50MB
- **Architecture**: CSP-Darknet backbone with additional cross-stage partial connections

### Training Details

- **Dataset**: Proprietary medical imaging dataset with annotated sensitive regions
- **Annotations**: 10,000+ bounding boxes across multiple classes
- **Training Regime**:
  - Epochs: 100
  - Batch size: 16
  - Optimizer: SGD with momentum
  - Learning rate: 0.01 with cosine annealing
  - Data augmentation: Random flip, rotation, color jittering

### Model Performance

- **mAP@0.5**: 0.94
- **mAP@0.5:0.95**: 0.86
- **Inference Speed**: 15ms per image on GPU, 200ms on CPU
- **Classes**: Configurable based on medical specialty requirements

### Model Integration

- Located in `/model/best.pt`
- Loaded dynamically at system startup
- Configurable inference parameters via environment variables
- GPU acceleration when available, graceful fallback to CPU

### Model Maintenance

- Periodic retraining with new data
- Performance monitoring and drift detection
- Version control for model files
- A/B testing framework for model improvements

## API Reference

### Authentication Endpoints

#### Register New User

```
POST /api/auth/register/
```

- **Request Body**:
  ```json
  {
    "username": "doctor_smith",
    "email": "smith@hospital.com",
    "password": "secure_password",
    "role": "doctor"
  }
  ```
- **Response**: User details with JWT token

#### Obtain JWT Token

```
POST /api/auth/token/
```

- **Request Body**:
  ```json
  {
    "username": "doctor_smith",
    "password": "secure_password"
  }
  ```
- **Response**: JWT access and refresh tokens

#### Diffie-Hellman Key Exchange

```
GET /api/auth/key-exchange/
POST /api/auth/key-exchange/
```

- **GET Response**: DH parameters and server public key
- **POST Request Body**:
  ```json
  {
    "client_public_key": "large_integer_value"
  }
  ```
- **POST Response**: Success confirmation

### Patient Endpoints

#### Create Patient with Image

```
POST /api/patients/
```

- **Request Body**: Multipart form with patient details and image
- **Response**: Patient ID and processing status

#### Get Patient Details

```
GET /api/patients/{id}/
```

- **Response**: Patient information and latest image summary

#### List Patient Images

```
GET /api/patients/{id}/images/
```

- **Response**: List of all processed images for patient

### Image Endpoints

#### Restore Image

```
GET /api/images/{id}/restore/
```

- **Query Parameters**:
  - `format`: Output format (png/jpeg)
  - `enhance`: Enable enhancement (true/false)
- **Response**: Restored image file

#### Get Image Regions

```
GET /api/images/{id}/regions/
```

- **Response**: List of detected regions with metadata

#### Get Decrypted Region

```
GET /api/regions/{id}/
```

- **Response**: Decrypted region image file

## Database Schema

### Patient Model

```python
class Patient(models.Model):
    id = models.CharField(primary_key=True, max_length=50)
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    note = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

### ProcessedImage Model

```python
class ProcessedImage(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    blurred_image = models.ImageField(upload_to='patient_images/blurred/')
    grid_image = models.ImageField(upload_to='patient_images/grid/')
    restored_image = models.ImageField(upload_to='patient_images/restored/')
    enhanced = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    original_entropy = models.FloatField(null=True)
    encrypted_entropy = models.FloatField(null=True)
    encryption_time = models.FloatField(null=True)
    decryption_time = models.FloatField(null=True)
```

### CroppedRegion Model

```python
class CroppedRegion(models.Model):
    processed_image = models.ForeignKey(ProcessedImage, on_delete=models.CASCADE)
    class_name = models.CharField(max_length=50)
    confidence = models.FloatField()
    x1, y1, x2, y2 = models.IntegerField() # Coordinates
    cropped_image_data = models.BinaryField() # Encrypted data
    original_filename = models.CharField(max_length=255)
    image_format = models.CharField(max_length=10, default='JPEG')
    created_at = models.DateTimeField(auto_now_add=True)
```

### ImageFingerprint Model

```python
class ImageFingerprint(models.Model):
    processed_image = models.OneToOneField(ProcessedImage)
    color_histogram = models.BinaryField()
    avg_hash = models.CharField(max_length=64)
    phash = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

```

## Libraries & Dependencies

### Core Framework

- **Django 4.2+**: Web framework for backend development
- **Django REST Framework 3.14+**: API development toolkit

### Machine Learning & Computer Vision

- **PyTorch 2.0+**: Deep learning framework
- **Ultralytics YOLOv8**: Object detection model
- **OpenCV 4.7+**: Computer vision operations
  - Image processing
  - Contour detection
  - Image transformations
- **NumPy 1.24+**: Numerical computations
- **SciPy 1.10+**: Scientific computing
  - Entropy calculations
  - Statistical operations
- **Pillow 9.5+**: Image processing library
- **ImageHash 4.3+**: Perceptual image hashing

### Security & Encryption

- **Cryptography 39.0+**: Cryptographic recipes and primitives
  - Diffie-Hellman key exchange
  - Key derivation functions
- **PyCryptodome 3.17+**: Cryptographic library
  - AES-256-CBC implementation
  - PKCS#7 padding
- **PyJWT 2.6+**: JSON Web Token implementation
- **Django Simple JWT 5.2+**: JWT authentication for Django REST Framework

### Database & Caching

- **PostgreSQL 14+**: Primary database
- **psycopg2-binary 2.9+**: PostgreSQL adapter for Python
- **Redis 7.0+**: Caching and key storage
- **django-redis 5.2+**: Redis integration for Django

### Testing & Quality Assurance

- **pytest 7.3+**: Testing framework
- **pytest-django 4.5+**: Django integration for pytest
- **coverage 7.2+**: Code coverage measurement

### Utilities & Helpers

- **python-dotenv 1.0+**: Environment variable management
- **django-filter 23.1+**: Filtering for Django querysets
- **django-cors-headers 4.0+**: CORS headers for Django
- **drf-yasg 1.21+**: Swagger/OpenAPI documentation
- **gunicorn 20.1+**: WSGI HTTP server
- **whitenoise 6.4+**: Static file serving

### Frontend Dependencies

- **Bootstrap 5.2+**: CSS framework
- **Chart.js 4.2+**: JavaScript charting library
- **Axios 1.3+**: HTTP client

### Development Tools

- **black**: Python code formatter
- **flake8**: Code linter
- **isort**: Import sorter
- **pre-commit**: Git hooks manager

### External Services

- **SMTP Server**: For email notifications
- **S3-compatible Storage** (optional): For scalable image storage

### System Requirements

- **Python 3.8+**
- **CUDA 11.7+** (for GPU acceleration)
- **Node.js 18+** (for frontend development)
