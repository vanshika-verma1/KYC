# KYC License Validation & Face Matching API

A production-ready FastAPI-based system for validating US driver's licenses and performing face matching for KYC (Know Your Customer) purposes. Built with enhanced security, accuracy, and compliance features.

## 🚀 Latest Enhancements (v2.0)

✅ **Advanced OCR Engine**: Multi-engine OCR with Tesseract + EasyOCR fallback
✅ **Document Processing**: Automatic document straightening and advanced preprocessing
✅ **Region-Specific OCR**: Targeted text extraction for better accuracy
✅ **Performance Optimization**: Async processing, caching, and optimized pipelines
✅ **Comprehensive Audit Logging**: Full compliance tracking and audit trails
✅ **Enhanced Error Handling**: Detailed error reporting and recovery mechanisms

## 🚀 Features

- **US License Validation**: PDF417 barcode parsing with OCR text verification
- **Advanced OCR Engine**: Multi-engine OCR (Tesseract + EasyOCR) with automatic fallback
- **Document Processing**: Automatic document straightening and advanced preprocessing
- **Region-Specific OCR**: Targeted text extraction for maximum accuracy
- **Face Matching**: Advanced facial recognition with quality assessment
- **Performance Optimized**: Async processing, intelligent caching, and optimized pipelines
- **Production Ready**: Enhanced security, error handling, and comprehensive logging
- **Official Compliance**: Proper thresholds and validation for official use
- **Audit & Monitoring**: Comprehensive audit logging and performance metrics
- **Health Checks**: Built-in health monitoring and metrics endpoints

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Response Formats](#response-formats)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Tesseract OCR installed
- Webcam/Camera (for face capture)

### Installation

1. **Clone and setup**:
```bash
# Navigate to your project directory
cd /path/to/your/project

# Install Python dependencies
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
# Set allowed origins (replace with your frontend domains)
export ALLOWED_ORIGINS="http://localhost:3000,https://yourdomain.com"

# For Windows with Tesseract in Models folder (auto-detected)
# No additional configuration needed
```

3. **Run the API**:
```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

4. **Verify installation**:
```bash
curl http://localhost:8000/docs
# Should show Swagger UI documentation
```

## 📦 Installation Details

### Dependencies

The system requires the following packages:

```txt
facenet_pytorch     # Face recognition model
opencv-python       # Computer vision processing
fastapi            # Web framework
uvicorn            # ASGI server
pytesseract        # OCR functionality
zxing-cpp          # Barcode reading
python-multipart   # File upload handling
numpy             # Numerical computing
Pillow            # Image processing
python-dateutil   # Date utilities
```

### Tesseract OCR Setup

The system automatically detects Tesseract in the `Models/Tesseract-OCR/` folder. Ensure your Tesseract installation includes:
- `tesseract.exe` (main executable)
- `eng.traineddata` (English language pack)
- Required DLLs for Windows

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `"http://localhost:3000,http://localhost:8080"` | Comma-separated list of allowed CORS origins |
| `TESSERACT_CMD` | `auto-detected` | Full path to Tesseract executable (optional) |

### API Configuration

**File Upload Limits:**
- Maximum file size: 10MB per image
- Supported formats: JPG, JPEG, PNG, BMP, TIFF

**Security Settings:**
- Configurable CORS origins
- Input validation and sanitization
- Comprehensive error handling

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. License Validation
**POST** `/license/validate_license`

Validates US driver's license with barcode and OCR verification.

**Parameters:**
- `back_image` (file): License back image (with barcode)
- `front_image` (file): License front image (with photo/text)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/license/validate_license" \
  -F "back_image=@license_back.jpg" \
  -F "front_image=@license_front.jpg"
```

#### 2. Face Matching
**POST** `/selfie/validate_selfie`

Compares two face images for identity verification.

**Parameters:**
- `file1` (file): First image
- `file2` (file): Second image

**Example Request:**
```bash
curl -X POST "http://localhost:8000/selfie/validate_selfie" \
  -F "file1=@id_photo.jpg" \
  -F "file2=@selfie.jpg"
```

#### 3. Debug Images (Troubleshooting)
**GET** `/license/debug_images`

Lists all saved debug images for OCR troubleshooting.

**Example Request:**
```bash
curl http://localhost:8000/license/debug_images
```

**GET** `/license/debug_image/{filename}`

Retrieves a specific debug image showing OCR preprocessing.

**Example Request:**
```bash
curl http://localhost:8000/license/debug_image/debug_ocr_20241206_143052.jpg \
  --output ocr_debug.jpg
```

## 💡 Usage Examples

### Python Client Example

```python
import requests

# API Base URL
BASE_URL = "http://localhost:8000"

def validate_license(front_image_path, back_image_path):
    """Validate driver's license"""
    with open(front_image_path, 'rb') as front, open(back_image_path, 'rb') as back:
        files = {
            'front_image': front,
            'back_image': back
        }
        response = requests.post(f"{BASE_URL}/license/validate_license", files=files)
        return response.json()

def match_faces(image1_path, image2_path):
    """Match two face images"""
    with open(image1_path, 'rb') as img1, open(image2_path, 'rb') as img2:
        files = {
            'file1': img1,
            'file2': img2
        }
        response = requests.post(f"{BASE_URL}/selfie/validate_selfie", files=files)

        # Returns image with bounding boxes and match percentage
        if response.status_code == 200:
            with open('result.jpg', 'wb') as f:
                f.write(response.content)

            # Get match result from headers
            match_result = response.headers.get('X-Match-Result')
            similarity = response.headers.get('X-Similarity-Score')
            print(f"Match: {match_result}, Similarity: {similarity}%")

        return response
```

### JavaScript/Client Example

```javascript
const API_BASE = 'http://localhost:8000';

async function validateLicense(frontImageFile, backImageFile) {
    const formData = new FormData();
    formData.append('front_image', frontImageFile);
    formData.append('back_image', backImageFile);

    try {
        const response = await fetch(`${API_BASE}/license/validate_license`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('Validation Result:', result);
        return result;
    } catch (error) {
        console.error('Validation Error:', error);
    }
}

async function matchFaces(image1File, image2File) {
    const formData = new FormData();
    formData.append('file1', image1File);
    formData.append('file2', image2File);

    try {
        const response = await fetch(`${API_BASE}/selfie/validate_selfie`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            // Get result image
            const resultBlob = await response.blob();
            const resultImageUrl = URL.createObjectURL(resultBlob);

            // Get match info from headers
            const matchResult = response.headers.get('X-Match-Result');
            const similarity = response.headers.get('X-Similarity-Score');

            console.log(`Match: ${matchResult}, Similarity: ${similarity}%`);
            return { imageUrl: resultImageUrl, matchResult, similarity };
        }
    } catch (error) {
        console.error('Face Matching Error:', error);
    }
}
```

## 📊 Response Formats

### License Validation Response

```json
{
  "success": true,
  "parsed_fields": {
    "DAC": "SMITH",
    "DAD": "JOHN",
    "DBB": "01012020",
    "DBA": "01012030",
    "DAQ": "123456789"
  },
  "ocr_raw_text": "JOHN SMITH 123456789 DOB 01/01/2020 EXP 01/01/2030",
  "authenticity_score": 0.85,
  "confidence_level": "HIGH",
  "validations": {
    "name_match": true,
    "dob_match": true,
    "id_match": true,
    "dates_valid": true,
    "date_logic_valid": true,
    "image_quality": true
  },
  "image_quality_metrics": {
    "blur_score": 0.75,
    "resolution_score": 1.0,
    "front_image_size": "1080x720",
    "back_image_size": "1080x720",
    "document_straightened": true,
    "text_regions_found": 5,
    "preprocessing_applied": ["histogram_equalization", "gentle_denoising"]
  },
  "debug_info": {
    "debug_image_saved": true,
    "debug_image_path": "C:\\KYC\\debug_images\\debug_ocr_20241206_143052.jpg",
    "regions_processed": 5,
    "ocr_text_length": 28
  },
  "processing_info": {
    "ocr_engines": ["tesseract_enhanced"],
    "preprocessing_steps": ["document_straightening", "histogram_equalization", "region_detection"],
    "total_processing_time": "optimized",
    "enhanced_accuracy": true
  },
  "processing_timestamp": "2024-01-15T10:30:00.123456"
}
```

### Error Response

```json
{
  "detail": "No PDF417 barcode found - ensure back image shows barcode clearly"
}
```

### Face Matching Response

Returns a JPEG image with bounding boxes and match percentage overlay, plus headers:
- `X-Match-Result`: "True" or "False"
- `X-Similarity-Score`: "85.67"
- `X-Processing-Time`: Response time in seconds

### Enhanced Features (v2.0)

**Advanced OCR Capabilities:**
- Multi-engine OCR with automatic fallback (Tesseract → EasyOCR)
- Document straightening using edge detection and Hough transforms
- Region-specific OCR targeting for maximum accuracy
- Advanced image preprocessing (CLAHE, denoising, sharpening)

**Performance Optimizations:**
- Intelligent result caching (5-minute TTL)
- Async processing for CPU-intensive operations
- Thread pool execution for concurrent tasks
- Optimized image processing pipelines

**Compliance & Monitoring:**
- Comprehensive audit logging for regulatory compliance
- Performance metrics and health check endpoints
- Request tracking with unique IDs
- Processing time monitoring

## 🔧 Troubleshooting

### OCR Debug Images

For troubleshooting OCR issues, the system automatically saves debug images showing:
- **Green rectangles**: Detected text regions
- **Region numbers**: Order of processing priority
- **Processed image**: What the OCR engine actually sees

**Access debug images:**
```bash
# List all debug images
curl http://localhost:8000/license/debug_images

# Download specific debug image
curl "http://localhost:8000/license/debug_image/debug_ocr_20241206_143052.jpg" \
  --output debug_image.jpg
```

### Common Issues

**1. Tesseract Not Found**
```bash
# Check if Tesseract is in the Models folder
ls -la Models/Tesseract-OCR/tesseract.exe

# Or install system-wide Tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Ubuntu: sudo apt-get install tesseract-ocr
```

**2. Poor OCR Accuracy**
- Ensure license images are well-lit and in focus
- Check that text is clearly readable
- Verify barcode is not damaged or obscured

**3. Face Detection Issues**
- Ensure faces are clearly visible and well-lit
- Avoid extreme angles or partial face occlusion
- Check image quality requirements in response

**4. CORS Errors**
```bash
# Add your frontend domain to environment
export ALLOWED_ORIGINS="http://localhost:3000,https://yourapp.com"
```

**5. Memory Issues**
- Reduce image resolution before upload (recommended: 1080x720)
- Clear temporary files regularly
- Monitor system resources

### Logging

The API logs all operations to the console. Key log levels:
- `INFO`: Successful operations
- `WARNING`: Non-critical issues
- `ERROR`: Failed operations

### Performance Tuning

**For High Load:**
```bash
# Run with multiple workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Or use gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🔒 Security Considerations

### Production Deployment

1. **HTTPS Only**: Always use SSL/TLS in production
2. **API Keys**: Implement proper authentication (currently disabled per request)
3. **Rate Limiting**: Add rate limiting middleware
4. **Input Validation**: All inputs are validated, but consider additional sanitization
5. **Audit Logging**: Enable comprehensive audit trails for compliance

### File Upload Security

- File type validation (extension + content checking)
- File size limits (10MB max)
- Virus scanning recommended for production
- Secure file storage practices

### Network Security

- Use firewalls to restrict API access
- Implement proper CORS policies
- Monitor for unusual access patterns
- Regular security updates

## 📈 Monitoring & Health Checks

### Health Endpoint
```bash
curl http://localhost:8000/health
# Returns: {"status": "healthy"}
```

### Metrics to Monitor

- Response times for both endpoints
- Success/failure rates
- OCR confidence scores
- Face match thresholds
- System resource usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is intended for official KYC purposes. Ensure compliance with local regulations before deployment.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Check logs for detailed error messages
4. Ensure all prerequisites are met

---

**Built with ❤️ for secure KYC validation**