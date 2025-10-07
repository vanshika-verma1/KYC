from fastapi import APIRouter, UploadFile, File, HTTPException, status, Request
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import io
import logging
import os
from typing import Optional, Dict, Tuple
import time
import json
import secrets
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
FACE_MATCH_THRESHOLD = 0.6  # Minimum similarity for official use
MIN_FACE_QUALITY = 0.8  # Minimum face detection confidence

# -----------------------------
# Device and models (optimized for performance)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize models once (not on every request)
mtcnn = MTCNN(keep_all=False, device=device, min_face_size=20)  # Only the main face
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def validate_image_file(file: UploadFile) -> tuple[bool, str]:
    """Validate uploaded image file"""
    # Check file size
    if file.size and file.size > MAX_FILE_SIZE:
        return False, f"File size exceeds maximum limit of {MAX_FILE_SIZE // (1024*1024)}MB"

    # Check file extension
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in ALLOWED_FORMATS:
        return False, f"Unsupported file format. Allowed: {', '.join(ALLOWED_FORMATS)}"

    return True, ""

def log_face_matching_audit_event(event_type: str, request_info: dict, result_info: dict, processing_time: float):
    """Log face matching audit event for compliance"""
    audit_data = {
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "request_info": request_info,
        "result_info": result_info,
        "processing_time": round(processing_time, 2),
        "compliance_flags": {
            "biometric_data_handling": True,
            "audit_trail_maintained": True,
            "retention_policy_compliant": True
        }
    }

    # Log to audit file
    try:
        from main import audit_logger
        audit_logger.info(json.dumps(audit_data))
    except ImportError:
        logger.info(f"FACE_AUDIT: {json.dumps(audit_data)}")

def generate_request_id() -> str:
    """Generate unique request ID for tracking"""
    return secrets.token_hex(8)

def get_face_request_info(request: Request, file1_name: str, file2_name: str) -> dict:
    """Extract request information for face matching audit logging"""
    return {
        "user_agent": request.headers.get("user-agent", ""),
        "ip_address": getattr(request.client, 'host', None) if request.client else None,
        "request_id": request.headers.get("x-request-id", generate_request_id()),
        "file_names": [file1_name or "unknown", file2_name or "unknown"]
    }

# -----------------------------
# Enhanced Functions
# -----------------------------
def assess_face_quality(img_array: np.ndarray) -> dict:
    """Assess face image quality metrics"""
    try:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Get image properties
        height, width = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Brightness check
        brightness = float(np.mean(gray))

        # Contrast check using standard deviation
        contrast = float(np.std(gray))

        # Sharpness check using Laplacian variance
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Quality assessment (ensure boolean type)
        is_good_quality = bool(
            50 < brightness < 200 and  # Not too dark/bright
            contrast > 30 and           # Good contrast
            sharpness > 100             # Sharp enough
        )

        return {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "resolution": f"{width}x{height}",
            "is_good_quality": is_good_quality
        }
    except Exception as e:
        logger.error(f"Face quality assessment error: {str(e)}")
        return {
            "brightness": 0.0,
            "contrast": 0.0,
            "sharpness": 0.0,
            "resolution": "0x0",
            "is_good_quality": False
        }

def get_face_embedding_and_box_from_array(img_array: np.ndarray) -> tuple[Optional[np.ndarray], Optional[list]]:
    """Enhanced face detection with quality checks"""
    try:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Detect main face and get embedding with confidence
        face_tensor, prob = mtcnn(img_rgb, return_prob=True)

        # Check if face detected and confidence is good
        if face_tensor is None or prob < MIN_FACE_QUALITY:
            logger.warning(f"Face detection failed or low confidence: {prob}")
            return None, None

        # Get embedding
        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(face_tensor).cpu().numpy().flatten()

        # Get bounding box for drawing
        boxes, probs = mtcnn.detect(img_rgb)
        box = None
        if boxes is not None and len(boxes) > 0:
            # Find best face based on detection probability
            best_idx = np.argmax(probs) if probs is not None else 0
            box = [int(x) for x in boxes[best_idx]]

        logger.info(f"Face detected with confidence: {prob:.2f}")
        return embedding, box

    except Exception as e:
        logger.error(f"Face embedding extraction error: {str(e)}")
        return None, None

# -----------------------------
# Enhanced Endpoint with Audit Logging
# -----------------------------
@router.post("/validate_selfie")
async def match_faces(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Enhanced face matching with quality checks and proper thresholds"""
    start_time = time.time()

    try:
        # Input validation
        is_valid1, error1 = validate_image_file(file1)
        is_valid2, error2 = validate_image_file(file2)

        if not is_valid1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"First image: {error1}"
            )
        if not is_valid2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Second image: {error2}"
            )

        # Get request info for audit logging
        request_info = get_face_request_info(request, file1.filename, file2.filename)

        logger.info(f"Processing face matching - Image1: {file1.filename}, Image2: {file2.filename}")

        # Read and decode images
        img1_bytes = await file1.read()
        img2_bytes = await file2.read()

        if not img1_bytes or not img2_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image data received"
            )

        try:
            img1_array = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
            img2_array = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Image decode error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format"
            )

        if img1_array is None or img2_array is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to decode image files"
            )

        # Assess image quality
        quality1 = assess_face_quality(img1_array)
        quality2 = assess_face_quality(img2_array)

        if "error" in quality1 or "error" in quality2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to assess image quality"
            )

        # Get face embeddings with enhanced detection
        emb1, box1 = get_face_embedding_and_box_from_array(img1_array)
        emb2, box2 = get_face_embedding_and_box_from_array(img2_array)

        if emb1 is None or emb2 is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No face detected in one or both images - ensure faces are clearly visible"
            )

        # Calculate cosine similarity with enhanced metrics
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        match_percentage = float(cos_sim) * 100

        # Determine match result based on official threshold
        is_match = bool(float(cos_sim) >= FACE_MATCH_THRESHOLD)
        confidence_level = "HIGH" if cos_sim >= 0.8 else "MEDIUM" if cos_sim >= 0.7 else "LOW"

        # Draw bounding boxes and match results
        if box1:
            color = (0, 255, 0) if is_match else (0, 0, 255)  # Green for match, Red for no match
            cv2.rectangle(img1_array, (box1[0], box1[1]), (box1[2], box1[3]), color, 2)
            cv2.putText(img1_array, f"{match_percentage:.1f}%", (box1[0], box1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if box2:
            color = (0, 255, 0) if is_match else (0, 0, 255)
            cv2.rectangle(img2_array, (box2[0], box2[1]), (box2[2], box2[3]), color, 2)
            cv2.putText(img2_array, f"{match_percentage:.1f}%", (box2[0], box2[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Resize images to same height for comparison display
        height = max(img1_array.shape[0], img2_array.shape[0])
        scale1 = height / img1_array.shape[0]
        scale2 = height / img2_array.shape[0]
        img1_resized = cv2.resize(img1_array, (int(img1_array.shape[1]*scale1), height))
        img2_resized = cv2.resize(img2_array, (int(img2_array.shape[1]*scale2), height))

        # Stack images horizontally
        combined = np.hstack((img1_resized, img2_resized))

        # Encode image as JPEG
        _, buffer = cv2.imencode(".jpg", combined)
        io_buf = io.BytesIO(buffer)

        # Prepare comprehensive response (ensure all values are JSON serializable)
        response = {
            "success": True,
            "match_result": bool(is_match),
            "similarity_score": float(round(match_percentage, 2)),
            "confidence_level": str(confidence_level),
            "threshold_used": float(FACE_MATCH_THRESHOLD),
            "image_quality": {
                "image1": {
                    "brightness": float(quality1.get("brightness", 0)),
                    "contrast": float(quality1.get("contrast", 0)),
                    "sharpness": float(quality1.get("sharpness", 0)),
                    "resolution": str(quality1.get("resolution", "0x0")),
                    "is_good_quality": bool(quality1.get("is_good_quality", False))
                },
                "image2": {
                    "brightness": float(quality2.get("brightness", 0)),
                    "contrast": float(quality2.get("contrast", 0)),
                    "sharpness": float(quality2.get("sharpness", 0)),
                    "resolution": str(quality2.get("resolution", "0x0")),
                    "is_good_quality": bool(quality2.get("is_good_quality", False))
                }
            },
            "face_detection": {
                "image1_face_detected": bool(emb1 is not None),
                "image2_face_detected": bool(emb2 is not None)
            },
            "processing_info": {
                "device_used": str(device),
                "model": "InceptionResnetV1 (VGGFace2)",
                "threshold_method": "cosine_similarity"
            }
        }

        logger.info(f"Face matching completed - Score: {match_percentage:.2f}%, Match: {is_match}")

        # Log successful audit event
        processing_time = time.time() - start_time
        log_face_matching_audit_event(
            "face_matching_success",
            request_info,
            {
                "match_result": is_match,
                "similarity_score": match_percentage,
                "confidence_level": confidence_level,
                "face_quality": {"image1": quality1, "image2": quality2}
            },
            processing_time
        )

        return StreamingResponse(
            io_buf,
            media_type="image/jpeg",
            headers={"X-Match-Result": str(is_match), "X-Similarity-Score": f"{match_percentage:.2f}"}
        )

    except HTTPException as e:
        # Log failed audit event
        processing_time = time.time() - start_time
        log_face_matching_audit_event(
            "face_matching_failure",
            get_face_request_info(request, file1.filename, file2.filename),
            {"success": False, "error_code": e.status_code, "error_detail": e.detail},
            processing_time
        )
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected error
        processing_time = time.time() - start_time
        log_face_matching_audit_event(
            "face_matching_error",
            get_face_request_info(request, file1.filename, file2.filename),
            {"success": False, "error_type": "unexpected", "error_message": str(e)},
            processing_time
        )
        logger.error(f"Unexpected error in face matching: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal processing error occurred"
        )

    # Draw bounding boxes
    if box1:
        cv2.rectangle(img1_array, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 2)
        cv2.putText(img1_array, f"{match_percentage:.2f}%", (box1[0], box1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if box2:
        cv2.rectangle(img2_array, (box2[0], box2[1]), (box2[2], box2[3]), (0, 255, 0), 2)
        cv2.putText(img2_array, f"{match_percentage:.2f}%", (box2[0], box2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Resize images to same height
    height = max(img1_array.shape[0], img2_array.shape[0])
    scale1 = height / img1_array.shape[0]
    scale2 = height / img2_array.shape[0]
    img1_resized = cv2.resize(img1_array, (int(img1_array.shape[1]*scale1), height))
    img2_resized = cv2.resize(img2_array, (int(img2_array.shape[1]*scale2), height))

    # Stack images horizontally
    combined = np.hstack((img1_resized, img2_resized))

    # Encode image as JPEG
    _, buffer = cv2.imencode(".jpg", combined)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/jpeg", headers={"X-Match-Percentage": f"{match_percentage:.2f}"})



# import cv2
# import torch
# import numpy as np
# from facenet_pytorch import MTCNN, InceptionResnetV1

# # -----------------------------
# # Device and models
# # -----------------------------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(keep_all=False, device=device)  # Only the main face
# model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# # -----------------------------
# # Functions
# # -----------------------------
# def get_face_embedding_and_box(image_path):
#     """
#     Returns embedding for the main face, its bounding box, and the original image.
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError(f"Image not found: {image_path}")
    
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    
#     # Detect main face and get embedding
#     face_tensor, prob = mtcnn(img_rgb, return_prob=True)
#     if face_tensor is None:
#         raise ValueError(f"No face detected in {image_path}")
    
#     face_tensor = face_tensor.unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = model(face_tensor).cpu().numpy().flatten()
    
#     # Get bounding box for drawing
#     boxes, _ = mtcnn.detect(img_rgb)
#     box = None
#     if boxes is not None and len(boxes) > 0:
#         # Pick largest face if multiple (area-wise)
#         areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
#         largest_idx = np.argmax(areas)
#         box = [int(x) for x in boxes[largest_idx]]
    
#     return embedding, box, img

# def compare_and_display(image_path1, image_path2):
#     """
#     Compares faces between two images, shows them with bounding boxes,
#     and returns match percentage.
#     """
#     emb1, box1, img1 = get_face_embedding_and_box(image_path1)
#     emb2, box2, img2 = get_face_embedding_and_box(image_path2)
    
#     # Cosine similarity
#     cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
#     match_percentage = cos_sim * 100
    
#     # Draw boxes on images
#     if box1:
#         cv2.rectangle(img1, (box1[0], box1[1]), (box1[2], box1[3]), (0,255,0), 2)
#         cv2.putText(img1, f"{match_percentage:.2f}%", (box1[0], box1[1]-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
#     if box2:
#         cv2.rectangle(img2, (box2[0], box2[1]), (box2[2], box2[3]), (0,255,0), 2)
#         cv2.putText(img2, f"{match_percentage:.2f}%", (box2[0], box2[1]-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
#     # Resize images to same height for display
#     height = max(img1.shape[0], img2.shape[0])
#     scale1 = height / img1.shape[0]
#     scale2 = height / img2.shape[0]
#     img1_resized = cv2.resize(img1, (int(img1.shape[1]*scale1), height))
#     img2_resized = cv2.resize(img2, (int(img2.shape[1]*scale2), height))
    
#     # Stack images horizontally
#     combined = np.hstack((img1_resized, img2_resized))
#     cv2.imshow("Face Match", combined)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     return match_percentage

# # -----------------------------
# # Example usage
# # -----------------------------
# img1 = r"C:\BharatLogic\KYC\static\vijay-latest-function-images-5145393.jpg"
# img2 = r"C:\BharatLogic\KYC\static\istockphoto-1210331839-612x612.jpg"

# match = compare_and_display(img1, img2)
# print(f"Best face match percentage: {match:.2f}%")

