from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import io

router = APIRouter()

# -----------------------------
# Device and models
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)  # Only the main face
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# -----------------------------
# Functions
# -----------------------------
def get_face_embedding_and_box_from_array(img_array):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Detect main face and get embedding
    face_tensor, prob = mtcnn(img_rgb, return_prob=True)
    if face_tensor is None:
        return None, None

    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_tensor).cpu().numpy().flatten()

    # Get bounding box for drawing
    boxes, _ = mtcnn.detect(img_rgb)
    box = None
    if boxes is not None and len(boxes) > 0:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        largest_idx = np.argmax(areas)
        box = [int(x) for x in boxes[largest_idx]]

    return embedding, box

# -----------------------------
# Endpoint
# -----------------------------
@router.post("/validate_selfie")
async def match_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Read images
    img1_bytes = await file1.read()
    img2_bytes = await file2.read()
    img1_array = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2_array = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img1_array is None or img2_array is None:
        return {"error": "Invalid image files"}

    # Get embeddings and boxes
    emb1, box1 = get_face_embedding_and_box_from_array(img1_array)
    emb2, box2 = get_face_embedding_and_box_from_array(img2_array)

    if emb1 is None or emb2 is None:
        return {"error": "No face detected in one or both images"}

    # Cosine similarity
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    match_percentage = float(cos_sim) * 100

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

