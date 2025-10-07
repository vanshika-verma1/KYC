"""
liveness_server.py

Single-file prototype that:
 - Accepts base64 JPEG frames over WebSocket (FastAPI)
 - Detects face + landmarks with MediaPipe FaceMesh
 - Active checks: eye-blink (EAR), head-pose (yaw)
 - Passive PAD / anti-spoofing: runs provided ONNX model on face crop
 - Returns instruction JSON to client

Usage:
  1) Place your PAD ONNX file (e.g. AntiSpoofing_print-replay_1.5_128.onnx)
  2) Edit PAD_MODEL_PATH below if needed
  3) python liveness_server.py
  4) Open client.html and connect to ws://localhost:8000/ws
"""

import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import onnxruntime as ort
import mediapipe as mp
import math
from typing import Tuple

# ----------------- USER CONFIG -----------------
PAD_MODEL_PATH = r"C:\KYC\Models\Onnx\AntiSpoofing_print-replay_1.5_128.onnx"
SPOOF_THRESHOLD = 0.5        # depends on model; tune with validation
EAR_THRESHOLD = 0.23         # blink threshold (tune)
EAR_CONSEC_FRAMES = 2        # frames ear must stay below threshold to count blink
YAW_THRESHOLD_DEG = 20.0     # degrees to consider "turned head"
SEND_FRAME_INTERVAL_MS = 300 # client should send roughly this often (not enforced)
# ------------------------------------------------

# ----------------- PAD (ONNX) LOADING -----------------
pad_session = None
pad_input_name = None
pad_input_shape = None  # (C,H,W) or (N,C,H,W) shape from model metadata

def load_pad_model(onnx_path: str):
    global pad_session, pad_input_name, pad_input_shape
    pad_session = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
    pad_input_name = pad_session.get_inputs()[0].name
    shape = pad_session.get_inputs()[0].shape  # e.g. [1,3,128,128] or [None,3,128,128]
    # find H,W:
    if len(shape) == 4:
        _, c, h, w = shape
        pad_input_shape = (int(c), int(h), int(w))
    else:
        # fallback defaults
        pad_input_shape = (3, 128, 128)
    print(f"[PAD] Loaded ONNX {onnx_path} -> input shape {pad_input_shape}, input name '{pad_input_name}'")

def predict_spoof(face_bgr: np.ndarray) -> float:
    """
    Preprocess face crop and run pad_session -> return spoof_probability (0..1)
      - Attempts sensible default preprocessing:
         * resize to model HxW
         * BGR -> RGB
         * float32 /255.0
         * normalize to [-1,1] by default (img-0.5)/0.5
      - If the model expects different preprocessing you must change this.
    """
    if pad_session is None:
        raise RuntimeError("PAD model not loaded")
    _, H, W = pad_input_shape
    if face_bgr.size == 0:
        return 1.0  # empty crop - treat as spoof/high risk

    img = cv2.resize(face_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # default normalization (many PADs use [-1,1] or [0,1]); adjust as needed
    img = (img - 0.5) / 0.5
    # -> [C,H,W]
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    # -> [1,C,H,W]
    inp = img[np.newaxis, :]

    outputs = pad_session.run(None, {pad_input_name: inp})
    out = outputs[0]
    # Many PAD models output logits/probs shape (1,2) where [real_prob, spoof_prob].
    # If shape matches, try to read spoof index 1, else fallback to sigmoid/logit handling.
    if out is None:
        return 1.0
    out = np.array(out)
    # flatten to (N, K) or (N,)
    if out.ndim == 2 and out.shape[1] >= 2:
        # treat index 1 as spoof prob if values are already probs or logits
        # if outputs are logits, softmax needed:
        vec = out[0]
        # check if they are probabilities (sum ~1)
        s = float(np.sum(vec))
        if 0.9 <= s <= 1.1:
            spoof_prob = float(vec[1])
        else:
            # assume logits -> softmax
            exp = np.exp(vec - np.max(vec))
            probs = exp / np.sum(exp)
            spoof_prob = float(probs[1])
    elif out.ndim >= 1:
        # single-value output (logit -> apply sigmoid)
        val = float(out.flatten()[0])
        spoof_prob = 1.0 / (1.0 + math.exp(-val))
    else:
        spoof_prob = 1.0
    # clamp
    spoof_prob = float(max(0.0, min(1.0, spoof_prob)))
    return spoof_prob

# ----------------- MediaPipe FaceMesh utilities -----------------
mp_face_mesh = mp.solutions.face_mesh

def detect_landmarks_mp(image_rgb: np.ndarray):
    """
    Returns landmarks object or None.
    We run FaceMesh in a context manager per call to keep code simple.
    """
    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as fm:
        res = fm.process(image_rgb)
        if res.multi_face_landmarks:
            return res.multi_face_landmarks[0]
    return None

def lm_to_np(landmarks, img_shape: Tuple[int,int,int]) -> np.ndarray:
    h, w, _ = img_shape
    pts = []
    for lm in landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return np.array(pts, dtype=np.int32)

def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    # eye_pts: 6 points (x,y) order consistent with Mediapipe mesh indices chosen below
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def estimate_head_pose(landmarks_np: np.ndarray, img_shape: Tuple[int,int,int]) -> Tuple[float,float,float]:
    """
    SolvePnP with 6 points to get yaw, pitch, roll (degrees).
    Indices used correspond to MediaPipe 468 landmarks:
      - 1 = near nose tip (approx)
      - 152 = chin
      - 33 = left eye outer
      - 263 = right eye outer
      - 61 = mouth left
      - 291 = mouth right
    """
    try:
        image_pts = np.array([
            landmarks_np[1],
            landmarks_np[152],
            landmarks_np[33],
            landmarks_np[263],
            landmarks_np[61],
            landmarks_np[291]
        ], dtype="double")
    except Exception:
        # fallback if landmarks missing indices
        h, w, _ = img_shape
        return 0.0, 0.0, 0.0

    model_pts = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ], dtype="double")

    h, w, _ = img_shape
    focal = w
    center = (w / 2.0, h / 2.0)
    cam_mat = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4,1))
    success, rot_vec, trans_vec = cv2.solvePnP(model_pts, image_pts, cam_mat, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0
    rmat = cv2.Rodrigues(rot_vec)[0]
    proj = np.hstack((rmat, trans_vec))
    euler = cv2.decomposeProjectionMatrix(proj)[6]
    yaw, pitch, roll = [float(x) for x in euler]
    return yaw, pitch, roll

# ----------------- WebSocket server -----------------
class ClientState:
    def __init__(self):
        self.blink_counter = 0
        self.blinks = 0
        self.last_instruction = None

# Global clients dictionary - shared across all connections
clients = {}

async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients[ws] = ClientState()
    try:
        while True:
            msg = await ws.receive_text()
            obj = json.loads(msg)
            frame_b64 = obj.get("frame")
            if frame_b64 is None:
                await ws.send_json({"error": "no frame key"})
                continue

            # decode image
            try:
                imgbytes = base64.b64decode(frame_b64)
                arr = np.frombuffer(imgbytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception as e:
                await ws.send_json({"error": f"decode_failed: {str(e)}"})
                continue
            if frame is None:
                await ws.send_json({"error": "invalid_frame"})
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1) Detect landmarks + implicitly detect face (MediaPipe)
            landmarks = detect_landmarks_mp(rgb)
            if landmarks is None:
                await ws.send_json({"instruction": "no_face"})
                continue
            lms_np = lm_to_np(landmarks, frame.shape)  # (468,2)

            # 2) Compute EAR blink detection
            # Mediapipe iris/eye indices: using common set for outer/inner eye corners + eyelids
            # Left eye (approx) indices for FaceMesh: [33, 160, 158, 133, 153, 144]
            # Right eye: [362, 385, 387, 263, 373, 380]
            left_idx = [33, 160, 158, 133, 153, 144]
            right_idx = [362, 385, 387, 263, 373, 380]
            try:
                left_eye = lms_np[left_idx]
                right_eye = lms_np[right_idx]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
            except Exception:
                ear = 1.0  # cannot compute -> set high so blink not triggered

            state = clients[ws]
            if ear < EAR_THRESHOLD:
                state.blink_counter += 1
            else:
                if state.blink_counter >= EAR_CONSEC_FRAMES:
                    state.blinks += 1
                state.blink_counter = 0

            # 3) Head pose (yaw)
            yaw, pitch, roll = estimate_head_pose(lms_np, frame.shape)

            # 4) Crop face area for PAD model.
            # We'll compute a bbox from landmarks
            xs = lms_np[:,0]
            ys = lms_np[:,1]
            x1 = int(np.min(xs))
            x2 = int(np.max(xs))
            y1 = int(np.min(ys))
            y2 = int(np.max(ys))
            # pad a bit
            pad_x = int((x2 - x1) * 0.25)
            pad_y = int((y2 - y1) * 0.25)
            fx1 = max(0, x1 - pad_x)
            fy1 = max(0, y1 - pad_y)
            fx2 = min(w, x2 + pad_x)
            fy2 = min(h, y2 + pad_y)
            face_crop = frame[fy1:fy2, fx1:fx2]
            if face_crop.size == 0:
                spoof_prob = 1.0
            else:
                try:
                    spoof_prob = predict_spoof(face_crop)
                except Exception as e:
                    spoof_prob = 1.0

            # 5) Instruction logic (simple FSM)
            instruction = "done"
            if spoof_prob > SPOOF_THRESHOLD:
                instruction = "spoof_detected"
            elif state.blinks < 2:
                instruction = "blink"
            elif abs(yaw) < YAW_THRESHOLD_DEG:
                instruction = "turn_head"
            else:
                instruction = "done"

            state.last_instruction = instruction

            # 6) Reply with helpful data
            await ws.send_json({
                "instruction": instruction,
                "blinks_detected": state.blinks,
                "ear": float(ear),
                "yaw": float(yaw),
                "spoof_prob": float(spoof_prob),
                "bbox": [int(fx1), int(fy1), int(fx2), int(fy2)]
            })

    except WebSocketDisconnect:
        clients.pop(ws, None)
    except Exception as e:
        clients.pop(ws, None)
        # we cannot send to client if broken; just log
        print("Server exception:", str(e))

# ----------------- Liveness WebSocket Router -----------------
def get_liveness_websocket_router():
    """Returns the WebSocket router for liveness detection"""
    return app

def initialize_liveness_model():
    """Initialize the PAD model for liveness detection"""
    print("Loading PAD model:", PAD_MODEL_PATH)
    load_pad_model(PAD_MODEL_PATH)

# Create the FastAPI app instance for liveness
app = FastAPI()
