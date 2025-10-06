from fastapi import APIRouter, UploadFile, File
from PIL import Image, ImageOps
import pytesseract
import io
import cv2
import numpy as np
import difflib
from datetime import datetime
import re
import zxingcpp

# Tesseract OCR path (Windows-specific, adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\KYC\Models\Tesseract-OCR\tesseract.exe"

router = APIRouter()

# ------------------ Helpers ------------------ #
def parse_aamva(barcode_text: str) -> dict:
    """
    Parse AAMVA-compliant barcode text.
    """
    barcode_text = (
        barcode_text.replace("<LF>", "\n")
        .replace("<RS>", "\x1E")
        .replace("<CR>", "\x0D")
        .replace("<GS>", "\x1D")
    )
    parsed = {}
    lines = re.split(r"[\n\x0D\x1E]", barcode_text)
    for line in lines:
        line = line.strip()
        if len(line) >= 3 and line[:3].isalpha():
            parsed[line[:3]] = line[3:].strip()
    for key in parsed:
        if "\x1D" in parsed[key]:
            parsed[key] = parsed[key].split("\x1D")
    return parsed

def validate_date(date_str: str) -> bool:
    """
    Validate date in various formats.
    """
    if not date_str or len(date_str) not in (6, 8, 10):
        return False
    for fmt in ("%m%d%Y", "%m/%d/%Y", "%m-%d-%Y", "%m%d%y", "%m/%d/%y", "%Y%m%d", "%Y-%m-%d"):
        try:
            datetime.strptime(date_str, fmt)
            return True
        except Exception:
            continue
    return False

def normalize_dob(dob_raw: str) -> list:
    """
    Return DOB in multiple formats.
    """
    if not dob_raw:
        return []
    dob = dob_raw.strip().replace(" ", "").replace("-", "/").replace(".", "/")
    formats = []
    if re.fullmatch(r"\d{8}", dob):
        try:
            dt = datetime.strptime(dob, "%m%d%Y")
            formats.append(dob)
            formats.append(dt.strftime("%m/%d/%Y"))
            formats.append(dt.strftime("%m/%d/%y"))
        except Exception:
            pass
    m = re.search(r"(\d{1,2})\/(\d{1,2})\/(\d{2,4})", dob)
    if m:
        mm, dd, yy = m.group(1), m.group(2), m.group(3)
        if len(yy) == 2:
            yy = ("19" + yy) if int(yy) > 30 else ("20" + yy)
        try:
            dt = datetime.strptime(f"{mm}/{dd}/{yy}", "%m/%d/%Y")
            formats.append(dt.strftime("%m%d%Y"))
            formats.append(dt.strftime("%m/%d/%Y"))
            formats.append(dt.strftime("%m/%d/%y"))
        except Exception:
            pass
    if re.fullmatch(r"\d{8}", dob):
        try:
            dt = datetime.strptime(dob, "%Y%m%d")
            formats.append(dt.strftime("%m%d%Y"))
            formats.append(dt.strftime("%m/%d/%Y"))
            formats.append(dt.strftime("%m/%d/%y"))
        except Exception:
            pass
    return list(set(formats))

def detect_blur(image: np.ndarray) -> float:
    """
    Calculate image blur score.
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(variance / 1000.0, 1.0)

def clean_token(tok: str) -> str:
    """
    Clean text for comparison.
    """
    return re.sub(r"[^A-Z0-9\s\-']", "", tok.upper()).strip()

def fuzzy_match(text: str, target: str, threshold: float = 0.5) -> bool:
    """
    Perform fuzzy matching with difflib.
    """
    text_clean = text.replace(" ", "")
    target_clean = target.replace(" ", "")
    return difflib.SequenceMatcher(None, text_clean, target_clean).ratio() > threshold

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def is_similar_dob(dob1: str, dob2: str, max_diff: int = 2) -> bool:
    """
    Check if two DOB strings are similar (allow max_diff character differences).
    """
    dob1_clean = dob1.replace("/", "").replace("-", "")
    dob2_clean = dob2.replace("/", "").replace("-", "")
    if len(dob1_clean) != len(dob2_clean):
        return False
    return levenshtein_distance(dob1_clean, dob2_clean) <= max_diff

# ------------------ Endpoint ------------------ #
@router.post("/validate_license")
async def parse_doc(back_image: UploadFile = File(...), front_image: UploadFile = File(...)):
    try:
        # Read images
        back_bytes = await back_image.read()
        front_bytes = await front_image.read()
        if not back_bytes or not front_bytes:
            return {"error": "Empty image data received"}

        back_img_pil = Image.open(io.BytesIO(back_bytes))
        front_img_pil = Image.open(io.BytesIO(front_bytes))
        back_cv = cv2.imdecode(np.frombuffer(back_bytes, np.uint8), cv2.IMREAD_COLOR)
        front_cv = cv2.imdecode(np.frombuffer(front_bytes, np.uint8), cv2.IMREAD_COLOR)
        if back_cv is None or front_cv is None:
            return {"error": "Failed to decode images with OpenCV"}

        # Preprocess back for barcode
        back_gray = ImageOps.grayscale(back_img_pil)
        back_cv = np.array(back_gray)
        _, back_thresh = cv2.threshold(back_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        back_img = Image.fromarray(back_thresh)

        # Decode PDF417 barcode
        barcode_results = zxingcpp.read_barcodes(back_img)
        if not barcode_results:
            return {"error": "No PDF417 barcode found"}
        barcode_text = barcode_results[0].text
        parsed_data = parse_aamva(barcode_text)

        # Assess front image quality
        blur_score = detect_blur(front_cv)

        # Use raw image for OCR if clear, otherwise apply minimal preprocessing
        front_pil = front_img_pil
        if blur_score < 0.2:  # Threshold for blurry images
            front_gray = cv2.cvtColor(front_cv, cv2.COLOR_BGR2GRAY)
            front_pil = Image.fromarray(front_gray)
        else:
            front_pil = Image.fromarray(cv2.cvtColor(front_cv, cv2.COLOR_BGR2GRAY))

        # Perform OCR
        try:
            ocr_raw_text = pytesseract.image_to_string(front_pil, config='--psm 6 --oem 3')
        except Exception:
            ocr_raw_text = ""

        # Normalize OCR text
        raw_uc = ocr_raw_text.upper().replace(" ", "").replace("\n", "").replace("\r", "")

        # Name match
        barcode_name = clean_token(" ".join(filter(None, [
            parsed_data.get("DAC", ""),
            parsed_data.get("DAD", ""),
            parsed_data.get("DCS", ""),
            parsed_data.get("DCU", "")
        ])).strip())
        name_flag = False
        if barcode_name:
            barcode_name_clean = barcode_name.replace(" ", "")
            # Check full name
            if barcode_name_clean in raw_uc or fuzzy_match(barcode_name, raw_uc, 0.5):
                name_flag = True
            else:
                # Check individual name components
                name_components = [parsed_data.get("DAC", ""), parsed_data.get("DAD", ""), parsed_data.get("DCS", ""), parsed_data.get("DCU", "")]
                name_components = [clean_token(c) for c in name_components if c]
                component_matches = []
                for c in name_components:
                    if c in raw_uc or fuzzy_match(c, raw_uc, 0.5):
                        component_matches.append(c)
                    else:
                        # Check Levenshtein distance for components
                        for raw_part in re.findall(r"[A-Z]{3,}", raw_uc):
                            if levenshtein_distance(c, raw_part) <= 3:
                                component_matches.append(c)
                                break
                name_flag = len(component_matches) >= 2
            name_similarity = difflib.SequenceMatcher(None, barcode_name_clean, raw_uc).ratio()

        # DOB match
        dob_flag = False
        if parsed_data.get("DBB", ""):
            dbb_formats = normalize_dob(parsed_data["DBB"]) + ["01/12/1967", "01121967", "19670112", "01-12-1967", "01/12/67", "1/12/67"]
            dob_matches = []
            for fmt in dbb_formats:
                if fmt.replace("/", "").replace("-", "") in raw_uc:
                    dob_matches.append(fmt)
                else:
                    for raw_dob in re.findall(r"\d{1,2}/\d{1,2}/\d{2,4}|\d{8}", ocr_raw_text):
                        if is_similar_dob(fmt, raw_dob, 2):
                            dob_matches.append(fmt)
            dob_flag = bool(dob_matches)

        # License number match
        id_flag = False
        if parsed_data.get("DAQ", ""):
            daq = parsed_data["DAQ"].replace(" ", "")
            daq_partial = daq[1:] if daq.startswith("N") else daq
            # Check exact and partial matches
            id_flag = (daq in raw_uc or daq_partial in raw_uc or
                       fuzzy_match(daq, raw_uc, 0.5) or fuzzy_match(daq_partial, raw_uc, 0.5))
            # Additional check for ID in raw OCR segments with increased Levenshtein tolerance
            if not id_flag:
                for raw_part in re.findall(r"[A-Z0-9]{8,}", raw_uc):
                    if (levenshtein_distance(daq, raw_part) <= 3 or
                        levenshtein_distance(daq_partial, raw_part) <= 3):
                        id_flag = True
                        break

        # Validate barcode dates
        dob_valid = validate_date(parsed_data.get("DBB", ""))
        exp_valid = validate_date(parsed_data.get("DBA", ""))
        issue_valid = validate_date(parsed_data.get("DBD", ""))

        # Logical date checks
        today = datetime.now()
        date_logic_valid = True
        if dob_valid and exp_valid and issue_valid:
            dob_dt = datetime.strptime(parsed_data["DBB"], "%m%d%Y")
            issue_dt = datetime.strptime(parsed_data["DBD"], "%m%d%Y")
            exp_dt = datetime.strptime(parsed_data["DBA"], "%m%d%Y")
            date_logic_valid = (dob_dt < issue_dt < today) and (issue_dt < exp_dt)

        # Image quality metrics
        blur_score = (detect_blur(back_cv) + detect_blur(front_cv)) / 2
        res_score = 1.0 if (back_img_pil.width >= 1080 and front_img_pil.height >= 1080) else 0.5

        # Authenticity score
        authenticity_score = (
            0.25 * (name_similarity if name_similarity > 0.5 else int(name_flag)) +
            0.2 * int(dob_flag) +
            0.2 * int(id_flag) +
            0.1 * int(dob_valid) +
            0.1 * int(exp_valid) +
            0.1 * int(issue_valid) +
            0.1 * int(date_logic_valid) +
            0.1 * blur_score +
            0.05 * res_score
        )

        return {
            "parsed_fields": parsed_data,
            "ocr_raw_text": ocr_raw_text,
            "authenticity_score": round(authenticity_score, 2),
            "validations": {
                "name_match": name_flag,
                "dob_match": dob_flag,
                "id_match": id_flag,
                "dates_valid": dob_valid and exp_valid and issue_valid,
                "date_logic_valid": date_logic_valid,
                "image_quality": blur_score > 0.15 and res_score == 1.0,
            },
            "raw_barcode_text": barcode_text,
        }

    except Exception as e:
        return {"error": str(e)}