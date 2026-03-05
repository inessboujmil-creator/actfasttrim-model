
import cv2
import pytesseract
import re
import numpy as np
from datetime import datetime

def get_ocr_ready_frame(roi_frame, threshold_value):
    """
    Converts a frame region (ROI) into a clean, black-and-white image optimized for OCR.
    """
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    _, final_image = cv2.threshold(inverted, threshold_value, 255, cv2.THRESH_BINARY)
    return final_image

def parse_time_from_ocr(text):
    """Extracts HH:MM:SS from OCR text using regex."""
    match = re.search(r'(\d{2}):(\d{2}):(\d{2})', text)
    if match:
        return match.group(0)
    return None

def time_str_to_seconds(time_str):
    """Converts a HH:MM:SS string to total seconds for easy comparison."""
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return -1

def time_str_to_time_obj(time_str):
    """Converts a HH:MM:SS string to a datetime.time object for robust comparisons."""
    try:
        return datetime.strptime(time_str, "%H:%M:%S").time()
    except (ValueError, TypeError):
        return None
