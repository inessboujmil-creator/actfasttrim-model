
import cv2
import pytesseract
import re
import numpy as np

def get_ocr_ready_frame(roi_frame, threshold_value):
    """
    Converts a frame region (ROI) into a clean, black-and-white image optimized for OCR.
    This pipeline is crucial for getting accurate text from the video frames.
    """
    # 1. Convert to Grayscale: Tesseract works best on grayscale images.
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # 2. Invert Colors: If the text is white on a dark background, inverting it to black text on a white background improves recognition.
    inverted = cv2.bitwise_not(gray)

    # 3. Apply Binary Thresholding: This is the most critical step.
    # It converts the image to pure black and white, eliminating noise and compression artifacts.
    # The 'threshold_value' is loaded from config.txt, which you can tune using find_roi_auto.py.
    _, final_image = cv2.threshold(inverted, threshold_value, 255, cv2.THRESH_BINARY)

    # 4. (Optional) Upscale for Clarity: Sometimes, making the image larger can help Tesseract.
    # final_image = cv2.resize(final_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return final_image

def parse_time_from_ocr(text):
    """Extracts HH:MM:SS from OCR text using regex."""
    # This regex is designed to find a pattern of two digits, a colon, two digits, a colon, and two digits.
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
        # Handles cases where time_str is None or malformed
        return -1
