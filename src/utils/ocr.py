
import pytesseract
import cv2
import re
import numpy as np
from datetime import datetime, time, timedelta

def extract_timestamp_from_frame(frame, roi):
    """
    Extracts a timestamp from a specified region of interest (ROI) in a video frame
    using a robust image pre-processing pipeline for OCR.
    """
    y, h, x, w = roi
    roi_frame = frame[y:h, x:w]

    try:
        # 1. Convert ROI to grayscale
        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # 2. Apply a Gaussian blur to reduce noise
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # 3. Apply adaptive thresholding. This is more robust to varying lighting
        # than a fixed global threshold. It creates black text on a white background.
        thresh_frame = cv2.adaptiveThreshold(
            blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 4
        )

        # 4. Use Tesseract to extract text.
        # --oem 3: Default OCR Engine mode.
        # --psm 7: Treat the image as a single text line.
        # -c tessedit_char_whitelist: Restrict to numbers and colons for accuracy.
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:'
        text = pytesseract.image_to_string(thresh_frame, config=custom_config)

    except pytesseract.TesseractNotFoundError:
        # This error is fatal. It means the Tesseract installation is not found.
        print("\nFATAL: Tesseract is not installed or not in your system's PATH.")
        print("Please ensure Tesseract is installed and the path is correct in config.txt.\n")
        exit()
    except Exception as e:
        # Handle other unexpected errors during the OCR process.
        print(f"\nERROR: An unexpected error occurred during OCR image processing: {e}\n")
        return None

    # 5. Use a regular expression to find a valid timestamp in the extracted text.
    match = re.search(r'(\d{2}):(\d{2}):(\d{2})', text)
    if match:
        try:
            h, m, s = map(int, match.groups())
            # Validate that the parsed numbers form a real time.
            if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                return time(h, m, s)
        except ValueError:
            # This handles cases where the regex matches but conversion to int fails.
            return None
    return None

def is_time_fluctuation(last_time, current_time, threshold_seconds):
    """
    Checks if the difference between two time objects suggests a video file loop
    or a significant jump in time.
    """
    # If either time is missing, it's not a fluctuation.
    if not last_time or not current_time:
        return False

    last_dt = datetime.combine(datetime.today(), last_time)
    current_dt = datetime.combine(datetime.today(), current_time)

    # Handle the overnight case where the time resets (e.g., from 23:59 to 00:00).
    if current_dt < last_dt:
        current_dt += timedelta(days=1)

    time_difference = (current_dt - last_dt).total_seconds()

    # A fluctuation is a jump forward that is too large (e.g., a misread from 01:00 to 05:00)
    return time_difference > threshold_seconds
