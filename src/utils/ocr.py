
import pytesseract
import cv2
import re
import numpy as np
from datetime import datetime, time, timedelta

def extract_timestamp_from_frame(frame, roi):
    y, h, x, w = roi
    roi_frame = frame[y:h, x:w]

    # --- Start of new debug code ---
    # Save the exact frame being processed for manual inspection.
    cv2.imwrite("debug_frame.png", roi_frame)
    # --- End of new debug code ---

    try:
        # Pre-processing steps from previous attempts
        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        inverted_frame = cv2.bitwise_not(gray_frame)
        
        # Use Tesseract to extract text
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:'
        text = pytesseract.image_to_string(inverted_frame, config=custom_config)
        
    # --- Start of new explicit error handling ---
    except pytesseract.TesseractNotFoundError as e:
        print(f"\n\n--- OCR ERROR ---")
        print(f"FATAL: Tesseract is not installed or not in your PATH/environment variable.")
        print(f"Full Error: {e}")
        print(f"-----------------\n\n")
        return None
    except Exception as e:
        print(f"\n\n--- OCR ERROR ---")
        print(f"An unexpected error occurred during OCR text extraction.")
        print(f"Full Error: {e}")
        print(f"-----------------\n\n")
        return None
    # --- End of new explicit error handling ---

    # Search for a timestamp in the format HH:MM:SS
    match = re.search(r'(\d{2}):(\d{2}):(\d{2})', text)
    if match:
        try:
            h, m, s = map(int, match.groups())
            if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                return time(h, m, s)
        except ValueError:
            return None
    return None

def is_time_fluctuation(last_time, current_time, threshold_seconds):
    last_dt = datetime.combine(datetime.today(), last_time)
    current_dt = datetime.combine(datetime.today(), current_time)

    # Handle overnight case
    if current_dt < last_dt:
        current_dt += timedelta(days=1)

    time_difference = (current_dt - last_dt).total_seconds()

    return time_difference > threshold_seconds
