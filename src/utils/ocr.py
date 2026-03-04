
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

        # 2. Apply a binary threshold. This is a crucial step for cleaning the image.
        # We assume light-colored text on a darker background. Pixels with a value
        # over 150 (out of 255) will be set to pure white. This isolates the text.
        _, thresh_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
        
        # 3. Invert the image. Tesseract OCR engine performs best with black text
        # on a clean white background. This step achieves that.
        inverted_frame = cv2.bitwise_not(thresh_frame)

        # 4. Use Tesseract to extract text using a specific character whitelist
        # to maximize accuracy and speed.
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:'
        text = pytesseract.image_to_string(inverted_frame, config=custom_config)
        
    except pytesseract.TesseractNotFoundError:
        # This error is fatal. It means the Tesseract installation is not found.
        print("\nFATAL: Tesseract is not installed or not in your system's PATH.")
        print("Please ensure Tesseract is installed and the path is correct in config.txt.\n")
        # Exit the program because no OCR can proceed.
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
    last_dt = datetime.combine(datetime.today(), last_time)
    current_dt = datetime.combine(datetime.today(), current_time)

    # Handle the overnight case where the time resets (e.g., from 23:59 to 00:00).
    if current_dt < last_dt:
        current_dt += timedelta(days=1)

    time_difference = (current_dt - last_dt).total_seconds()

    return time_difference > threshold_seconds
