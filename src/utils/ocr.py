
import re
from datetime import datetime
import cv2
try:
    import pytesseract
except ImportError:
    print("Pytesseract not found. Please install it using: pip install pytesseract")
    exit()

def extract_timestamp_from_frame(frame, roi_coords):
    """
    Performs OCR on a specific region of a video frame to find a timestamp.
    """
    try:
        # Crop the frame to the Region of Interest (ROI)
        roi = frame[roi_coords[0]:roi_coords[1], roi_coords[2]:roi_coords[3]]
        # Convert to grayscale for better OCR performance
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Use Tesseract to find text, configured for a single line of text
        text = pytesseract.image_to_string(gray_roi, config='--psm 7').strip()

        # Search for a HH:MM:SS pattern in the extracted text
        match = re.search(r"(\d{2}:\d{2}:\d{2})", text)
        if match:
            return match.group(1)
            
    except Exception as e:
        print(f"ERROR: An error occurred during OCR: {e}")
        
    return None

def is_time_fluctuation(time_str1, time_str2, fluctuation_seconds):
    """
    Checks if the difference between two time strings exceeds a defined threshold,
    indicating an illogical jump.
    """
    try:
        t1 = datetime.strptime(time_str1, '%H:%M:%S')
        t2 = datetime.strptime(time_str2, '%H:%M:%S')
        delta = abs((t2 - t1).total_seconds())
        # A fluctuation is a jump greater than the threshold but not a day-wrap
        return delta > fluctuation_seconds and delta < 86300 # 23h 58m 20s
    except ValueError:
        return True # Treat parsing errors as fluctuations
