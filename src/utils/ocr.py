
import cv2
import pytesseract
import re
from datetime import datetime

def extract_timestamp_from_frame(frame, roi_coords, debug=False):
    """
    Extracts a timestamp from a specific region of interest (ROI) in a video frame.

    Args:
        frame: The video frame (as a NumPy array) to process.
        roi_coords: A list [y1, y2, x1, x2] defining the ROI.
        debug (bool): If True, saves the processed OCR frame for inspection.

    Returns:
        A tuple containing:
        - A datetime.time object if a valid timestamp is found, otherwise None.
        - The raw, cleaned text extracted by OCR.
    """
    try:
        # 1. Crop the frame to the region of interest.
        y1, y2, x1, x2 = roi_coords
        roi = frame[y1:y2, x1:x2]

        # 2. Convert the ROI to grayscale for better OCR accuracy.
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 3. Apply adaptive thresholding to create a binary image.
        thresh_frame = cv2.adaptiveThreshold(
            gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # --- DEBUGGING --- #
        if debug:
            debug_filename = 'debug_ocr_frame.png'
            cv2.imwrite(debug_filename, thresh_frame)
            print(f"\n--- OCR DEBUG MODE ---")
            print(f"  - Saved the processed frame for OCR to '{debug_filename}'.")
            print(f"  - Inspect this image to verify the ROI and image quality.")
            print(f"  - Set DEBUG_OCR to False in config.txt for normal operation.")
            exit() # Stop the script after saving the debug image
            
        # 4. Use Tesseract to extract text.
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:'
        text = pytesseract.image_to_string(thresh_frame, config=custom_config)

    except pytesseract.TesseractNotFoundError:
        print("\nFATAL: Tesseract is not installed or not in your system's PATH.")
        print("Please ensure Tesseract is installed and the path is correct in config.txt.\n")
        exit()
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during OCR image processing: {e}\n")
        return None, ""

    cleaned_text = text.strip()
    match = re.search(r'(\d{2}:\d{2}:\d{2})', cleaned_text)

    if match:
        time_str = match.group(1)
        try:
            parsed_time = datetime.strptime(time_str, '%H:%M:%S').time()
            return parsed_time, cleaned_text
        except ValueError:
            return None, cleaned_text
    else:
        return None, cleaned_text

def is_time_fluctuation(last_time, current_time, fluctuation_seconds):
    today = datetime.now().date()
    dt1 = datetime.combine(today, last_time)
    dt2 = datetime.combine(today, current_time)
    time_difference = abs((dt2 - dt1).total_seconds())
    if time_difference > fluctuation_seconds and time_difference < 86300:
        return True
    return False
