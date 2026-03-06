import cv2
import numpy as np
import pytesseract
from datetime import timedelta

def time_str_to_seconds(time_str):
    """Converts a HH:MM:SS string to total seconds."""
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        return None

def extract_time_from_frame(frame, roi, threshold_value, debug=False):
    """Extracts the timestamp from the ROI of a single frame."""
    y_start, y_end, x_start, x_end = roi
    
    # Isolate the region of interest (ROI)
    roi_frame = frame[y_start:y_end, x_start:x_end]
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold
    _, thresh_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Use Tesseract to do OCR on the processed image
    config = '--psm 7 -c tessedit_char_whitelist=0123456789:' 
    try:
        text = pytesseract.image_to_string(thresh_frame, config=config).strip()
    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract is not installed or not in your PATH.")
        return None

    # Debugging: Save the processed image
    if debug:
        cv2.imwrite(f'debug_ocr_{text.replace(":", "_")}.png', thresh_frame)

    # Basic validation and conversion to seconds
    if len(text) == 8 and text[2] == ':' and text[5] == ':':
        return time_str_to_seconds(text)

    return None

def find_timestamp_in_video(cap, target_seconds, roi, threshold_value, debug=False, search_range_seconds=300):
    """Efficiently searches for a target timestamp in a video.
    
    Args:
        cap: The OpenCV VideoCapture object.
        target_seconds (int): The timestamp to find, in seconds.
        roi (list): The [y_start, y_end, x_start, x_end] of the timestamp.
        threshold_value (int): The grayscale value for binary thresholding.
        debug (bool): Whether to save debugging images.
        search_range_seconds (int): The +/- range around the estimated frame to search.

    Returns:
        A tuple of (found_seconds, frame_number) or (None, None) if not found.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Estimate the frame number where the target time should be
    estimated_frame = int(target_seconds * fps)

    # Define a search range around the estimate
    search_start_frame = max(0, estimated_frame - int(search_range_seconds * fps))
    search_end_frame = min(total_frames, estimated_frame + int(search_range_seconds * fps))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, search_start_frame)
    
    current_frame_num = search_start_frame
    while current_frame_num < search_end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every Nth frame to speed up the search
        if current_frame_num % int(fps) == 0:
            extracted_seconds = extract_time_from_frame(frame, roi, threshold_value, debug)
            
            if extracted_seconds is not None:
                # If the extracted time is within a small margin of the target, we found it
                if abs(extracted_seconds - target_seconds) <= 1:
                    return extracted_seconds, current_frame_num
                
                # If we have jumped past the target time, we can stop searching
                if extracted_seconds > target_seconds + 5:
                    return None, None

        current_frame_num += 1
        
    return None, None
