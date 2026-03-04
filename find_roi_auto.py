
import cv2
import pytesseract
import re
import numpy as np
from tqdm import tqdm

def get_ocr_text(frame, roi, threshold):
    """Helper function to perform OCR on a given ROI with a specific threshold."""
    try:
        y_start, y_end, x_start, x_end = roi
        roi_frame = frame[y_start:y_end, x_start:x_end]

        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        _, thresh_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
        inverted_frame = cv2.bitwise_not(thresh_frame)

        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:'
        text = pytesseract.image_to_string(inverted_frame, config=custom_config)
        return text.strip()
    except Exception:
        return ""

def main():
    """Main function to automatically find the best ROI and threshold for OCR."""
    # --- 1. Get Video Path ---
    video_path_input = input("\n>>> Please enter the full path to one of your video files and press Enter: ")
    video_path = video_path_input.strip().strip('\"')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nERROR: Could not open video file: {video_path}")
        return

    # --- 2. Get a Sample Frame ---
    cap.set(cv2.CAP_PROP_POS_MSEC, 5000) # Use a frame 5 seconds in
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("\nERROR: Could not read a frame from the video.")
        return

    height, width, _ = frame.shape
    print(f"\nVideo frame dimensions: {width}x{height}")
    print("Starting automated search for the timestamp. This may take a minute...")

    # --- 3. Coarse Search for ROI --- 
    # Divide the frame into a grid and search each cell for a timestamp.
    grid_size = 15 # Search in a 15x15 grid
    cell_w, cell_h = width // grid_size, height // grid_size
    found_rois = []
    initial_threshold = 150 # A reasonable starting threshold

    print("\nPhase 1: Searching for candidate regions...")
    with tqdm(total=grid_size*grid_size) as pbar:
        for r in range(grid_size):
            for c in range(grid_size):
                y_start, x_start = r * cell_h, c * cell_w
                y_end, x_end = y_start + cell_h, x_start + cell_w
                roi = [y_start, y_end, x_start, x_end]

                text = get_ocr_text(frame, roi, initial_threshold)
                if re.search(r'\d{2}:\d{2}:\d{2}', text):
                    print(f"  -> Found potential timestamp in region: {roi}")
                    found_rois.append(roi)
                pbar.update(1)

    if not found_rois:
        print("\n--- SEARCH FAILED ---")
        print("Could not automatically locate a timestamp in the video frame.")
        print("This can happen if the text is very unclear, very small, or not present at the 5-second mark.")
        print("Please try running the manual `roi_finder.py` script as a fallback.")
        return

    # --- 4. Refine ROI --- 
    # Merge overlapping candidate ROIs into one larger ROI
    if not found_rois:
        # This case is already handled above, but as a safeguard:
        print("Error: No ROIs found after search phase.")
        return

    x_starts = [r[2] for r in found_rois]
    y_starts = [r[0] for r in found_rois]
    x_ends = [r[3] for r in found_rois]
    y_ends = [r[1] for r in found_rois]
    
    # Add a small padding to the merged ROI
    padding = 5 
    final_roi = [
        max(0, min(y_starts) - padding),
        min(height, max(y_ends) + padding),
        max(0, min(x_starts) - padding),
        min(width, max(x_ends) + padding)
    ]
    print(f"\nPhase 1 Complete. Refined ROI: {final_roi}")

    # --- 5. Optimize Threshold --- 
    print("\nPhase 2: Optimizing brightness threshold for the refined ROI...")
    best_threshold = -1
    best_match_clarity = -1 # Higher is better (more digits recognized)

    # Test a range of thresholds to find the best one
    with tqdm(total=(190 - 110)) as pbar:
        for threshold in range(110, 191): # Test thresholds from 110 to 190
            text = get_ocr_text(frame, final_roi, threshold)
            if re.search(r'\d{2}:\d{2}:\d{2}', text):
                # A simple clarity metric: how many digits are in the string?
                clarity = sum(c.isdigit() for c in text)
                if clarity > best_match_clarity:
                    best_match_clarity = clarity
                    best_threshold = threshold
            pbar.update(1)

    if best_threshold == -1:
        print("\n--- OPTIMIZATION FAILED ---")
        print("Found a region that might contain a timestamp, but could not optimize the threshold.")
        print("The text might be too distorted. Using default values as a fallback.")
        best_threshold = 150 # Fallback to default
    else:
        print(f"\nPhase 2 Complete. Optimal threshold found: {best_threshold}")

    # --- 6. Final Output ---
    print("\n\n--- AUTOMATED SEARCH SUCCESS! ---")
    print("The script can now be fixed. Please copy the following output and paste it in the chat.")
    print("\n----- COPY BELOW THIS LINE -----")
    print(f'"roi": {final_roi},')
    print(f'"threshold": {best_threshold}')
    print("----- COPY ABOVE THIS LINE -----\n")

if __name__ == "__main__":
    main()
