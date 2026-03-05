
import cv2
import pytesseract
import re
import sys
from tqdm import tqdm
import numpy as np

def save_debug_image(frame, video_path):
    """Saves a debug image to help diagnose OCR failures."""
    try:
        debug_image_path = "debug_image.png"
        # Add text to the image explaining what it is
        cv2.putText(frame, "AUTOMATED SEARCH FAILED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        cv2.putText(frame, "Gemini was unable to find the timestamp automatically.", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, f"Please show this image to Gemini to resolve the issue.", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite(debug_image_path, frame)
        print(f"\n--- DIAGNOSTIC FILE CREATED ---")
        print(f"I have saved a file named '{debug_image_path}'.")
        print("Please show me this image so I can finalize the solution.")
    except Exception as e:
        print(f"\nERROR: Could not save the debug image. {e}")

def get_ocr_text(frame, roi, threshold):
    """Performs OCR on a given ROI."""
    try:
        y_start, y_end, x_start, x_end = roi
        roi_frame = frame[y_start:y_end, x_start:x_end]
        
        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        _, thresh_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
        inverted_frame = cv2.bitwise_not(thresh_frame)

        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:'
        text = pytesseract.image_to_string(inverted_frame, config=config)
        return text.strip()
    except Exception:
        return ""

def main():
    """Main function to automatically find the timestamp ROI and threshold."""
    video_path_input = input("\n>>> Please enter the full path to one of your video files and press Enter: ")
    video_path = video_path_input.strip().strip('\"')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nERROR: Could not open video file: {video_path}")
        return

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"\nVideo dimensions: {width}x{height}")

    # --- Aggressively targeted search based on user-provided images ---
    search_area = [
        0,              # y_start = top of the screen
        int(height * 0.15), # y_end = 15% down from the top
        int(width * 0.70), # x_start = 70% from the left
        width           # x_end = right edge of the screen
    ]
    print(f"Starting a rapid, targeted search in the top-right corner: {search_area}")

    search_times_msec = [5000, 10000, 2000] 
    thresholds = range(100, 221, 15) # Test a wide range of thresholds
    found_rois = []
    best_frame = None
    
    with tqdm(total=len(search_times_msec) * len(thresholds), desc="Finding Timestamp") as pbar:
        for msec in search_times_msec:
            cap.set(cv2.CAP_PROP_POS_MSEC, msec)
            ret, frame = cap.read()
            if not ret:
                pbar.update(len(thresholds))
                continue

            crop_y_start, crop_y_end, crop_x_start, crop_x_end = search_area
            cropped_frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

            for t in thresholds:
                text = get_ocr_text(cropped_frame, [0, cropped_frame.shape[0], 0, cropped_frame.shape[1]], t)
                if re.search(r'\d{2}:\d{2}:\d{2}', text):
                    print(f"\n  -> Potential timestamp found at {msec/1000:.0f}s with threshold {t}!")
                    # Since we are using a tight search area, we can use it as the final ROI
                    final_roi = search_area
                    # Now we just need to optimize the threshold
                    best_frame = frame
                    found_rois.append(final_roi)
                    break
                pbar.update(1)
            if found_rois:
                break

    if not found_rois:
        print("\n\n--- AUTOMATED SEARCH FAILED ---")
        print("My automated tool was still unable to find the timestamp.")
        cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
        ret, frame = cap.read()
        if ret:
            save_debug_image(frame, video_path)
        cap.release()
        return

    # --- Refine Threshold --- 
    print("\nOptimizing brightness threshold...")
    final_roi = found_rois[0]
    best_threshold = -1
    best_clarity = -1
    with tqdm(total=len(range(80, 231)), desc="Optimizing Threshold") as pbar:
        for t in range(80, 231):
            text = get_ocr_text(best_frame, final_roi, t)
            if re.search(r'\d{2}:\d{2}:\d{2}', text):
                clarity = sum(c.isdigit() for c in text)
                if clarity > best_clarity:
                    best_clarity = clarity
                    best_threshold = t
            pbar.update(1)

    cap.release()

    if best_threshold == -1:
        best_threshold = 150 # Fallback
        print("\nCould not optimize threshold, using fallback value.")

    print("\n\n--- AUTOMATED SEARCH SUCCESS! ---")
    print("Please copy the following output and paste it in the chat.")
    print("\n----- COPY BELOW THIS LINE -----")
    print(f'"roi": {final_roi},')
    print(f'"threshold": {best_threshold}')
    print("----- COPY ABOVE THIS LINE -----\n")

if __name__ == "__main__":
    main()
