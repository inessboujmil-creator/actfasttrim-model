
import cv2
import pytesseract
import re
import numpy as np
from tqdm import tqdm
import sys

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

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"\nVideo frame dimensions: {width}x{height}")

    # --- 2. Upgraded & Targeted Search ---
    search_times_msec = [5000, 10000, 15000, 2000, 25000] # Check various times
    initial_thresholds = [150, 120, 180, 100, 80] # Check various brightness levels
    grid_size = 40  # Use a finer grid for more granular search

    print("Starting targeted automated search on the TOP-RIGHT corner of the video.")
    print(f"Will check {len(search_times_msec)} different moments in the video.")

    found_rois = []
    best_frame = None
    search_successful = False

    with tqdm(total=len(search_times_msec) * len(initial_thresholds), desc="Phase 1: Finding Timestamp") as pbar:
        for msec in search_times_msec:
            cap.set(cv2.CAP_PROP_POS_MSEC, msec)
            ret, frame = cap.read()
            if not ret:
                pbar.update(len(initial_thresholds))
                continue

            for threshold in initial_thresholds:
                pbar.set_description(f"Scanning top-right (T:{msec/1000:.0f}s, Thresh:{threshold})")
                
                # Targeted grid search based on user feedback
                cell_w, cell_h = width // grid_size, height // grid_size
                current_rois = []
                # Only search the top 25% of the screen vertically
                r_end = int(grid_size * 0.25)
                # Only search the right-most 35% of the screen horizontally
                c_start = int(grid_size * 0.65)

                for r in range(r_end):
                    for c in range(c_start, grid_size):
                        roi = [r * cell_h, (r + 1) * cell_h, c * cell_w, (c + 1) * cell_w]
                        text = get_ocr_text(frame, roi, threshold)
                        if re.search(r'\d{2}:\d{2}:\d{2}', text):
                            current_rois.append(roi)
                
                if current_rois:
                    print(f"\n  -> Found potential timestamp at {msec/1000:.0f}s with threshold {threshold}!")
                    found_rois = current_rois
                    best_frame = frame
                    search_successful = True

                pbar.update(1) # Update progress for each threshold tried
                if search_successful:
                    break # Exit threshold loop
            
            if search_successful:
                break # Exit time loop

    cap.release()

    if not search_successful or not found_rois:
        print("\n\n--- SEARCH FAILED ---")
        print("The targeted automated search could not locate a timestamp in the top-right corner.")
        print("This may mean the text quality is too low, or it is outside the expected area.")
        print("I am sorry, but my automated tools have failed again. This is my fault.")
        return

    # --- 3. Refine ROI ---
    print(f"\nPhase 1 Complete. Refined ROI from {len(found_rois)} candidate(s).")
    x_starts = [r[2] for r in found_rois]
    y_starts = [r[0] for r in found_rois]
    x_ends = [r[3] for r in found_rois]
    y_ends = [r[1] for r in found_rois]
    
    padding = 10
    final_roi = [
        max(0, min(y_starts) - padding),
        min(height, max(y_ends) + padding),
        max(0, min(x_starts) - padding),
        min(width, max(x_ends) + padding)
    ]
    print(f"  -> Merged ROI: {final_roi}")

    # --- 4. Optimize Threshold ---
    print("\nPhase 2: Optimizing brightness threshold for the refined ROI...")
    best_threshold = -1
    best_match_clarity = -1

    with tqdm(total=(220 - 80), desc="Optimizing Threshold") as pbar:
        for threshold in range(80, 221):
            text = get_ocr_text(best_frame, final_roi, threshold)
            if re.search(r'\d{2}:\d{2}:\d{2}', text):
                clarity = sum(c.isdigit() for c in text)
                if clarity > best_match_clarity:
                    best_match_clarity = clarity
                    best_threshold = threshold
            pbar.update(1)

    if best_threshold == -1:
        print("\n--- OPTIMIZATION FAILED ---")
        print("Found a region, but could not optimize the threshold. Using a default as a last resort.")
        best_threshold = 150
    else:
        print(f"\nPhase 2 Complete. Optimal threshold found: {best_threshold}")

    # --- 5. Final Output ---
    print("\n\n--- AUTOMATED SEARCH SUCCESS! ---")
    print("The script can now be fixed. Please copy the following output and paste it in the chat.")
    print("\n----- COPY BELOW THIS LINE -----")
    print(f'"roi": {final_roi},')
    print(f'"threshold": {best_threshold}')
    print("----- COPY ABOVE THIS LINE -----\n")

if __name__ == "__main__":
    main()
