
import os
import subprocess
import cv2
import pytesseract
import re
from datetime import datetime, timedelta
import numpy as np

def parse_time_from_ocr(text):
    """Extracts HH:MM:SS from OCR text using regex."""
    match = re.search(r'(\d{2}):(\d{2}):(\d{2})', text)
    if match:
        return match.group(0)
    return None

def time_str_to_seconds(time_str):
    """Converts a HH:MM:SS string to total seconds."""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def get_ocr_ready_frame(roi_frame):
    """
    Applies a new, advanced, multi-stage image processing pipeline designed for
    maximum OCR accuracy on difficult, noisy video timestamps. This is the definitive pipeline.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # 2. Upscale the image significantly (3x) to provide more detail for OCR
    upscaled = cv2.resize(gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 3. Apply a Median Blur to remove salt-and-pepper noise which is common in digital video
    blurred = cv2.medianBlur(upscaled, 3)

    # 4. Sharpen the image to make the edges of the digits more distinct using a sharpening kernel
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    # 5. Apply Otsu's thresholding to automatically find the optimal value to create a clean black-and-white image
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Ensure black text on a white background, which Tesseract performs best with.
    # If the image is mostly black (low mean pixel value), invert it.
    if cv2.mean(thresh)[0] < 128:
        final_image = cv2.bitwise_not(thresh)
    else:
        final_image = thresh

    return final_image

def process_video_file(video_path, output_folder, timestamp_roi, ocr_fluctuation_seconds, target_times, debug_ocr=False):
    """
    Processes a single video file to find target timestamps via OCR and trim one-minute clips.
    """
    normalized_video_path = os.path.normpath(video_path)

    cap = cv2.VideoCapture(normalized_video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {normalized_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"ERROR: Could not get FPS for video: {normalized_video_path}. Skipping.")
        return

    y1, y2, x1, x2 = timestamp_roi
    processed_targets = set()
    last_valid_time_seconds = -1

    # --- Tesseract Configuration ---
    # PSM 7: Treat the image as a single text line.
    tesseract_config = '--psm 7 -c tessedit_char_whitelist=0123456789:'

    print(f"INFO: Processing {os.path.basename(normalized_video_path)} with FPS: {fps:.2f}")
    if debug_ocr:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            roi_frame = frame[y1:y2, x1:x2]
            cv2.imwrite("debug_ocr_frame_raw.png", roi_frame)
            
            processed_frame_for_debug = get_ocr_ready_frame(roi_frame)
            
            debug_filename = "debug_ocr_frame_processed.png"
            cv2.imwrite(debug_filename, processed_frame_for_debug)
            print(f"INFO: DEBUG_OCR is True. Saved RAW and PROCESSED timestamp ROI to debug files.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_interval = int(fps)
    frame_num = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print("INFO: End of video file reached.")
            break

        roi_frame = frame[y1:y2, x1:x2]
        ocr_ready_frame = get_ocr_ready_frame(roi_frame)
        
        try:
            ocr_text = pytesseract.image_to_string(ocr_ready_frame, config=tesseract_config).strip()
            
            if ocr_text:
                print(f"DEBUG: Raw OCR output for frame #{frame_num}: '{ocr_text}'")

            current_time_str = parse_time_from_ocr(ocr_text)

            if current_time_str:
                current_time_seconds = time_str_to_seconds(current_time_str)

                if last_valid_time_seconds != -1 and abs(current_time_seconds - last_valid_time_seconds) > ocr_fluctuation_seconds:
                    print(f"WARN: OCR fluctuation detected. Read: {current_time_str}, Last Valid: {timedelta(seconds=last_valid_time_seconds)}. Skipping frame.")
                    frame_num += frame_interval
                    continue
                
                last_valid_time_seconds = current_time_seconds

                for target in target_times:
                    if target == current_time_str and target not in processed_targets:
                        print(f"INFO: Match found for {target} at frame {frame_num}!")
                        
                        start_time_seconds = frame_num / fps
                        output_filename = f"{os.path.splitext(os.path.basename(normalized_video_path))[0]}_{target.replace(':', '_')}.avi"
                        output_path = os.path.join(output_folder, output_filename)
                        
                        command = [
                            'ffmpeg', '-y', '-ss', str(start_time_seconds),
                            '-i', normalized_video_path,
                            '-t', '00:01:00', '-c:v', 'libx264', '-preset', 'medium',
                            '-crf', '23', '-c:a', 'aac', '-b:a', '192k', output_path
                        ]
                        
                        print(f"INFO: Trimming 1-minute clip for {target}...")
                        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        
                        print(f"SUCCESS: Saved trimmed clip to {output_path}")
                        processed_targets.add(target)
            
            if len(processed_targets) == len(target_times):
                print(f"INFO: All target times found for {os.path.basename(normalized_video_path)}. Moving to next video.")
                break

        except Exception as e:
            print(f"ERROR: An error occurred during OCR or trimming for frame {frame_num}. Details: {e}")

        frame_num += frame_interval

    cap.release()
    print(f"INFO: Finished processing {os.path.basename(normalized_video_path)}.")
