
import os
import subprocess
import cv2
import pytesseract
from datetime import datetime, timedelta

# Ensure we are using the corrected OCR utilities
from .ocr import get_ocr_ready_frame, parse_time_from_ocr, time_str_to_seconds

def process_video_file(video_path, output_folder, timestamp_roi, ocr_threshold, ocr_fluctuation_seconds, target_times, debug_ocr=False):
    """
    Processes a single video file, now passing the crucial ocr_threshold to the OCR utility.
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

    # Standard Tesseract configuration for reading segmented text like a clock
    tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:'

    print(f"INFO: Processing {os.path.basename(normalized_video_path)} with FPS: {fps:.2f}")
    # Debugging block to save a sample of what OCR is seeing
    if debug_ocr:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            roi_frame = frame[y1:y2, x1:x2]
            cv2.imwrite("debug_ocr_frame_raw.png", roi_frame)
            # Pass the threshold to the debug frame generation
            processed_frame_for_debug = get_ocr_ready_frame(roi_frame, ocr_threshold)
            cv2.imwrite("debug_ocr_frame_processed.png", processed_frame_for_debug)
            print(f"INFO: DEBUG_OCR is True. Saved RAW and PROCESSED timestamp ROI to debug files.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset frame position

    frame_interval = int(fps)  # Check one frame per second
    frame_num = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print("INFO: End of video file reached.")
            break

        # 1. Extract the timestamp area from the frame
        roi_frame = frame[y1:y2, x1:x2]
        # 2. Process the ROI to make it OCR-ready, using the new threshold
        ocr_ready_frame = get_ocr_ready_frame(roi_frame, ocr_threshold)
        
        try:
            # 3. Perform OCR on the cleaned image
            ocr_text = pytesseract.image_to_string(ocr_ready_frame, config=tesseract_config).strip()
            current_time_str = parse_time_from_ocr(ocr_text)

            if current_time_str:
                # This is the "Intelligent Fluctuation Detection"
                current_time_seconds = time_str_to_seconds(current_time_str)

                if last_valid_time_seconds != -1 and abs(current_time_seconds - last_valid_time_seconds) > ocr_fluctuation_seconds:
                    print(f"WARN: OCR fluctuation detected. Read '{current_time_str}' but expected near {last_valid_time_seconds}. Skipping frame.")
                    frame_num += frame_interval
                    continue # Skip this unreliable reading
                
                last_valid_time_seconds = current_time_seconds

                # 4. Check if the valid time matches any of our targets
                for target in target_times:
                    if target == current_time_str and target not in processed_targets:
                        print(f"INFO: Match found for {target} at frame {frame_num}!")
                        
                        start_time_seconds = frame_num / fps
                        output_filename = f"{os.path.splitext(os.path.basename(normalized_video_path))[0]}_{target.replace(':', '_')}.mp4"
                        output_path = os.path.join(output_folder, output_filename)
                        
                        command = [
                            'ffmpeg', '-y', '-ss', str(start_time_seconds),
                            '-i', normalized_video_path,
                            '-t', '00:01:00', '-c', 'copy', output_path
                        ]
                        
                        print(f"INFO: Trimming 1-minute clip for {target}...")
                        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        print(f"SUCCESS: Saved trimmed clip to {output_path}")
                        processed_targets.add(target)
            
            elif ocr_text:
                 print(f"DEBUG: OCR found invalid text for frame #{frame_num}: '{ocr_text}'")

            if len(processed_targets) == len(target_times):
                print(f"INFO: All target times found for this video. Moving to next.")
                break

        except Exception as e:
            print(f"ERROR: An error occurred during video processing. Details: {e}")

        frame_num += frame_interval

    cap.release()
    print(f"INFO: Finished processing {os.path.basename(normalized_video_path)}.")
