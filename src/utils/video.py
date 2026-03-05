
import os
import subprocess
import cv2
import pytesseract
from datetime import datetime, timedelta

# Import the new time object conversion function
from .ocr import get_ocr_ready_frame, parse_time_from_ocr, time_str_to_time_obj

def process_video_file(video_path, output_folder, timestamp_roi, ocr_threshold, ocr_fluctuation_seconds, target_times, debug_ocr=False):
    """
    Processes a single video file using robust, interval-based timestamp matching.
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

    # Convert string target times to time objects for comparison
    target_time_objs = [time_str_to_time_obj(t) for t in target_times]
    target_time_objs.sort()

    y1, y2, x1, x2 = timestamp_roi
    processed_targets = set()
    # --- Use time objects for tracking --- 
    previous_time_obj = None

    tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:'

    print(f"INFO: Processing {os.path.basename(normalized_video_path)} with FPS: {fps:.2f}")
    if debug_ocr:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            roi_frame = frame[y1:y2, x1:x2]
            cv2.imwrite("debug_ocr_frame_raw.png", roi_frame)
            processed_frame_for_debug = get_ocr_ready_frame(roi_frame, ocr_threshold)
            cv2.imwrite("debug_ocr_frame_processed.png", processed_frame_for_debug)
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
        ocr_ready_frame = get_ocr_ready_frame(roi_frame, ocr_threshold)
        
        try:
            ocr_text = pytesseract.image_to_string(ocr_ready_frame, config=tesseract_config).strip()
            current_time_str = parse_time_from_ocr(ocr_text)
            current_time_obj = time_str_to_time_obj(current_time_str)

            if current_time_obj:
                if previous_time_obj is None:
                    previous_time_obj = current_time_obj
                    frame_num += frame_interval
                    continue

                # --- Inspired Fluctuation Logic --- #
                is_valid_interval = True
                # Convert to seconds since midnight for easy comparison
                current_seconds = current_time_obj.hour * 3600 + current_time_obj.minute * 60 + current_time_obj.second
                previous_seconds = previous_time_obj.hour * 3600 + previous_time_obj.minute * 60 + previous_time_obj.second
                delta_seconds = current_seconds - previous_seconds

                is_midnight_cross = False
                if delta_seconds < 0:
                    # Plausible if previous was late at night and current is early morning
                    if previous_time_obj.hour == 23 and current_time_obj.hour == 0:
                        is_midnight_cross = True
                    else:
                        is_valid_interval = False
                        print(f"WARN: OCR fluctuation. Time jumped backward from {previous_time_obj} to {current_time_obj}. Skipping.")

                if abs(delta_seconds) > ocr_fluctuation_seconds and not is_midnight_cross:
                     is_valid_interval = False
                     print(f"WARN: OCR fluctuation. Time jumped too far from {previous_time_obj} to {current_time_obj}. Skipping.")
                # --- End Fluctuation Logic --- #

                if is_valid_interval:
                    for target_obj in target_time_objs:
                        if str(target_obj) in processed_targets:
                            continue
                        
                        # --- Inspired Interval Matching Logic --- #
                        regular_cross = (previous_time_obj < target_obj <= current_time_obj) and not is_midnight_cross
                        midnight_cross_check = is_midnight_cross and (target_obj > previous_time_obj or target_obj <= current_time_obj)

                        if regular_cross or midnight_cross_check:
                            print(f"INFO: Match found for {target_obj.strftime('%H:%M:%S')}! Timestamp crossed between checks.")
                            
                            start_seconds = frame_num / fps
                            output_filename = f"{os.path.splitext(os.path.basename(normalized_video_path))[0]}_{target_obj.strftime('%H_%M_%S')}.mp4"
                            output_path = os.path.join(output_folder, output_filename)
                            
                            command = [
                                'ffmpeg', '-y', '-ss', str(start_seconds),
                                '-i', normalized_video_path,
                                '-t', '00:01:00', '-c', 'copy', output_path
                            ]
                            
                            print(f"INFO: Trimming 1-minute clip for {target_obj.strftime('%H:%M:%S')}...")
                            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            
                            print(f"SUCCESS: Saved trimmed clip to {output_path}")
                            processed_targets.add(str(target_obj))
                
                previous_time_obj = current_time_obj
            
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
