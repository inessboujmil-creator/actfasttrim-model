
import os
import subprocess
import cv2
import pytesseract
from datetime import datetime
import imageio_ffmpeg

from .ocr import get_ocr_ready_frame, parse_time_from_ocr, time_str_to_time_obj

def process_video_file(video_path, output_folder, timestamp_roi, ocr_threshold, ocr_fluctuation_seconds, target_times, debug_ocr=False):
    """
    Processes a single video file using robust, interval-based timestamp matching and reliable re-encoding for trims.
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

    target_time_objs = sorted([time_str_to_time_obj(t) for t in target_times if time_str_to_time_obj(t) is not None])

    y1, y2, x1, x2 = timestamp_roi
    processed_targets = set()
    previous_time_obj = None

    tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:'

    print(f"INFO: Processing {os.path.basename(normalized_video_path)} with FPS: {fps:.2f}")
    if debug_ocr:
        pass

    frame_interval = int(fps)
    frame_num = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
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

                is_valid_interval, is_midnight_cross = check_time_interval(previous_time_obj, current_time_obj, ocr_fluctuation_seconds)

                if is_valid_interval:
                    for target_obj in target_time_objs:
                        if str(target_obj) in processed_targets:
                            continue
                        
                        if has_crossed_target(previous_time_obj, current_time_obj, target_obj, is_midnight_cross):
                            print(f"INFO: Match found for {target_obj.strftime('%H:%M:%S')}!")
                            start_seconds = frame_num / fps
                            output_filename = f"{os.path.splitext(os.path.basename(normalized_video_path))[0]}_{target_obj.strftime('%H_%M_%S')}.mp4"
                            output_path = os.path.join(output_folder, output_filename)
                            
                            # --- Use the reliable re-encoding method --- #
                            trim_video_with_reencode(normalized_video_path, output_path, start_seconds, 60)

                            processed_targets.add(str(target_obj))
                
                previous_time_obj = current_time_obj
            
            if len(processed_targets) == len(target_times):
                break

        except Exception as e:
            print(f"ERROR: An error occurred during video processing. Details: {e}")

        frame_num += frame_interval

    cap.release()

def check_time_interval(prev_time, curr_time, fluctuation_seconds):
    is_valid = True
    is_midnight_cross = False
    delta_seconds = (datetime.combine(datetime.min, curr_time) - datetime.combine(datetime.min, prev_time)).total_seconds()

    if delta_seconds < 0:
        if prev_time.hour == 23 and curr_time.hour == 0:
            is_midnight_cross = True
            delta_seconds += 86400 # Add a day's seconds
        else:
            is_valid = False
            print(f"WARN: OCR fluctuation. Time jumped backward from {prev_time} to {curr_time}. Skipping.")

    if abs(delta_seconds) > fluctuation_seconds and not is_midnight_cross:
         is_valid = False
         print(f"WARN: OCR fluctuation. Time jumped too far from {prev_time} to {curr_time}. Skipping.")
    
    return is_valid, is_midnight_cross

def has_crossed_target(prev_time, curr_time, target_time, is_midnight_cross):
    if is_midnight_cross:
        return target_time > prev_time or target_time <= curr_time
    else:
        return prev_time < target_time <= curr_time

def trim_video_with_reencode(input_path, output_path, start_time, duration):
    """
    Trims a video using a reliable re-encoding FFmpeg call.
    """
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe, '-y', '-ss', str(start_time), '-i', input_path,
            '-t', str(duration), '-c:v', 'libx264', '-preset', 'medium',
            '-c:a', 'aac', output_path
        ]
        
        print(f"INFO: Trimming 1-minute clip for {os.path.basename(output_path)}... (Re-encoding for reliability)")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"SUCCESS: Saved trimmed clip to {output_path}")

    except subprocess.CalledProcessError as e:
        print("--- FFMPEG ERROR ---")
        print(f"Command failed with exit code {e.returncode} when creating {output_path}")
        print("STDERR:", e.stderr)
    except FileNotFoundError:
        print("[ERROR] ffmpeg executable not found via imageio-ffmpeg. Please ensure ffmpeg is installed.")

