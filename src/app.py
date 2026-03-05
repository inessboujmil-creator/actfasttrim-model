
import os
import time
import cv2
from datetime import datetime, timedelta, time as dt_time

from .utils.video import trim_video_clip
from .utils.ocr import extract_timestamp_from_frame, is_time_fluctuation

def process_video_file(video_path, output_folder, roi, fluctuation_seconds, debug_ocr=False):
    """
    Processes a single video file to find and trim clips based on timestamp jumps.
    This function now includes detailed, frame-by-frame logging and an OCR debug mode.
    """
    print(f"\nPROCESSING: '{os.path.basename(video_path)}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return 0

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        print(f"WARNING: Could not determine frame rate for {video_path}. Assuming 30 FPS.")
        frame_rate = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  - Video Info: {total_frames} frames at {frame_rate:.2f} FPS.")
    
    # If debug mode is on, process only the first frame and exit.
    if debug_ocr:
        ret, frame = cap.read()
        if ret:
            # This call will save the debug image and then exit the script
            extract_timestamp_from_frame(frame, roi, debug=True)
        else:
            print("ERROR: Could not read the first frame of the video for debugging.")
        cap.release()
        return 0

    print(f"  - Analyzing one frame every second of video...")
    print("  --------------------------------------------------")
    print("  Frame | Video Time | OCR Raw Text      | Status")
    print("  --------------------------------------------------")

    last_valid_time = None
    last_frame_number = -1
    match_count = 0
    consecutive_no_reads = 0

    for frame_number in range(0, total_frames, int(frame_rate)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            print(f"  - WARN: Could not read frame {frame_number}. Skipping.")
            continue

        current_time, raw_text = extract_timestamp_from_frame(frame, roi)
        video_time_sec = frame_number / frame_rate

        log_status = "-"
        if raw_text:
            log_status = f"Found: '{raw_text}'"
        else:
            log_status = "No text found"
        if current_time:
            log_status += f" -> Parsed: {current_time}"
        
        print(f"  {frame_number:<5d} | {timedelta(seconds=int(video_time_sec))} | {raw_text:<15s} | {log_status}")

        if current_time:
            consecutive_no_reads = 0
            if last_valid_time and is_time_fluctuation(last_valid_time, current_time, fluctuation_seconds):
                match_count += 1
                print(f"  *** Time Jump DETECTED! ***")
                print(f"    - From: {last_valid_time} -> To: {current_time}")
                
                start_time_seconds = (last_frame_number / frame_rate) - 30
                start_time_seconds = max(0, start_time_seconds)
                
                base, _ = os.path.splitext(os.path.basename(video_path))
                output_filename = f"{base}_clip_{match_count}.mp4"
                output_path = os.path.join(output_folder, output_filename)
                
                trim_video_clip(video_path, output_path, start_time_seconds, duration_seconds=60)
                
            last_valid_time = current_time
            last_frame_number = frame_number
        else:
            consecutive_no_reads += 1

        if consecutive_no_reads > 10:
            print("  - INFO: Stopping analysis for this file due to 10 consecutive failed OCR reads.")
            break

    cap.release()
    print(f"\nFINISHED: '{os.path.basename(video_path)}'. Found {match_count} match(es).")
    return match_count
