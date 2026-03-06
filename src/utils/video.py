import cv2
import os
import sys
import numpy as np
from datetime import datetime, timedelta
import pytesseract

from .ocr import extract_time_from_frame, find_timestamp_in_video, time_str_to_seconds

def process_video_file(video_path, output_folder, target_times, timestamp_roi, ocr_threshold, debug_ocr=False):
    """
    Processes a single video file to find and trim clips at target times.

    Args:
        video_path (str): The full path to the video file.
        output_folder (str): The folder where trimmed clips will be saved.
        target_times (list): A list of HH:MM:SS strings to search for.
        timestamp_roi (list): The [y_start, y_end, x_start, x_end] of the timestamp.
        ocr_threshold (int): The grayscale threshold for OCR.
        debug_ocr (bool): If True, saves processed timestamp images for debugging.
    """
    print(f"\n--- Processing Video ---")
    print(f"  - Source: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"ERROR: Video FPS is zero for {video_path}. Skipping.")
        cap.release()
        return

    print(f"INFO: Processing {os.path.basename(video_path)} with FPS: {fps:.2f}")

    for target_time_str in target_times:
        print(f"  -> Searching for {target_time_str}...")
        
        target_seconds = time_str_to_seconds(target_time_str)
        timestamp_seconds, frame_number = find_timestamp_in_video(
            cap, 
            target_seconds, 
            timestamp_roi, 
            ocr_threshold,
            debug_ocr
        )

        if timestamp_seconds is not None:
            print(f"INFO: Match found for {target_time_str}!")
            
            base_filename = os.path.splitext(os.path.basename(video_path))[0]
            time_str_safe = target_time_str.replace(':', '_')
            output_filename = f"{base_filename}_{time_str_safe}.mp4"
            output_path = os.path.join(output_folder, output_filename)
            
            trim_video_clip(video_path, output_path, frame_number, fps)
        else:
            print(f"  -> Target {target_time_str} not found in video.")
    
    cap.release()

def trim_video_clip(video_path, output_path, start_frame, fps, duration_seconds=60):
    """Trims a 1-minute clip from a video starting from a specific frame and re-encodes it."""
    print(f"INFO: Trimming 1-minute clip for {os.path.basename(output_path)}... (Re-encoding for reliability)")

    start_time_seconds = start_frame / fps

    # Use ffmpeg for reliable trimming and re-encoding
    ffmpeg_command = (
        f'ffmpeg -y -ss {start_time_seconds} -i "{video_path}" '
        f'-t {duration_seconds} -c:v libx264 -preset fast -crf 22 -c:a aac -b:a 128k "{output_path}"'
    )

    try:
        # os.system is used for simplicity; for production, subprocess is better
        result = os.system(ffmpeg_command)
        if result == 0:
            print(f"SUCCESS: Saved trimmed clip to {output_path}")
        else:
            print(f"ERROR: FFmpeg command failed with exit code {result}. Clip may not have been saved.")
    except Exception as e:
        print(f"ERROR: An exception occurred during trimming: {e}")
