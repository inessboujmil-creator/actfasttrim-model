import os
import subprocess
import cv2
import pytesseract
from datetime import datetime
import imageio_ffmpeg

from .ocr import get_ocr_ready_frame, parse_time_from_ocr, time_str_to_time_obj


def find_frame_by_binary_search(cap, target_time_obj, timestamp_roi, ocr_threshold, total_frames, fps, debug_ocr=False):
    """
    Finds the frame index closest to the target time using a binary search.
    """
    low = 0
    high = total_frames - 1
    best_frame_index = -1
    tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:'
    y1, y2, x1, x2 = timestamp_roi

    print(f"  -> Searching for {target_time_obj.strftime('%H:%M:%S')}...")

    while low <= high:
        mid = (low + high) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        if not ret:
            high = mid - 1
            continue

        roi_frame = frame[y1:y2, x1:x2]
        ocr_ready_frame = get_ocr_ready_frame(roi_frame, ocr_threshold)

        try:
            ocr_text = pytesseract.image_to_string(
                ocr_ready_frame, config=tesseract_config).strip()
            current_time_str = parse_time_from_ocr(ocr_text)
            current_time_obj = time_str_to_time_obj(current_time_str)

            if current_time_obj:
                # Check if current time is within a reasonable range of the target
                time_diff = (datetime.combine(datetime.min, current_time_obj) -
                             datetime.combine(datetime.min, target_time_obj)).total_seconds()

                # If we are very close (e.g., within the search interval), start linear scan
                if abs(time_diff) < 60:  # If within 1 minute, we are close enough
                    best_frame_index = mid
                    break

                if current_time_obj < target_time_obj:
                    low = mid + 1
                else:
                    high = mid - 1
            else:  # If OCR fails, we can't make a decision, so we shrink the search space
                high = mid - 1
        except Exception:
            high = mid - 1

    # After binary search, we do a short linear scan from the 'best_frame_index'
    if best_frame_index != -1:
        # Scan 1 minute before
        start_scan_frame = max(0, best_frame_index - int(fps * 60))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_scan_frame)

        previous_time_obj = None
        for i in range(int(fps * 120)):  # Scan up to 2 minutes
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_index = start_scan_frame + i
            seconds_into_video = current_frame_index / fps
            roi_frame = frame[y1:y2, x1:x2]
            ocr_ready_frame = get_ocr_ready_frame(roi_frame, ocr_threshold)
            ocr_text = pytesseract.image_to_string(
                ocr_ready_frame, config=tesseract_config).strip()
            current_time_str = parse_time_from_ocr(ocr_text)
            current_time_obj = time_str_to_time_obj(current_time_str)

            if current_time_obj:
                if previous_time_obj:
                    is_valid, is_midnight = check_time_interval(
                        previous_time_obj, current_time_obj, 300)
                    if is_valid and has_crossed_target(previous_time_obj, current_time_obj, target_time_obj, is_midnight):
                        return current_frame_index

                previous_time_obj = current_time_obj
    return -1


def process_video_file(video_path, output_folder, timestamp_roi, ocr_threshold, ocr_fluctuation_seconds, target_times, debug_ocr=False):
    """
    Processes a single video file, trimming clips for each target time.
    Returns True if the video was opened and processed successfully, False otherwise.
    """
    normalized_video_path = os.path.normpath(video_path)
    cap = cv2.VideoCapture(normalized_video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {normalized_video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or total_frames == 0:
        print(
            f"ERROR: Could not get FPS or total frames for video: {normalized_video_path}. Skipping.")
        cap.release()
        return False

    print(
        f"INFO: Processing {os.path.basename(normalized_video_path)} with FPS: {fps:.2f}")

    target_time_objs = sorted(
        [time_str_to_time_obj(t) for t in target_times if time_str_to_time_obj(t) is not None])

    for target_obj in target_time_objs:
        found_frame = find_frame_by_binary_search(
            cap, target_obj, timestamp_roi, ocr_threshold, total_frames, fps, debug_ocr)

        if found_frame != -1:
            print(f"\nINFO: Match found for {target_obj.strftime('%H:%M:%S')}!")
            start_seconds = found_frame / fps
            output_filename = f"{os.path.splitext(os.path.basename(normalized_video_path))[0]}_{target_obj.strftime('%H_%M_%S')}.mp4"
            output_path = os.path.join(output_folder, output_filename)

            trim_video_with_reencode(
                normalized_video_path, output_path, start_seconds, 60)
        else:
            print(
                f"  -> Target {target_obj.strftime('%H:%M:%S')} not found in video.")

    cap.release()
    print()
    return True


def check_time_interval(prev_time, curr_time, fluctuation_seconds):
    is_valid = True
    is_midnight_cross = False
    delta_seconds = (datetime.combine(datetime.min, curr_time) -
                     datetime.combine(datetime.min, prev_time)).total_seconds()

    if delta_seconds < 0:
        if prev_time.hour == 23 and curr_time.hour == 0:
            is_midnight_cross = True
            delta_seconds += 86400
        else:
            is_valid = False

    if abs(delta_seconds) > fluctuation_seconds and not is_midnight_cross:
        is_valid = False

    return is_valid, is_midnight_cross


def has_crossed_target(prev_time, curr_time, target_time, is_midnight_cross):
    if is_midnight_cross:
        return target_time > prev_time or target_time <= curr_time
    else:
        return prev_time < target_time <= curr_time


def trim_video_with_reencode(input_path, output_path, start_time, duration):
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe, '-y', '-ss', str(start_time), '-i', input_path,
            '-t', str(duration), '-c:v', 'libx264', '-preset', 'medium',
            '-c:a', 'aac', output_path
        ]

        print(
            f"\nINFO: Trimming 1-minute clip for {os.path.basename(output_path)}... (Re-encoding for reliability)")
        result = subprocess.run(
            command, check=True, capture_output=True, text=True)
        print(f"SUCCESS: Saved trimmed clip to {output_path}")

    except subprocess.CalledProcessError as e:
        print("\n--- FFMPEG ERROR ---")
        print(
            f"Command failed with exit code {e.returncode} when creating {output_path}")
        print("STDERR:", e.stderr)
    except FileNotFoundError:
        print("[ERROR] ffmpeg executable not found via imageio-ffmpeg. Please ensure ffmpeg is installed.")
