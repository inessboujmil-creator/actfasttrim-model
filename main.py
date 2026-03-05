
import configparser
import os
import time
import traceback
from datetime import datetime, timedelta
from collections import defaultdict
import re

from src.app import process_video_file
from src.utils.video import get_processed_files, add_to_processed_files

# --- Constants ---
CONFIG_FILE = 'config.txt'
PROCESSED_FILES_DB = 'processed_files.txt'

def load_configuration():
    """
    Loads settings from config.txt, handling potential Windows path issues.
    """
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"FATAL: Configuration file '{CONFIG_FILE}' not found.")

    print(f"INFO: Loading configuration from: {os.path.abspath(CONFIG_FILE)}")
    config = configparser.ConfigParser(delimiters=('='))
    config.read(CONFIG_FILE)
    return config

def get_day_from_filename(filename):
    """
    Extracts the date (e.g., '20260205') from a filename using regex.
    """
    match = re.search(r'(\d{8})', filename)
    return match.group(1) if match else None

def find_unprocessed_videos_grouped_by_day(config, already_processed):
    """
    Scans source folders and groups unprocessed videos by day, returning a sorted dictionary.
    """
    folder_pairs = config['FOLDER_PAIRS']
    days_to_process = config.getint('SETTINGS', 'DAYS_TO_PROCESS', fallback=7)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_process)
    
    videos_by_day = defaultdict(lambda: defaultdict(list))

    print(f"INFO: Searching for video files from the last {days_to_process} days...")

    for src_folder, _ in folder_pairs.items():
        src_folder = src_folder.strip()
        if not os.path.isdir(src_folder):
            print(f"ERROR: Source folder not found: {src_folder}")
            continue
        
        print(f"INFO: Scanning folder: {src_folder}")
        for filename in os.listdir(src_folder):
            if filename.lower().endswith(".mp4") and filename not in already_processed:
                day_str = get_day_from_filename(filename)
                if not day_str:
                    continue

                file_path = os.path.join(src_folder, filename)
                try:
                    file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if start_date <= file_mod_time <= end_date:
                        videos_by_day[day_str][src_folder].append(file_path)
                except Exception as e:
                    print(f"WARN: Could not process file info for {filename}. Reason: {e}")

    # Sort the days chronologically and then sort the files within each folder
    sorted_days = sorted(videos_by_day.keys())
    sorted_videos_by_day = {day: videos_by_day[day] for day in sorted_days}
    for day in sorted_videos_by_day:
        for folder in sorted_videos_by_day[day]:
            sorted_videos_by_day[day][folder].sort()
            
    return sorted_videos_by_day

def main():
    """
    Main loop to orchestrate the turn-based, chronological processing of videos.
    """
    config = load_configuration()
    
    tesseract_path = config.get('SETTINGS', 'TESSERACT_PATH', fallback='').strip()
    if tesseract_path and os.path.exists(tesseract_path):
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print("INFO: Tesseract command path loaded successfully.")
    else:
        print("WARNING: Tesseract path not set or invalid. OCR will not be available.")

    timestamp_roi = eval(config.get('SETTINGS', 'TIMESTAMP_ROI', fallback='[0,0,0,0]'))
    fluctuation_seconds = config.getint('SETTINGS', 'OCR_FLUCTUATION_SECONDS', fallback=300)
    scan_interval = config.getint('SETTINGS', 'SCAN_INTERVAL_SECONDS', fallback=300)
    debug_ocr = config.getboolean('SETTINGS', 'DEBUG_OCR', fallback=False)
    
    target_times_str = config.get('SETTINGS', 'TARGET_TIMES', fallback='')
    target_times = [t.strip() for t in target_times_str.split(',')]
    
    folder_pairs = {k.strip(): v.strip() for k, v in config['FOLDER_PAIRS'].items()}
    
    print("\n--- Automated Video Processing System ---")
    print(f"Monitoring {len(folder_pairs)} folder pair(s).")
    print(f"Scan Interval: {scan_interval} seconds.")
    print(f"Target Times: {target_times}")
    print("System started. Press Ctrl+C to stop.")

    try:
        while True:
            print("\n" + "="*50)
            print(f"INFO: Starting new scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            already_processed = get_processed_files(PROCESSED_FILES_DB)
            videos_by_day = find_unprocessed_videos_grouped_by_day(config, already_processed)
            
            if not videos_by_day:
                print("INFO: No new video files to process.")
            else:
                total_videos = sum(len(files) for day_data in videos_by_day.values() for files in day_data.values())
                print(f"INFO: Found {total_videos} new video file(s) across {len(videos_by_day)} day(s).")
                
                # Process day by day, oldest first
                for day, folders_data in videos_by_day.items():
                    print(f"\n--- Processing Day: {day} ---")
                    
                    # Process folder by folder within the day
                    for folder, video_list in folders_data.items():
                        print(f"--- Folder: {folder} ---")
                        for video_path in video_list:
                            filename = os.path.basename(video_path)
                            output_folder = folder_pairs.get(folder)

                            if not output_folder:
                                print(f"ERROR: No output folder configured for {folder}. Skipping.")
                                continue
                            
                            os.makedirs(output_folder, exist_ok=True)
                            
                            print(f"\n--- Processing Video ---")
                            print(f"  - Source: {video_path}")
                            
                            try:
                                process_video_file(
                                    video_path=video_path,
                                    output_folder=output_folder,
                                    timestamp_roi=timestamp_roi,
                                    ocr_fluctuation_seconds=fluctuation_seconds,
                                    target_times=target_times,
                                    debug_ocr=debug_ocr
                                )
                                add_to_processed_files(PROCESSED_FILES_DB, filename)
                                print(f"INFO: Successfully processed and logged {filename}.")

                            except Exception:
                                print(f"\nFATAL ERROR during video processing: {filename}")
                                traceback.print_exc()
                                with open("error_log.txt", "a") as f:
                                    f.write(f"{datetime.now()}: {filename} - {traceback.format_exc()}\n")
            
            print(f"\nINFO: Scan complete. Waiting for {scan_interval} seconds.")
            time.sleep(scan_interval)

    except KeyboardInterrupt:
        print("\nINFO: System stopped by user.")
    except Exception:
        print("\nFATAL: An unexpected error occurred in the main loop.")
        traceback.print_exc()

if __name__ == "__main__":
    main()
