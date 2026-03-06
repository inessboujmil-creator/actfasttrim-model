import sys
import os
import time
from datetime import datetime
import re
import json
from configparser import ConfigParser, NoSectionError, NoOptionError
import pytesseract

from src.utils.video import process_video_file
from src.utils.ocr import time_str_to_time_obj

CONFIG_FILE = 'config.txt'
PROCESSED_FILES_DB = 'processed_files.json'

def load_configuration():
    """Loads settings from the config file."""
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: Configuration file '{CONFIG_FILE}' not found.")
        sys.exit(1)
    config = ConfigParser(allow_no_value=True, strict=False)
    config.read(CONFIG_FILE)
    return config

def get_config_value(config, section, option, is_json=False, is_list=False, is_int=False):
    """Safely gets a value from the config parser."""
    try:
        value = config.get(section, option)
        if is_json: return json.loads(value)
        if is_list: return [item.strip() for item in value.split(',')]
        if is_int: return int(value)
        return value
    except (NoSectionError, NoOptionError):
        print(f"ERROR: '{option}' not found in section '[{section}]'. Please check your config.txt.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse JSON for '{option}' in '[{section}]'.")
        sys.exit(1)

def load_folder_pairs(config):
    """Loads and validates folder pairs from the configuration."""
    if not config.has_section('FOLDER_PAIRS'):
        print("ERROR: [FOLDER_PAIRS] section is missing from config.txt.")
        sys.exit(1)
    
    folders_data = {}
    for key, value in config.items('FOLDER_PAIRS'):
        if not value:
            print(f"WARN: Skipping empty value for key '{key}' in [FOLDER_PAIRS].")
            continue
        parts = [p.strip() for p in value.split(',')]
        if len(parts) != 2:
            print(f"ERROR: Invalid format for '{key}' in [FOLDER_PAIRS]. Should be: 'C:\\input\\path, C:\\output\\path'")
            continue
        input_path = os.path.normpath(parts[0])
        output_path = os.path.normpath(parts[1])
        folders_data[input_path] = output_path
    
    if not folders_data:
        print("ERROR: No valid folder pairs found in [FOLDER_PAIRS]. Exiting.")
        sys.exit(1)
    return folders_data

def get_processed_files(db_path):
    """Loads the set of processed file paths from a JSON file."""
    if not os.path.exists(db_path):
        return set()
    try:
        with open(db_path, 'r') as f:
            return set(json.load(f))
    except (json.JSONDecodeError, IOError):
        print(f"WARN: Could not read or parse '{db_path}'. Starting with an empty list of processed files.")
        return set()

def save_processed_files(db_path, processed_set):
    """Saves the set of processed file paths to the database."""
    try:
        with open(db_path, 'w') as f:
            json.dump(list(processed_set), f, indent=4)
    except IOError as e:
        print(f"WARN: Could not write to processed files database '{db_path}': {e}")

def cleanup_processed_files(db_path, processed_files, all_source_folders):
    """
    Keeps a persistent record of all processed files.
    This function no longer removes entries from the database, even if the source video is deleted.
    This prevents accidental re-processing.
    """
    return processed_files

def find_all_unprocessed_videos(folders_data, processed_files):
    """Scans all folders and returns a flat list of all unprocessed video file paths."""
    unprocessed_videos = []

    for source_folder in folders_data.keys():
        if not os.path.isdir(source_folder):
            print(f"WARN: Source folder not found, skipping: {source_folder}")
            continue
        
        for filename in os.listdir(source_folder):
            if not filename.lower().endswith(('.mp4', '.avi')):
                continue

            video_path = os.path.normpath(os.path.join(source_folder, filename))
            if video_path in processed_files:
                continue

            if re.search(r'(\d{8})', filename):
                unprocessed_videos.append(video_path)

    if unprocessed_videos:
        print(f"INFO: Found {len(unprocessed_videos)} total new video file(s) to process.")
    return unprocessed_videos

def group_videos_by_day(video_paths):
    """Groups a list of video paths into a dictionary keyed by day (YYYYMMDD)."""
    videos_by_day = {}
    for path in video_paths:
        match = re.search(r'(\d{8})', os.path.basename(path))
        if match:
            day_str = match.group(1)
            if day_str not in videos_by_day:
                videos_by_day[day_str] = []
            videos_by_day[day_str].append(path)
    
    for day in videos_by_day:
        videos_by_day[day].sort()
    return videos_by_day

def main():
    """Main function to run the video processing system in a continuous loop."""
    config = load_configuration()
    
    # --- Configuration Loading --- #
    tesseract_path = get_config_value(config, 'SETTINGS', 'TESSERACT_PATH')
    timestamp_roi = get_config_value(config, 'SETTINGS', 'TIMESTAMP_ROI', is_json=True)
    ocr_threshold = get_config_value(config, 'SETTINGS', 'OCR_THRESHOLD', is_int=True)
    ocr_fluctuation_seconds = get_config_value(config, 'SETTINGS', 'OCR_FLUCTUATION_SECONDS', is_int=True)
    target_times = get_config_value(config, 'SETTINGS', 'TARGET_TIMES', is_list=True)
    debug_ocr = config.getboolean('SETTINGS', 'DEBUG_OCR', fallback=False)
    
    try:
        scan_interval = config.getint('SETTINGS', 'SCAN_INTERVAL_SECONDS')
    except (NoOptionError, NoSectionError):
        scan_interval = 7200
        print(f"INFO: 'SCAN_INTERVAL_SECONDS' not in config. Defaulting to {scan_interval} seconds (2 hours).")

    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        print(f"ERROR: Tesseract executable not found at '{tesseract_path}'.")
        sys.exit(1)

    folders_data = load_folder_pairs(config)
    
    target_times.sort(key=lambda t: time_str_to_time_obj(t) or datetime.min.time())

    print("\n--- Automated Video Processing System (Continuous Monitoring) ---")
    print(f"Scanning {len(folders_data)} folder pair(s) every {scan_interval} seconds. Press Ctrl+C to stop.")

    try:
        while True:
            print(f"\n{'='*60}\nINFO: Starting scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            processed_files = get_processed_files(PROCESSED_FILES_DB)
            processed_files = cleanup_processed_files(PROCESSED_FILES_DB, processed_files, folders_data.keys())
            
            all_new_videos = find_all_unprocessed_videos(folders_data, processed_files)

            if not all_new_videos:
                print("INFO: No new videos found.")
            else:
                print("INFO: New videos detected. Grouping by day for chronological processing.")
                videos_by_day = group_videos_by_day(all_new_videos)
                
                oldest_day = sorted(videos_by_day.keys())[0]

                print(f"\n--- Processing Global Oldest Day: {oldest_day} ---")
                for video_path in videos_by_day[oldest_day]:
                    source_dir = os.path.normpath(os.path.dirname(video_path))
                    output_folder_path = folders_data.get(source_dir)
                    
                    if not output_folder_path:
                        print(f"WARN: No output folder configured for source: {source_dir}. Skipping file.")
                        continue

                    if not os.path.exists(output_folder_path):
                        os.makedirs(output_folder_path)
                    
                    print(f"\n--- Processing Video: {os.path.basename(video_path)} ---")
                    if process_video_file(
                        video_path=video_path,
                        output_folder=output_folder_path,
                        timestamp_roi=timestamp_roi,
                        ocr_threshold=ocr_threshold,
                        ocr_fluctuation_seconds=ocr_fluctuation_seconds,
                        target_times=target_times,
                        debug_ocr=debug_ocr
                    ):
                        processed_files.add(video_path)
                        save_processed_files(PROCESSED_FILES_DB, processed_files)
                
                print(f"\nINFO: Finished processing all files for {oldest_day}.")

            print(f"\nINFO: Scan complete. Waiting for {scan_interval} seconds...")
            time.sleep(scan_interval)

    except KeyboardInterrupt:
        print("\n\nINFO: User interrupted the process. System shutting down.")
    except Exception as e:
        print(f"\nFATAL: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.exit(0)

if __name__ == "__main__":
    main()
