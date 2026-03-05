
import sys
import os
from datetime import datetime, timedelta
import re
import json
from configparser import ConfigParser, NoSectionError, NoOptionError
import pytesseract

from src.utils.video import process_video_file
from src.utils.ocr import time_str_to_seconds

CONFIG_FILE = 'config.txt'
PROCESSED_FILES_DB = 'processed_files.json'

def load_configuration():
    """Loads settings from the config file."""
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: Configuration file '{CONFIG_FILE}' not found.")
        sys.exit(1)
    # Use strict=False to allow for duplicate keys if they were to occur, though the new format prevents the root cause.
    config = ConfigParser(strict=False) 
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
        print(f"ERROR: '{option}' not found in section '[{section}]'.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse JSON for '{option}' in '[{section}]'.")
        sys.exit(1)

def load_folder_pairs(config):
    """Loads and validates folder pairs from the configuration."""
    if not config.has_section('FOLDER_PAIRS'):
        print("ERROR: [FOLDER_PAIRS] section is missing.")
        sys.exit(1)
    
    folders_data = {}
    for key, value in config.items('FOLDER_PAIRS'):
        # New parsing logic for 'key = input_path, output_path'
        parts = [p.strip() for p in value.split(',')]
        if len(parts) != 2:
            print(f"ERROR: Invalid format for '{key}' in [FOLDER_PAIRS]. Should be: 'input/path, output/path'")
            continue
        input_path = os.path.normpath(parts[0])
        output_path = os.path.normpath(parts[1])
        folders_data[input_path] = output_path
    return folders_data

def get_processed_files(db_path):
    """Loads the set of processed file paths from a JSON file."""
    try:
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                return set(json.load(f))
    except (json.JSONDecodeError, IOError):
        return set()

def add_to_processed_files(db_path, file_path):
    """Adds a file path to the processed files database."""
    processed = get_processed_files(db_path)
    processed.add(file_path)
    with open(db_path, 'w') as f:
        json.dump(list(processed), f, indent=4)

def find_all_unprocessed_videos(folders_data, days_to_process, processed_files):
    """Scans all folders and returns a flat list of all unprocessed video file paths."""
    print(f"INFO: Searching for video files from the last {days_to_process} days across all folders...")
    unprocessed_videos = []
    date_limit = datetime.now() - timedelta(days=days_to_process)

    for source_folder in folders_data.keys():
        if not os.path.isdir(source_folder):
            print(f"WARN: Source folder not found: {source_folder}")
            continue
        
        for filename in os.listdir(source_folder):
            if not filename.lower().endswith(('.mp4', '.avi')):
                continue

            video_path = os.path.join(source_folder, filename)
            if video_path in processed_files:
                continue

            match = re.search(r'(\d{8})', filename)
            if not match:
                continue
            
            day_str = match.group(1)
            try:
                video_date = datetime.strptime(day_str, '%Y%m%d')
                if video_date.date() >= date_limit.date():
                    unprocessed_videos.append(video_path)
            except ValueError:
                continue
    
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
    """Main execution function for a single, one-time scan."""
    config = load_configuration()
    
    # --- Configuration Loading --- #
    tesseract_path = get_config_value(config, 'SETTINGS', 'TESSERACT_PATH')
    timestamp_roi = get_config_value(config, 'SETTINGS', 'TIMESTAMP_ROI', is_json=True)
    ocr_threshold = get_config_value(config, 'SETTINGS', 'OCR_THRESHOLD', is_int=True)
    ocr_fluctuation_seconds = get_config_value(config, 'SETTINGS', 'OCR_FLUCTUATION_SECONDS', is_int=True)
    days_to_process = get_config_value(config, 'SETTINGS', 'DAYS_TO_PROCESS', is_int=True)
    target_times = get_config_value(config, 'SETTINGS', 'TARGET_TIMES', is_list=True)
    debug_ocr = config.getboolean('SETTINGS', 'DEBUG_OCR', fallback=False)

    if os.path.exists(tesseract_path):
        pytesseract.tesseract_cmd = tesseract_path
    
    folders_data = load_folder_pairs(config)
    if not folders_data:
        print("ERROR: [FOLDER_PAIRS] section is empty. Exiting.")
        sys.exit(1)

    target_times.sort(key=time_str_to_seconds)
    
    print("\n--- Automated Video Processing System (One-Time Scan) ---")
    print(f"Scanning {len(folders_data)} folder pair(s).")

    try:
        print(f"\n{'='*60}\nINFO: Starting scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        processed_files = get_processed_files(PROCESSED_FILES_DB)
        all_new_videos = find_all_unprocessed_videos(folders_data, days_to_process, processed_files)

        if not all_new_videos:
            print("INFO: No new videos found.")
        else:
            print("INFO: New videos detected. Grouping by day for chronological processing.")
            videos_by_day = group_videos_by_day(all_new_videos)
            sorted_days = sorted(videos_by_day.keys())

            for day in sorted_days:
                print(f"\n--- Processing Global Oldest Day: {day} ---")
                for video_path in videos_by_day[day]:
                    source_dir = os.path.normpath(os.path.dirname(video_path))
                    output_folder_path = folders_data.get(source_dir)
                    
                    if not output_folder_path:
                        print(f"WARN: No output folder configured for source: {source_dir}. Skipping file.")
                        continue

                    if not os.path.exists(output_folder_path):
                        print(f"INFO: Creating output directory: {output_folder_path}")
                        os.makedirs(output_folder_path)
                    
                    print("\n--- Processing Video ---")
                    print(f"  - Source: {video_path}")
                    process__video_file(
                        video_path=video_path,
                        output_folder=output_folder_path,
                        timestamp_roi=timestamp_roi,
                        ocr_threshold=ocr_threshold,
                        ocr_fluctuation_seconds=ocr_fluctuation_seconds,
                        target_times=target_times,
                        debug_ocr=debug_ocr
                    )
                    add_to_processed_files(PROCESSED_FILES_DB, video_path)
            
            print("\nINFO: Finished processing all detected files.")

    except Exception as e:
        print(f"FATAL: An unexpected error occurred: {e}")
    finally:
        print("Scan complete. System shut down.")

if __name__ == "__main__":
    main()
