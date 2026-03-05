
import sys
import os
import time
from datetime import datetime, timedelta
import re
import json
from configparser import ConfigParser, NoSectionError, NoOptionError
import pytesseract

# Assuming src/app.py is in the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from app import process_video_file, time_str_to_seconds

# --- Constants ---
CONFIG_FILE = 'config.txt'
PROCESSED_FILES_DB = 'processed_files.json'

def load_configuration():
    """Loads settings from the config file."""
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: Configuration file '{CONFIG_FILE}' not found.")
        sys.exit(1)

    print(f"INFO: Loading configuration from: {os.path.abspath(CONFIG_FILE)}")
    config = ConfigParser()
    config.read(CONFIG_FILE)
    return config

def get_config_value(config, section, option, is_json=False, is_list=False, is_int=False):
    """Safely gets a value from the config parser."""
    try:
        value = config.get(section, option)
        if is_json:
            return json.loads(value)
        if is_list:
            return [item.strip() for item in value.split(',')]
        if is_int:
            return int(value)
        return value
    except (NoSectionError, NoOptionError):
        print(f"ERROR: '{option}' not found in section '[{section}]' of the config file.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse JSON value for '{option}' in section '[{section}]'.")
        sys.exit(1)

def load_folder_pairs(config):
    """Loads and validates folder pairs from the configuration."""
    if not config.has_section('FOLDER_PAIRS'):
        print("ERROR: [FOLDER_PAIRS] section is missing from the config file.")
        sys.exit(1)

    folders_data = {}
    for _, value in config.items('FOLDER_PAIRS'):
        parts = [p.strip() for p in value.split(',')]
        if len(parts) != 2:
            print(f"ERROR: Invalid format in [FOLDER_PAIRS]. Each line must be 'source, destination'. Problem line: {value}")
            sys.exit(1)
        source, destination = parts
        folders_data[source] = destination
    return folders_data

def get_processed_files(db_path):
    """Loads the set of processed file paths from a JSON file."""
    if not os.path.exists(db_path):
        return set()
    with open(db_path, 'r') as f:
        try:
            return set(json.load(f))
        except json.JSONDecodeError:
            return set()

def add_to_processed_files(db_path, file_path):
    """Adds a file path to the processed files database."""
    processed = get_processed_files(db_path)
    processed.add(file_path)
    with open(db_path, 'w') as f:
        json.dump(list(processed), f, indent=4)

def find_unprocessed_videos_grouped_by_day(folders_data, days_to_process, processed_files):
    """Scans folders, finds unprocessed videos, and groups them by day."""
    print(f"INFO: Searching for video files from the last {days_to_process} days...")
    videos_by_day = {}
    today = datetime.now()
    date_limit = today - timedelta(days=days_to_process)

    for source_folder, _ in folders_data.items():
        print(f"INFO: Scanning folder: {source_folder}")
        if not os.path.isdir(source_folder):
            print(f"WARN: Source folder not found: {source_folder}")
            continue
        
        for filename in os.listdir(source_folder):
            if not filename.lower().endswith('.mp4'):
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
                if video_date >= date_limit:
                    if day_str not in videos_by_day:
                        videos_by_day[day_str] = []
                    videos_by_day[day_str].append(video_path)
            except ValueError:
                continue
    
    for day in videos_by_day:
        videos_by_day[day].sort()

    found_count = sum(len(v) for v in videos_by_day.values())
    day_count = len(videos_by_day)
    print(f"INFO: Found {found_count} new video file(s) across {day_count} day(s).")

    return videos_by_day

def main():
    """Main execution function."""
    config = load_configuration()
    
    tesseract_path = get_config_value(config, 'SETTINGS', 'TESSERACT_PATH')
    timestamp_roi = get_config_value(config, 'SETTINGS', 'TIMESTAMP_ROI', is_json=True)
    ocr_fluctuation_seconds = get_config_value(config, 'SETTINGS', 'OCR_FLUCTUATION_SECONDS', is_int=True)
    days_to_process = get_config_value(config, 'SETTINGS', 'DAYS_TO_PROCESS', is_int=True)
    scan_interval = get_config_value(config, 'SETTINGS', 'SCAN_INTERVAL_SECONDS', is_int=True)
    target_times = get_config_value(config, 'SETTINGS', 'TARGET_TIMES', is_list=True)
    debug_ocr = config.getboolean('SETTINGS', 'DEBUG_OCR')

    if os.path.exists(tesseract_path):
        pytesseract.tesseract_cmd = tesseract_path
        print("INFO: Tesseract command path loaded successfully.")
    else:
        print(f"WARNING: Tesseract path not set or invalid. OCR will not be available.")

    folders_data = load_folder_pairs(config)
    if not folders_data:
        print("ERROR: [FOLDER_PAIRS] section is empty or invalid. Please check the config file.")
        sys.exit(1)

    target_times.sort(key=time_str_to_seconds)
    
    print("\n--- Automated Video Processing System ---")
    print(f"Monitoring {len(folders_data)} folder pair(s).")
    print(f"Scan Interval: {scan_interval} seconds.")
    print(f"Target Times: {target_times}")
    print("System started. Press Ctrl+C to stop.")

    try:
        while True:
            print("\n==================================================")
            print(f"INFO: Starting new scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            processed_files = get_processed_files(PROCESSED_FILES_DB)
            videos_by_day = find_unprocessed_videos_grouped_by_day(folders_data, days_to_process, processed_files)
            sorted_days = sorted(videos_by_day.keys())

            if not sorted_days:
                time.sleep(scan_interval)
                continue

            for day in sorted_days:
                print(f"\n--- Processing Day: {day} ---")
                for source_folder, output_folder_name in folders_data.items():
                    videos_in_folder_for_day = [v for v in videos_by_day.get(day, []) if os.path.dirname(v) == os.path.normpath(source_folder)]
                    if not videos_in_folder_for_day:
                        continue
                    
                    # --- DEFINITIVE PATH FIX ---
                    # Construct the correct absolute path for the output folder.
                    # It is created as a sibling to the parent of the source folder.
                    parent_of_source = os.path.dirname(os.path.normpath(source_folder))
                    output_folder_path = os.path.join(parent_of_source, output_folder_name)
                    # --- END FIX ---

                    print(f"--- Folder: {source_folder} ---")
                    if not os.path.exists(output_folder_path):
                        print(f"INFO: Creating output directory: {output_folder_path}")
                        os.makedirs(output_folder_path)
                    
                    for video_path in videos_in_folder_for_day:
                        print("\n--- Processing Video ---")
                        print(f"  - Source: {video_path}")
                        process_video_file(
                            video_path=video_path,
                            output_folder=output_folder_path, # Use the corrected, absolute path
                            timestamp_roi=timestamp_roi,
                            ocr_fluctuation_seconds=ocr_fluctuation_seconds,
                            target_times=target_times,
                            debug_ocr=debug_ocr
                        )
                        add_to_processed_files(PROCESSED_FILES_DB, video_path)

            print(f"\nINFO: Scan complete. Waiting for {scan_interval} seconds...")
            time.sleep(scan_interval)

    except KeyboardInterrupt:
        print("\nINFO: System stopped by user.")
    except Exception as e:
        print(f"FATAL: An unexpected error occurred: {e}")
    finally:
        print("System shut down.")

if __name__ == "__main__":
    main()
