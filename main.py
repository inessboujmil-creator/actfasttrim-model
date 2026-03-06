
import sys
import os
import time
from datetime import datetime
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
    config = ConfigParser(strict=False)
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
        print(f"ERROR: Missing '{option}' in section '{section}' of the config file.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON in '{option}' in section '{section}'.")
        sys.exit(1)

def get_processed_files(db_path):
    """Loads the set of processed file paths from the database."""
    try:
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                return set(json.load(f))
    except (IOError, json.JSONDecodeError):
        return set()
    return set()

def save_processed_files(db_path, processed_files):
    """Saves the set of processed file paths to the database."""
    try:
        with open(db_path, 'w') as f:
            json.dump(list(processed_files), f, indent=4)
    except IOError:
        print(f"WARN: Could not write to processed files database: {db_path}")

def cleanup_processed_files(db_path, processed_files, all_source_folders):
    """Removes records of files that no longer exist in source folders."""
    existing_files = set()
    for folder in all_source_folders:
        if os.path.isdir(folder):
            for filename in os.listdir(folder):
                existing_files.add(os.path.join(folder, filename))

    cleaned_files = processed_files.intersection(existing_files)
    if len(cleaned_files) < len(processed_files):
        print(f"INFO: Cleaned up {len(processed_files) - len(cleaned_files)} deleted video(s) from the processed files list.")
        save_processed_files(db_path, cleaned_files)
    return cleaned_files

def find_all_unprocessed_videos(folders_data, processed_files):
    """Finds all video files in the source folders that have not been processed yet."""
    unprocessed_videos = []
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

            if re.search(r'(\d{8})', filename):
                unprocessed_videos.append(video_path)

    if unprocessed_videos:
        print(f"INFO: Found {len(unprocessed_videos)} total new video file(s) to process.")
    return unprocessed_videos

def group_videos_by_day(video_paths):
    """Groups videos by the day (YYYYMMDD) found in their filenames."""
    videos_by_day = {}
    for path in video_paths:
        match = re.search(r'(\d{8})', os.path.basename(path))
        if match:
            day = match.group(1)
            if day not in videos_by_day:
                videos_by_day[day] = []
            videos_by_day[day].append(path)
    return videos_by_day

def main():
    """Main function to run the video processing system."""
    config = load_configuration()

    try:
        pytesseract.pytesseract.tesseract_cmd = get_config_value(config, 'System', 'tesseract_path')
        folders_data = get_config_value(config, 'Folders', 'folder_pairs', is_json=True)
        target_times = get_config_value(config, 'Trimming', 'target_times', is_list=True)
        ocr_threshold = get_config_value(config, 'Trimming', 'ocr_threshold', is_int=True)
        scan_interval = get_config_value(config, 'System', 'scan_interval_seconds', is_int=True)
        debug_ocr = config.getboolean('System', 'debug_ocr', fallback=False)

    except SystemExit:
        return

    target_times.sort(key=time_str_to_seconds)

    print("\n--- Automated Video Processing System (Continuous Monitoring) ---")
    print(f"Scanning {len(folders_data)} folder pair(s) every {scan_interval} seconds.")

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
                
                day_videos = sorted(videos_by_day[oldest_day])

                for video_path in day_videos:
                    normalized_path = os.path.normpath(video_path)
                    source_folder = os.path.dirname(normalized_path)
                    output_folder = folders_data.get(source_folder)

                    if not output_folder:
                        print(f"WARN: No output folder configured for {source_folder}. Skipping.")
                        continue
                    
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                        print(f"INFO: Created output folder: {output_folder}")

                    process_video_file(
                        normalized_path, 
                        output_folder, 
                        target_times, 
                        ocr_threshold,
                        debug_ocr
                    )
                    
                    processed_files.add(normalized_path)
                    save_processed_files(PROCESSED_FILES_DB, processed_files)

                print(f"\nINFO: Finished processing all files for {oldest_day}.")

            print(f"INFO: Scan complete. Waiting for {scan_interval} seconds...")
            time.sleep(scan_interval)

    except KeyboardInterrupt:
        print("\nINFO: User interrupted the process. System shutting down.")
    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected error occurred: {e}")
        print("System shutting down.")
    finally:
        sys.exit(0)

if __name__ == '__main__':
    main()
