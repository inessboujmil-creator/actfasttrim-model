
import configparser
import os
import time
import traceback
from datetime import datetime, timedelta

from src.app import process_video_file
from src.utils.video import get_processed_files, add_to_processed_files

# --- Constants ---
CONFIG_FILE = 'config.txt'
PROCESSED_FILES_DB = 'processed_files.txt'

def load_configuration():
    """
    Loads settings from the config.txt file using configparser.
    This provides a structured and robust way to read the configuration.
    """
    if not os.path.exists(CONFIG_FILE):
        print(f"\nFATAL: Configuration file '{CONFIG_FILE}' not found.")
        print("Please ensure the file exists and is correctly formatted.")
        exit()

    print(f"INFO: Loading configuration from: {os.path.abspath(CONFIG_FILE)}")
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config

def find_unprocessed_videos(config, already_processed):
    """
    Scans all source folders for new .mp4 video files that haven't been processed yet.
    
    Returns:
        A list of full video file paths, sorted chronologically by filename.
    """
    folder_pairs = config['FOLDER_PAIRS']
    days_to_process = config.getint('SETTINGS', 'DAYS_TO_PROCESS', fallback=7)
    
    unprocessed_videos = []
    
    # Calculate the date range to search for videos
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_process)
    
    print(f"INFO: Searching for video files from the last {days_to_process} days...")

    for src_folder in folder_pairs:
        if not os.path.isdir(src_folder):
            print(f"ERROR: Source folder not found: {src_folder}")
            continue
            
        print(f"INFO: Scanning folder: {src_folder}")
        try:
            for filename in os.listdir(src_folder):
                if filename.lower().endswith(".mp4") and filename not in already_processed:
                    try:
                        file_path = os.path.join(src_folder, filename)
                        # Check if the file's modification time is within the desired date range
                        file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if start_date <= file_mod_time <= end_date:
                            unprocessed_videos.append(file_path)
                    except Exception:
                        # In case the filename format is unexpected
                        continue
        except Exception as e:
            print(f"ERROR: Could not read files in {src_folder}. Reason: {e}")
            
    # Sort files by their full path, which usually includes a timestamp
    return sorted(unprocessed_videos)

def main():
    """
    Main loop to continuously scan for and process new video files.
    """
    config = load_configuration()
    
    # --- Tesseract Configuration ---
    tesseract_path = config.get('SETTINGS', 'TESSERACT_PATH', fallback=None)
    if tesseract_path and os.path.exists(tesseract_path.strip('\"')):
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print("INFO: Tesseract command path loaded successfully. OCR is active.")
    else:
        print("WARNING: Tesseract path not found or not set in config. OCR will not be available.")

    # --- Load other settings ---
    timestamp_roi = eval(config.get('SETTINGS', 'TIMESTAMP_ROI', fallback='[0,0,0,0]'))
    fluctuation_seconds = config.getint('SETTINGS', 'OCR_FLUCTUATION_SECONDS', fallback=300)
    scan_interval = config.getint('SETTINGS', 'SCAN_INTERVAL_SECONDS', fallback=300)
    debug_ocr = config.getboolean('SETTINGS', 'DEBUG_OCR', fallback=False)
    folder_pairs = {k.strip('\"'): v.strip('\"') for k, v in config['FOLDER_PAIRS'].items()}
    
    print("\n--- Automated Video Processing System ---")
    print(f"Monitoring {len(folder_pairs)} folder pair(s).")
    print(f"Scan Interval: {scan_interval} seconds.")
    print("System started. Press Ctrl+C to stop.")

    try:
        while True:
            print("\n" + "="*50)
            print(f"INFO: Starting new scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            already_processed = get_processed_files(PROCESSED_FILES_DB)
            new_videos = find_unprocessed_videos(config, already_processed)
            
            if not new_videos:
                print("INFO: No new video files to process.")
            else:
                print(f"INFO: Found {len(new_videos)} new video file(s) to process.")
                
                for video_path in new_videos:
                    src_folder = os.path.dirname(video_path)
                    filename = os.path.basename(video_path)
                    output_folder = folder_pairs.get(src_folder)

                    if not output_folder:
                        print(f"ERROR: No output folder configured for source: {src_folder}. Skipping file.")
                        continue
                    
                    os.makedirs(output_folder, exist_ok=True)
                    
                    print(f"\n--- Processing Video ---")
                    print(f"  - Source: {video_path}")
                    print(f"  - Output: {output_folder}")
                    
                    try:
                        process_video_file(
                            video_path=video_path,
                            output_folder=output_folder,
                            timestamp_roi=timestamp_roi,
                            ocr_fluctuation_seconds=fluctuation_seconds,
                            debug_ocr=debug_ocr
                        )
                        # Add to the processed list only after successful processing
                        add_to_processed_files(PROCESSED_FILES_DB, filename)
                        print(f"INFO: Successfully processed and logged {filename}.")

                        # If in debug mode, stop after one file.
                        if debug_ocr:
                            print("\nINFO: DEBUG_OCR is True. Halting after one file.")
                            return

                    except Exception as e:
                        print(f"\nFATAL ERROR during video processing: {e}")
                        print("An unrecoverable error occurred. See details below.")
                        traceback.print_exc()
                        print(f"Skipping file {filename} due to error.")
                        # Optionally, log the filename to a separate error log
                        with open("error_log.txt", "a") as f:
                            f.write(f"{datetime.now()}: {filename} - {e}\n")
            
            print(f"INFO: Scan complete. Waiting for {scan_interval} seconds before next scan.")
            time.sleep(scan_interval)

    except KeyboardInterrupt:
        print("\nINFO: System stopped by user. Exiting gracefully.")
    except Exception as e:
        print(f"\nFATAL: An unexpected error occurred in the main loop: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
