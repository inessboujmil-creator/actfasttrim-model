
from src.app import process_video_file
import time
import os
from src.utils.video import get_processed_files, add_to_processed_files

# --- CONFIGURATION ---
CONFIG_FILE = 'config.txt'
PROCESSED_FILES_DB = 'processed_files.txt'

def load_configuration():
    """Loads all settings from the config.txt file."""
    config = {
        'TESSERACT_CMD': None,
        'FOLDER_PAIRS': {},
        'SCAN_INTERVAL_SECONDS': 60,
        'TIMESTAMP_ROI': [10, 50, 1100, 1280], # Default ROI
        'OCR_FLUCTUATION_SECONDS': 10, # Default fluctuation
        'DEBUG_OCR': False # Default OCR debug mode
    }
    
    print(f"INFO: Loading configuration from: {os.path.abspath(CONFIG_FILE)}")
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                if key == 'TESSERACT_CMD':
                    config[key] = value
                    print(f"  - Loaded {key}: {value}")
                elif key == 'FOLDER_CONFIG':
                    src, dst = value.split(' -> ')
                    config['FOLDER_PAIRS'][src.strip()] = dst.strip()
                    print(f"  - Loaded {key}: {src.strip()} -> {dst.strip()}")
                elif key in ['SCAN_INTERVAL_SECONDS', 'OCR_FLUCTUATION_SECONDS']:
                    config[key] = int(value)
                    print(f"  - Loaded {key}: {value}")
                elif key == 'TIMESTAMP_ROI':
                    config[key] = [int(v.strip()) for v in value.split(',')]
                    print(f"  - Loaded {key}: {config[key]}")
                elif key == 'DEBUG_OCR':
                    config[key] = (value.lower() == 'true')
                    print(f"  - Loaded {key}: {config[key]}")
    except FileNotFoundError:
        print(f"\nFATAL: Configuration file '{CONFIG_FILE}' not found.")
        print("Please create it and configure the required paths and settings.")
        exit()
    except Exception as e:
        print(f"\nFATAL: Error parsing configuration file. Please check the syntax. Error: {e}")
        exit()
        
    return config

def main():
    config = load_configuration()
    
    # Set Tesseract path if specified
    if config['TESSERACT_CMD']:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = config['TESSERACT_CMD']
        print("INFO: Tesseract command path loaded successfully. OCR is active.")
    else:
        print("INFO: TESSERACT_CMD not set in config. OCR will not be available.")


    print("\n--- Automated Video Processing System ---")
    monitored_folders = list(config['FOLDER_PAIRS'].keys())
    print(f"Monitoring folders: {', '.join(monitored_folders)}")
    print("System started. Press Ctrl+C to stop.")
    
    try:
        while True:
            print(f"\nINFO: Scanning all folders for new files... (Interval: {config['SCAN_INTERVAL_SECONDS']}s)")
            processed_files_this_run = False
            
            # Get the list of files that have already been processed
            already_processed = get_processed_files(PROCESSED_FILES_DB)
            
            # Find all new, unprocessed video files across all monitored folders
            all_new_files = {}
            for src_folder in monitored_folders:
                try:
                    for filename in os.listdir(src_folder):
                        if filename.endswith(".avi") and filename not in already_processed:
                            day_key = filename[:8] # Group by day (YYYYMMDD)
                            if day_key not in all_new_files:
                                all_new_files[day_key] = {f: [] for f in monitored_folders}
                            all_new_files[day_key][src_folder].append(filename)
                except FileNotFoundError:
                    print(f"ERROR: Source folder not found: {src_folder}")
                except Exception as e:
                    print(f"ERROR: Could not read files in {src_folder}. Reason: {e}")

            if not all_new_files:
                print("INFO: No new files to process.")
            else:
                print(f"INFO: Found new files to process in {len(config['FOLDER_PAIRS'])} folder(s).")

                # Process the oldest day first to handle backlog chronologically
                oldest_day = sorted(all_new_files.keys())[0]
                print(f"\n--- Processing oldest day found: {oldest_day} ---")
                
                files_for_the_day = all_new_files[oldest_day]
                
                # Process folder by folder for that day
                for src_folder, filenames in files_for_the_day.items():
                    if not filenames:
                        continue

                    output_folder = config['FOLDER_PAIRS'][src_folder]
                    os.makedirs(output_folder, exist_ok=True)
                    
                    print(f"\n-> Turn for '{os.path.basename(src_folder)}': Processing {len(filenames)} file(s) for day {oldest_day}.")
                    
                    # Sort files chronologically to process them in order
                    sorted_files = sorted(filenames)
                    
                    for filename in sorted_files:
                        video_path = os.path.join(src_folder, filename)
                        process_video_file(
                            video_path,
                            output_folder,
                            config['TIMESTAMP_ROI'],
                            config['OCR_FLUCTUATION_SECONDS'],
                            config['DEBUG_OCR']
                        )
                        # After processing, add the file to our database
                        add_to_processed_files(PROCESSED_FILES_DB, filename)
                        processed_files_this_run = True

            # If we processed files, scan again immediately. Otherwise, wait.
            if not processed_files_this_run:
                time.sleep(config['SCAN_INTERVAL_SECONDS'])

    except KeyboardInterrupt:
        print("\nINFO: System stopped by user. Exiting.")
    except Exception as e:
        print(f"\nFATAL: An unexpected error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
