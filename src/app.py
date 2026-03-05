
import os
import time
import cv2
import ast

from src.utils.video import (
    cleanup_processed_db,
    get_processed_files,
    add_to_processed_files,
    trim_video_clip
)
from src.utils.ocr import (
    extract_timestamp_from_frame,
    is_time_fluctuation
)

class VideoProcessor:
    def __init__(self):
        # --- Configuration Initialization ---
        self.folder_configs = []
        self.tesseract_cmd = None
        self.scan_interval = 60
        self.timestamp_roi = [10, 50, 1100, 1280]
        self.ocr_fluctuation_seconds = 10
        self.processed_files_db = "processed_files.txt"

        self._load_config()

        # --- Post-config setup ---
        self.cam_folders = [config["input"] for config in self.folder_configs]

        if self.tesseract_cmd:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            print("INFO: Tesseract command path loaded successfully. OCR is active.")
        else:
            print("WARNING: Tesseract path not set in config.txt. OCR will likely fail if Tesseract is not in the system's PATH.")

        self.target_times = [
            "00:30:00", "01:30:00", "02:30:00", "03:30:00", "04:30:00", "05:30:00",
            "06:30:00", "08:30:00", "09:45:00", "10:45:00", "13:00:00", "14:00:00",
            "16:00:00", "17:45:00", "20:30:00", "21:30:00", "22:30:00", "23:30:00"
        ]

    def _load_config(self):
        """Loads configuration from config.txt."""
        project_root = os.getcwd()
        config_path = os.path.join(project_root, 'config.txt')
        print(f"INFO: Loading configuration from: {config_path}")

        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue

                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"')

                    if key == "TESSERACT_CMD":
                        self.tesseract_cmd = value
                        print(f"  - Loaded TESSERACT_CMD: {value}")
                    elif key == "SCAN_INTERVAL_SECONDS":
                        self.scan_interval = int(value)
                        print(f"  - Loaded SCAN_INTERVAL_SECONDS: {value}")
                    elif key == "TIMESTAMP_ROI":
                        self.timestamp_roi = ast.literal_eval(value)
                        print(f"  - Loaded TIMESTAMP_ROI: {value}")
                    elif key == "OCR_FLUCTUATION_SECONDS":
                        self.ocr_fluctuation_seconds = int(value)
                        print(f"  - Loaded OCR_FLUCTUATION_SECONDS: {value}")
                    elif key == "FOLDER_CONFIG":
                        if '|' in value:
                            input_path, output_path = [p.strip().strip('"') for p in value.split('|', 1)]
                            if os.path.isdir(input_path):
                                self.folder_configs.append({"input": input_path, "output": output_path})
                                print(f"  - Loaded FOLDER_CONFIG: {input_path} -> {output_path}")
                            else:
                                print(f"  - WARNING: Input folder does not exist, skipping: {input_path}")
                        else:
                            print(f"  - WARNING: Invalid FOLDER_CONFIG format: {value}")

        except FileNotFoundError:
            print(f"FATAL: The config.txt file was not found at {config_path}")
            print("Please ensure the config.txt file exists and is configured correctly.")
            exit()
        except Exception as e:
            print(f"FATAL: Could not parse config.txt file. Reason: {e}")
            exit()
            
        if not self.folder_configs:
            print("FATAL: No valid FOLDER_CONFIG entries found in config.txt. The program cannot continue.")
            exit()

    def run(self):
        print("\n--- Automated Video Processing System ---")
        print(f"Monitoring folders: {', '.join(self.cam_folders)}")
        print("System started. Press Ctrl+C to stop.")

        while True:
            try:
                print(f"\nINFO: Scanning all folders for new files... (Interval: {self.scan_interval}s)")
                
                unprocessed_by_folder_and_day = {}
                all_current_filenames = set()
                processed_files = get_processed_files(self.processed_files_db)

                for folder_config in self.folder_configs:
                    folder_path = folder_config["input"]
                    files_in_folder_by_day = {}
                    try:
                        for filename in os.listdir(folder_path):
                            if filename.lower().endswith((".avi", ".mp4", ".mov")):
                                full_path = os.path.join(folder_path, filename)
                                all_current_filenames.add(filename)
                                if filename not in processed_files:
                                    day_str = filename[:8]
                                    if day_str.isdigit() and len(day_str) == 8:
                                        if day_str not in files_in_folder_by_day:
                                            files_in_folder_by_day[day_str] = []
                                        files_in_folder_by_day[day_str].append(full_path)
                    except FileNotFoundError:
                        print(f"WARNING: Folder not found during scan: {folder_path}. Skipping.")
                        continue
                    
                    if files_in_folder_by_day:
                        unprocessed_by_folder_and_day[folder_path] = files_in_folder_by_day
                
                cleanup_processed_db(self.processed_files_db, all_current_filenames)

                if not unprocessed_by_folder_and_day:
                    print("INFO: No new video files found.")
                else:
                    print(f"INFO: Found new files to process in {len(unprocessed_by_folder_and_day)} folder(s).")

                turn_index = 0
                processed_a_day = True
                while processed_a_day:
                    processed_a_day = False
                    try:
                        oldest_day_global = min(day for folder_days in unprocessed_by_folder_and_day.values() for day in folder_days.keys())
                    except ValueError:
                        break

                    print(f"\n--- Processing oldest day found: {oldest_day_global} ---")
                    
                    for i in range(len(self.folder_configs)):
                        folder_to_process = self.folder_configs[turn_index]["input"]

                        if folder_to_process in unprocessed_by_folder_and_day and oldest_day_global in unprocessed_by_folder_and_day[folder_to_process]:
                            files_for_day = sorted(unprocessed_by_folder_and_day[folder_to_process][oldest_day_global])
                            print(f"-> Turn for '{os.path.basename(folder_to_process)}': Processing {len(files_for_day)} file(s) for day {oldest_day_global}.")

                            for video_path in files_for_day:
                                self._process_single_video(video_path)

                            del unprocessed_by_folder_and_day[folder_to_process][oldest_day_global]
                            if not unprocessed_by_folder_and_day[folder_to_process]:
                                del unprocessed_by_folder_and_day[folder_to_process]
                            
                            processed_a_day = True
                        
                        turn_index = (turn_index + 1) % len(self.folder_configs)

                print(f"\nINFO: Scan cycle complete. Waiting for {self.scan_interval} seconds...")
                time.sleep(self.scan_interval)

            except KeyboardInterrupt:
                print("\nINFO: Manual interruption detected. Shutting down.")
                break
            except Exception as e:
                print(f"An unexpected error occurred in the main loop: {e}")
                print("Continuing after a 20-second delay...")
                time.sleep(20)

    def _process_single_video(self, video_path):
        filename = os.path.basename(video_path)
        folder_info = next((item for item in self.folder_configs if item["input"] in video_path), None)
        if not folder_info:
            print(f"WARNING: No folder configuration found for {video_path}. Skipping.")
            return

        print(f"\nPROCESSING: '{filename}'")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"ERROR: Could not open video file: {video_path}")
                add_to_processed_files(self.processed_files_db, filename)
                return

            found_times_in_video = set()
            last_ocr_time = None
            frame_skip = 90
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                if frame_pos % frame_skip != 0:
                    continue

                current_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                ocr_time = extract_timestamp_from_frame(frame, self.timestamp_roi)

                if ocr_time:
                    print(f"  - OCR Time: {ocr_time.strftime('%H:%M:%S')} (Video Time: {int(current_pos_sec)}s)", end='\r', flush=True)

                    if last_ocr_time and is_time_fluctuation(last_ocr_time, ocr_time, self.ocr_fluctuation_seconds):
                        print(f"\n  >> WARNING: Fluctuation detected ({last_ocr_time.strftime('%H:%M:%S')} -> {ocr_time.strftime('%H:%M:%S')}). Skipping check.")
                        last_ocr_time = ocr_time 
                        continue 

                    last_ocr_time = ocr_time
                    timestamp_str = ocr_time.strftime('%H:%M:%S')

                    if timestamp_str in self.target_times and timestamp_str not in found_times_in_video:
                        print(f"\n  >> SUCCESS: Match found for target time '{timestamp_str}'! <<")
                        
                        match_second = (ocr_time.hour * 3600) + (ocr_time.minute * 60) + ocr_time.second
                        output_filename = f"{os.path.splitext(filename)[0]}_trimmed_{timestamp_str.replace(':', '')}.avi"
                        output_path = os.path.join(folder_info["output"], output_filename)

                        os.makedirs(folder_info["output"], exist_ok=True)
                        
                        print(f"  >> ACTION: Initializing trim to '{output_path}'...")
                        trim_video_clip(video_path, output_path, start_seconds=match_second)
                        found_times_in_video.add(timestamp_str)

            cap.release()
            print(f"\nFINISHED: '{filename}'. Found {len(found_times_in_video)} match(es).")
            add_to_processed_files(self.processed_files_db, filename)

        except Exception as e:
            print(f"\nERROR: An unexpected error occurred while processing {filename}. Reason: {e}")
            add_to_processed_files(self.processed_files_db, os.path.basename(video_path))
