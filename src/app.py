
import os
import time
import cv2

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
        # --- Start of CWD-based .env Loading ---
        # The script is always run from the project root, so CWD is the correct path.
        project_root = os.getcwd()
        dotenv_path = os.path.join(project_root, '.env')
        print(f"DEBUG: Attempting to load .env from CWD: {dotenv_path}")

        try:
            with open(dotenv_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == "TESSERACT_CMD":
                            os.environ[key] = value  # Set variable for this script run
                            print(f"DEBUG: Manually set os.environ['TESSERACT_CMD'].")
                            break
        except FileNotFoundError:
            print(f"ERROR: The .env file was not found at {dotenv_path}")
            print("FATAL: Please ensure the .env file exists in the same directory as main.py")
        except Exception as e:
            print(f"ERROR: Could not parse .env file. Reason: {e}")
        # --- End of CWD-based .env Loading ---

        self.folder_configs = [
            {"input": r"E:\Records\Local Records\Ch1_CAM01", "output": r"E:\Records\Local Records\Trimmed_Cam01"},
            {"input": r"E:\Records\Local Records\Ch1_CAM02", "output": r"E:\Records\Local Records\Trimmed_Cam02"}
        ]

        self.cam_folders = [config["input"] for config in self.folder_configs]
        self.processed_files_db = os.getenv("PROCESSED_FILES_DB", "processed_files.txt")
        self.scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
        
        self.tesseract_cmd = os.getenv("TESSERACT_CMD")
        print(f"DEBUG: Tesseract command path from os.getenv: {self.tesseract_cmd}")
        
        if self.tesseract_cmd:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            print("INFO: Tesseract command path loaded successfully. OCR is active.")
        else:
            print("WARNING: Tesseract path not loaded. OCR will fail.")

        self.target_times = [
            "00:30:00", "01:30:00", "02:30:00", "03:30:00", "04:30:00", "05:30:00",
            "06:30:00", "08:30:00", "09:45:00", "10:45:00", "13:00:00", "14:00:00",
            "16:00:00", "17:45:00", "20:30:00", "21:30:00", "22:30:00", "23:30:00"
        ]
        self.timestamp_roi = [10, 50, 1100, 1280]
        self.ocr_fluctuation_seconds = 5

    def run(self):
        print("--- Automated Video Processing System ---")
        print(f"Monitoring folders: {", ".join(self.cam_folders)}")
        print("System started. Press Ctrl+C to stop.")

        while True:
            try:
                print("\nINFO: Scanning all folders for new files...")
                
                unprocessed_by_folder_and_day = {}
                all_current_filenames = set()
                processed_files = get_processed_files(self.processed_files_db)

                for folder_config in self.folder_configs:
                    folder_path = folder_config["input"]
                    files_in_folder_by_day = {}
                    try:
                        for filename in os.listdir(folder_path):
                            if filename.lower().endswith((".avi", ".mp4", ".mov")):
                                all_current_filenames.add(filename)
                                if filename not in processed_files:
                                    day_str = filename[:8]
                                    if day_str.isdigit() and len(day_str) == 8:
                                        if day_str not in files_in_folder_by_day:
                                            files_in_folder_by_day[day_str] = []
                                        files_in_folder_by_day[day_str].append(os.path.join(folder_path, filename))
                    except FileNotFoundError:
                        print(f"WARNING: Folder not found: {folder_path}. Skipping.")
                        continue
                    
                    if files_in_folder_by_day:
                        unprocessed_by_folder_and_day[folder_path] = files_in_folder_by_day
                
                cleanup_processed_db(self.processed_files_db, all_current_filenames)

                if not unprocessed_by_folder_and_day:
                    print("INFO: No new video files found.")
                else:
                    print(f"INFO: Found new files to process in {len(unprocessed_by_folder_and_day)} folder(s).")

                turn_index = 0
                while unprocessed_by_folder_and_day:
                    try:
                        oldest_day_global = min(day for folder_days in unprocessed_by_folder_and_day.values() for day in folder_days.keys())
                    except ValueError:
                        break 

                    print(f"\n--- Processing oldest day found: {oldest_day_global} ---")
                    
                    folder_to_process = self.folder_configs[turn_index]["input"]

                    if folder_to_process in unprocessed_by_folder_and_day and oldest_day_global in unprocessed_by_folder_and_day[folder_to_process]:
                        files_for_day = unprocessed_by_folder_and_day[folder_to_process][oldest_day_global]
                        print(f"-> Turn for '{os.path.basename(folder_to_process)}': Processing {len(files_for_day)} file(s).")

                        for video_path in sorted(files_for_day):
                            self._process_single_video(video_path)

                        del unprocessed_by_folder_and_day[folder_to_process][oldest_day_global]
                        if not unprocessed_by_folder_and_day[folder_to_process]:
                            del unprocessed_by_folder_and_day[folder_to_process]
                    
                    turn_index = (turn_index + 1) % len(self.folder_configs)
                
                print(f"\nINFO: Scan cycle complete. Waiting for {self.scan_interval} seconds...")
                time.sleep(self.scan_interval)

            except KeyboardInterrupt:
                print("\nINFO: Manual interruption detected. Shutting down.")
                break
            except Exception as e:
                print(f"An unexpected error occurred in the main loop: {e}")
                time.sleep(20)

    def _process_single_video(self, video_path):
        filename = os.path.basename(video_path)
        folder_info = next((item for item in self.folder_configs if item["input"] in video_path), None)
        if not folder_info:
            print(f"WARNING: No folder configuration found for {video_path}. Skipping.")
            return

        print(f"\nPROCESSING: '{filename}' in '{os.path.basename(folder_info['input'])}'")
        print("_______________________________________")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"ERROR: Could not open video file: {video_path}")
                return

            found_times_in_video = set()
            last_ocr_time = None
            frame_skip = 90
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                current_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                ocr_time = extract_timestamp_from_frame(frame, self.timestamp_roi)

                if ocr_time:
                    print(f"  {ocr_time.strftime('%H:%M:%S')}", flush=True)

                    if last_ocr_time and is_time_fluctuation(last_ocr_time, ocr_time, self.ocr_fluctuation_seconds):
                        print(f"  >> WARNING: Time fluctuation detected. Skipping check for this frame to prevent false match.")
                        last_ocr_time = ocr_time 
                        continue 

                    last_ocr_time = ocr_time
                    timestamp_str = ocr_time.strftime('%H:%M:%S')

                    if timestamp_str in self.target_times and timestamp_str not in found_times_in_video:
                        print(f"  >> SUCCESS: Match found for target time '{timestamp_str}'! <<")
                        
                        match_second = (ocr_time.hour * 3600) + (ocr_time.minute * 60) + ocr_time.second
                        output_filename = f"{os.path.splitext(filename)[0]}_trimmed_{timestamp_str.replace(':', '')}.avi"
                        output_path = os.path.join(folder_info["output"], output_filename)

                        if not os.path.exists(folder_info["output"]):
                            os.makedirs(folder_info["output"])
                        
                        print(f"  >> ACTION: Initializing trim to '{output_path}'...")
                        trim_video_clip(video_path, output_path, start_seconds=match_second)
                        found_times_in_video.add(timestamp_str)
                else:
                    # This message will now only appear if OCR genuinely fails to read a timestamp
                    print(f"  -> OCR scan at video time: {int(current_pos_sec)}s... (No valid timestamp detected)", flush=True)
            
            cap.release()
            add_to_processed_files(self.processed_files_db, filename)
            print(f"FINISHED: '{filename}'. Found {len(found_times_in_video)} target(s).")

        except Exception as e:
            print(f"ERROR: Failed to process {filename}. Reason: {e}")
