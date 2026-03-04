
import os
import time
from dotenv import load_dotenv
import cv2

from src.utils.video import (
    get_video_files_by_folder_and_day,
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
        load_dotenv() # Still useful for TESSERACT_CMD or other settings

        # --- New hardcoded folder configuration ---
        self.folder_configs = [
            {
                "input": r"E:\Records\Local Records\Ch1_CAM01",
                "output": r"E:\Records\Local Records\Trimmed_Cam01"
            },
            {
                "input": r"E:\Records\Local Records\Ch1_CAM02",
                "output": r"E:\Records\Local Records\Trimmed_Cam02"
            }
        ]

        self.cam_folders = [config["input"] for config in self.folder_configs]
        # ------------------------------------------

        self.processed_files_db = os.getenv("PROCESSED_FILES_DB", "processed_files.txt")
        self.scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
        self.tesseract_cmd = os.getenv("TESSERACT_CMD")
        self.target_times = [
            "00:30:00", "01:30:00", "02:30:00", "03:30:00", "04:30:00", "05:30:00",
            "06:30:00", "08:30:00", "09:45:00", "10:45:00", "13:00:00", "14:00:00",
            "16:00:00", "17:45:00", "20:30:00", "21:30:00", "22:30:00", "23:30:00"
        ]
        self.timestamp_roi = [10, 50, 1100, 1280]
        self.ocr_fluctuation_seconds = 5

        if self.tesseract_cmd:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def run(self):
        print("--- Automated Video Processing System ---")
        print(f"Monitoring folders: {", ".join(self.cam_folders)}")
        print("System started. Press Ctrl+C to stop.")

        while True:
            try:
                videos_by_folder, all_filenames = get_video_files_by_folder_and_day(self.cam_folders)
                cleanup_processed_db(self.processed_files_db, all_filenames)
                processed_files = get_processed_files(self.processed_files_db)

                oldest_day = None
                for folder in self.cam_folders:
                    for day in sorted(videos_by_folder[folder].keys()):
                        if any(os.path.basename(v) not in processed_files for v in videos_by_folder[folder][day]):
                            if oldest_day is None or day < oldest_day:
                                oldest_day = day

                if oldest_day is None:
                    print(f"INFO: No new videos to process. Waiting for {self.scan_interval} seconds...")
                    time.sleep(self.scan_interval)
                    continue

                print(f"--- Processing oldest day: {oldest_day} ---")

                # Updated turn-based processing loop
                for config in self.folder_configs:
                    folder = config["input"]
                    output_folder = config["output"]

                    if oldest_day in videos_by_folder[folder]:
                        videos_to_process = [v for v in videos_by_folder[folder][oldest_day] if os.path.basename(v) not in processed_files]
                        
                        for video_path in sorted(videos_to_process):
                            filename = os.path.basename(video_path)
                            print(f"\nPROCESSING: '{filename}' in '{os.path.basename(folder)}'")
                            
                            cap = cv2.VideoCapture(video_path)
                            if not cap.isOpened():
                                print(f"ERROR: Could not open video file: {video_path}")
                                continue

                            fps = cap.get(cv2.CAP_PROP_FPS)
                            if fps == 0:
                                print(f"WARNING: Video FPS is 0 for '{filename}'. Cannot calculate time. Skipping.")
                                cap.release()
                                add_to_processed_files(self.processed_files_db, filename)
                                continue

                            found_times_in_video = set()
                            last_valid_timestamp = None

                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                timestamp_str = extract_timestamp_from_frame(frame, self.timestamp_roi)

                                if timestamp_str:
                                    if last_valid_timestamp and is_time_fluctuation(last_valid_timestamp, timestamp_str, self.ocr_fluctuation_seconds):
                                        print(f"  -> OCR FLUCTUATION: Jump from {last_valid_timestamp} to {timestamp_str}. Skipping frame.")
                                        continue
                                    
                                    last_valid_timestamp = timestamp_str

                                    if timestamp_str in self.target_times and timestamp_str not in found_times_in_video:
                                        print(f"  -> TARGET FOUND: {timestamp_str} in '{filename}'")
                                        
                                        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                                        match_second = frame_number / fps

                                        # --- Use the specific output folder from the config ---
                                        os.makedirs(output_folder, exist_ok=True)
                                        
                                        base_name = os.path.splitext(filename)[0]
                                        time_filename_part = timestamp_str.replace(':', '_')
                                        output_filename = f"{base_name}__{time_filename_part}.mp4"
                                        output_path = os.path.join(output_folder, output_filename)

                                        trim_video_clip(video_path, output_path, start_seconds=match_second)
                                        found_times_in_video.add(timestamp_str)
                            
                            cap.release()
                            add_to_processed_files(self.processed_files_db, filename)
                            print(f"FINISHED: '{filename}'. Found {len(found_times_in_video)} target(s).")

            except KeyboardInterrupt:
                print("\nINFO: Manual interruption detected. Shutting down.")
                break
            except Exception as e:
                print(f"An unexpected error occurred in the main loop: {e}")
                time.sleep(20)
