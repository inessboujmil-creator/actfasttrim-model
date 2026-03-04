
import os
import time
from dotenv import load_dotenv
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
        load_dotenv()

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
        self.processed_files_db = os.getenv("PROCESSED_FILES_DB", "processed_files.txt")
        self.scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
        
        self.tesseract_cmd = os.getenv("TESSERACT_CMD")
        print(f"DEBUG: Tesseract command path from .env: {self.tesseract_cmd}")
        if self.tesseract_cmd:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

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
                # ... (main loop code remains the same)
                pass
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

                    # --- Start of Corrected Fluctuation Logic ---
                    if last_ocr_time and is_time_fluctuation(last_ocr_time, ocr_time, self.ocr_fluctuation_seconds):
                        print(f"  >> WARNING: Time fluctuation detected. Skipping check for this frame to prevent false match.")
                        last_ocr_time = ocr_time # Update time but skip the check for this frame
                        continue # Immediately move to the next frame
                    # --- End of Corrected Fluctuation Logic ---

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
                    print(f"  -> Scanning at video time: {int(current_pos_sec)}s...", flush=True)
            
            cap.release()
            add_to_processed_files(self.processed_files_db, filename)
            print(f"FINISHED: '{filename}'. Found {len(found_times_in_video)} target(s).")

        except Exception as e:
            print(f"ERROR: Failed to process {filename}. Reason: {e}")

