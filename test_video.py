
import cv2
import os

def run_video_test():
    # This is the exact video file the main script is getting stuck on.
    video_path = r"E:\Records\Local Records\Ch1_CAM01\20260227162042834.avi"
    
    print("--- Simplified Video Test ---")
    print(f"Target file: {video_path}")

    # 1. Check if the file exists
    if not os.path.exists(video_path):
        print("\nRESULT: FAILURE")
        print("The video file was not found at the specified path.")
        return

    print("Step 1: File found.")

    # 2. Try to open with OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("\nRESULT: FAILURE")
            print("OpenCV could not open the video file. This is often caused by missing video codecs on your system.")
            return
        print("Step 2: Video opened successfully.")
    except Exception as e:
        print(f"\nRESULT: FAILURE. An unexpected error occurred: {e}")
        return

    # 3. Try to read one frame
    try:
        ret, frame = cap.read()
        if not ret:
            print("\nRESULT: FAILURE")
            print("The video file was opened, but no frame could be read. This strongly suggests the file is corrupt or uses an unsupported codec.")
            cap.release()
            return
        print("Step 3: A frame was read successfully.")
    except Exception as e:
        print(f"\nRESULT: FAILURE. An error occurred while reading a frame: {e}")
        cap.release()
        return

    cap.release()
    print("\n--- RESULT: SUCCESS ---")
    print("This test confirms that your Python/OpenCV environment can read your video files.")

if __name__ == "__main__":
    run_video_test()
