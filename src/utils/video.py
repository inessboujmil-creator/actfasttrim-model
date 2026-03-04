
import os
import re
import subprocess
from collections import defaultdict
from datetime import timedelta

def get_processed_files(db_path):
    """Reads the set of already processed filenames from the database file."""
    if not os.path.exists(db_path):
        return set()
    with open(db_path, "r") as f:
        return set(line.strip() for line in f)

def add_to_processed_files(db_path, filename):
    """Appends a new filename to the processed files database."""
    with open(db_path, "a") as f:
        f.write(filename + "\n")
    print(f"INFO: Added '{filename}' to processed files database.")

def cleanup_processed_db(db_path, all_current_videos):
    """
    Cleans the processed files database by removing entries for videos
    that no longer exist in the source folders.
    """
    processed_filenames = get_processed_files(db_path)
    existing_processed_files = processed_filenames.intersection(all_current_videos)

    if len(existing_processed_files) < len(processed_filenames):
        print("INFO: Cleaning up processed files database...")
        with open(db_path, "w") as f:
            for filename in sorted(list(existing_processed_files)):
                f.write(filename + "\n")
        print(f"INFO: Removed {len(processed_filenames) - len(existing_processed_files)} non-existent video(s) from the database.")

def get_video_files_by_folder_and_day(folders):
    """
    Scans source folders and returns a dictionary of video paths grouped by folder 
    and then by day, along with a set of all unique video filenames found.
    """
    videos_by_folder = {folder: defaultdict(list) for folder in folders}
    all_video_filenames = set()
    print("INFO: Scanning for video files...")
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"WARNING: Folder '{folder}' not found. Skipping.")
            continue
        
        for filename in os.listdir(folder):
            if filename.endswith(".avi"):
                match = re.search(r"(\d{8})", filename)
                if match:
                    day = match.group(1)
                    videos_by_folder[folder][day].append(os.path.join(folder, filename))
                    all_video_filenames.add(filename)
                else:
                    print(f"WARNING: Could not extract date from filename: {filename}")

    return videos_by_folder, all_video_filenames

def trim_video_clip(video_path, output_path, start_seconds, duration_seconds=60):
    """
    Trims a video using FFmpeg, forcing re-encoding to ensure a high-quality,
    playable output file.
    """
    print(f"ACTION: Trimming video '{video_path}'...")
    print(f"  -> Start: {timedelta(seconds=start_seconds)}, Duration: {duration_seconds}s")
    print(f"  -> Output: {output_path}")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-ss", str(start_seconds),  # Start time for the trim
        "-t", str(duration_seconds),  # Duration of the clip
        "-c:v", "libx244",         # Re-encode video to H.264
        "-c:a", "aac",             # Re-encode audio to AAC
        "-strict", "-2",           # Necessary for some FFmpeg/AAC versions
        "-y",                      # Overwrite output file if it exists
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"SUCCESS: Successfully trimmed and saved clip to '{output_path}'.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: FFmpeg failed to trim video '{video_path}'.")
        print(f"  -> Command: {' '.join(command)}")
        print(f"  -> FFmpeg Error: {e.stderr.decode()}")
    except FileNotFoundError:
        print("ERROR: FFmpeg command not found. Please ensure FFmpeg is installed and in your system's PATH.")
        exit()
