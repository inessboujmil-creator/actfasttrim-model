
import os
import subprocess
from datetime import timedelta

def get_processed_files(db_path):
    """Reads the set of already processed filenames from the database file."""
    if not os.path.exists(db_path):
        return set()
    try:
        with open(db_path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        print(f"WARNING: Could not read processed files database at {db_path}. Reason: {e}")
        return set()

def add_to_processed_files(db_path, filename):
    """Appends a new filename to the processed files database."""
    try:
        with open(db_path, "a") as f:
            f.write(filename + "\n")
    except Exception as e:
        print(f"ERROR: Failed to write to processed files database at {db_path}. Reason: {e}")

def cleanup_processed_db(db_path, all_current_videos):
    """
    Cleans the processed files database by removing entries for videos
    that no longer exist in the source folders.
    """
    processed_filenames = get_processed_files(db_path)
    # Ensure all_current_videos contains only basenames for accurate comparison
    current_basenames = {os.path.basename(f) for f in all_current_videos}
    
    # Determine which files are in the DB but not in the filesystem anymore
    files_to_remove = processed_filenames - current_basenames

    if files_to_remove:
        print(f"INFO: Cleaning up database. Removing {len(files_to_remove)} file(s) that no longer exist.")
        # The remaining files are the intersection
        existing_processed_files = processed_filenames.intersection(current_basenames)
        try:
            with open(db_path, "w") as f:
                for filename in sorted(list(existing_processed_files)):
                    f.write(filename + "\n")
            print("INFO: Database cleanup complete.")
        except Exception as e:
            print(f"ERROR: Failed to write to database during cleanup. Reason: {e}")

def trim_video_clip(video_path, output_path, start_seconds, duration_seconds=60):
    """
    Trims a video using FFmpeg with re-encoding to ensure a high-quality output.
    This version is optimized for robustness.
    """
    print(f"ACTION: Trimming '{os.path.basename(video_path)}'...")
    print(f"  - Start Time: {timedelta(seconds=start_seconds)} ({start_seconds}s)")
    print(f"  - Duration: {duration_seconds}s")
    print(f"  - Output: {output_path}")

    # -hide_banner: Suppresses printing FFmpeg version and build info.
    # -i: Input file.
    # -ss: Seeks to the specified start time. Placing it before -i can be faster for some formats as it seeks to the nearest keyframe.
    # -t: Specifies the duration of the clip to be extracted.
    # -c:v libx264: Video codec - H.264. A widely compatible and high-quality codec.
    # -preset veryfast: A good balance between encoding speed and compression.
    # -crf 23: Constant Rate Factor. A measure of quality (lower is better). 23 is a good default.
    # -c:a aac: Audio codec - Advanced Audio Coding. Standard for most modern applications.
    # -b:a 128k: Audio bitrate. 128 kbps is a standard quality for stereo audio.
    # -y: Overwrite output file without asking.
    command = [
        "ffmpeg",
        "-hide_banner",
        "-ss", str(start_seconds),
        "-i", video_path,
        "-t", str(duration_seconds),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-y",
        output_path
    ]

    try:
        # Using Popen for more control over output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"SUCCESS: Clip saved to '{output_path}'.")
        else:
            print(f"ERROR: FFmpeg failed for '{os.path.basename(video_path)}'.")
            print(f"  - Return Code: {process.returncode}")
            # Print the last few lines of stderr for concise error reporting
            error_lines = stderr.strip().split('\n')[-5:]
            print("  - FFmpeg Error (last 5 lines):")
            for line in error_lines:
                print(f"    {line}")

    except FileNotFoundError:
        print("FATAL: The 'ffmpeg' command was not found.")
        print("Please ensure FFmpeg is installed and its location is included in your system's PATH.")
        # Exit because the core functionality of the app is missing.
        exit()
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during the trim process. Reason: {e}")
