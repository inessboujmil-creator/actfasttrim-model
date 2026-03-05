
import os
from flask import Flask, render_template, jsonify
import logging

# --- Basic Setup ---
app = Flask(__name__, template_folder='src')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
OUTPUT_FOLDER_KEY = "E:\\Records\\Record Plan\\Trimmed_CAM01" # This should be one of the output folders from your config
VIDEO_URL_PREFIX = "/static/videos" # The URL prefix for accessing videos

def get_processed_videos():
    """Scans the output folder for trimmed video files."""
    video_files = []
    # This is a placeholder. We will need to get the actual output directory.
    # For now, let's assume a fixed path for demonstration.
    output_dir = os.path.normpath("E:/Records/Record Plan/Trimmed_CAM01")
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith(".mp4"):
                video_files.append(f)
    return sorted(video_files, reverse=True)

@app.route("/")
def index():
    """Serves the main web interface."""
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    """Provides the current status and list of processed videos."""
    videos = get_processed_videos()
    return jsonify({
        "status": "Running",
        "processed_files": videos,
        "log_messages": [] # We will populate this later
    })

if __name__ == "__main__":
    # Note: In a real deployment, use a production WSGI server like Gunicorn.
    # The host '0.0.0.0' makes the server accessible from your local network.
    app.run(host='0.0.0.0', port=8080, debug=True)
