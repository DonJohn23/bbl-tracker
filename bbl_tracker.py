import os
import csv
import tempfile
from flask import Flask, request, jsonify
from pytube import YouTube

import torch
import cv2
import ffmpeg

from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential


# ============================================
# ALLOWLIST ALL YOLO MODULES (FINAL FIX)
# ============================================

torch.serialization.add_safe_globals([
    DetectionModel,
    Conv,
    Sequential
])

# Also disable weights_only safety
def unsafe_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)

# Patch YOLO internal loader to use unsafe load
torch.load = unsafe_load


# ============================================
# Load YOLO model normally (Render won’t block it now)
# ============================================

MODEL_PATH = "fullcourt.pt"
model = YOLO(MODEL_PATH)


app = Flask(__name__)
FPS = 30


def download_youtube(url):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension="mp4").first()
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    stream.download(filename=temp_video.name)
    return temp_video.name


def extract_ball_positions(video_file):
    output_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name

    results = model.track(
        source=video_file,
        save=False,
        stream=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    last_x, last_y = None, None

    def box_area(x1, y1, x2, y2):
        return abs((x2 - x1) * (y2 - y1))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_sec", "ball_x", "ball_y"])

        for res in results:
            frame_idx = int(getattr(res, "frame_id", 0))
            time_sec = round(frame_idx / FPS, 2)

            candidates = []
            if hasattr(res, "boxes") and res.boxes is not None:
                for b in res.boxes:
                    x1, y1, x2, y2 = map(float, b.xyxy[0])
                    candidates.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "area": box_area(x1, y1, x2, y2)
                    })

            if not candidates:
                continue

            chosen = min(candidates, key=lambda c: c["area"])
            cx = (chosen["x1"] + chosen["x2"]) / 2
            cy = (chosen["y1"] + chosen["y2"]) / 2

            if last_x is not None:
                if abs(cx - last_x) > 150 or abs(cy - last_y) > 150:
                    cx, cy = last_x, last_y

            last_x, last_y = cx, cy

            writer.writerow([frame_idx, time_sec, round(cx, 2), round(cy, 2)])

    return output_csv


@app.route("/run", methods=["POST"])
def run():
    data = request.json
    youtube_url = data.get("youtube_url")

    if not youtube_url:
        return jsonify({"error": "Missing youtube_url"}), 400

    video_path = download_youtube(youtube_url)
    csv_path = extract_ball_positions(video_path)

    return jsonify({"status": "success", "csv_path": csv_path})


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
