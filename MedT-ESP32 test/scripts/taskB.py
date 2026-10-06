import os
import cv2
import csv
import numpy as np
from pathlib import Path


VIDEO_FOLDER = "videos"              
OUTPUT_CSV = "task_b_frame_data.csv" 
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
RNG_SEED = 42  # for reproducible simulated IR values

# -----------------------------
# HELPERS
# -----------------------------
def get_video_files(folder):
    folder_path = Path(folder)
    return sorted(
        [f for f in folder_path.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
    )

def compute_avg_brightness(gray_frame):
    return float(np.mean(gray_frame))

def simulate_ir_for_video(num_frames, rng):
    """
    Simulate per-frame IR values for one video.
    Baseline is random between 10 and 245.
    Each frame deviates by -10 to +10.
    Final values are clamped to 0-255.
    """

    baseline_ir = rng.integers(10, 246)  # upper bound exclusive
    deviations = rng.integers(-10, 11, size=num_frames)
    ir_values = np.clip(baseline_ir + deviations, 0, 255)
    return ir_values.astype(int), int(baseline_ir)

# main stuff 
def process_videos(video_folder, output_csv):
    rng = np.random.default_rng(RNG_SEED)
    video_files = get_video_files(video_folder)

    if not video_files:
        print(f"No video files found in '{video_folder}'.")
        return

    all_rows = []

    for video_id, video_path in enumerate(video_files, start=1):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            print(f"Warning: could not determine frame count for {video_path.name}")
            frame_count = 0

        # Simulate IR values for all frames in this video
        ir_values, baseline_ir = simulate_ir_for_video(frame_count, rng)

        print(f"\nProcessing video {video_id}: {video_path.name}")
        print(f"FPS: {fps}")
        print(f"Frame count: {frame_count}")
        print(f"Simulated baseline IR: {baseline_ir}")

        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = compute_avg_brightness(gray)

            # Handle mismatch safely if frame_count metadata is imperfect
            if frame_index < len(ir_values):
                avg_ir = int(ir_values[frame_index])
            else:
                avg_ir = int(np.clip(baseline_ir + rng.integers(-10, 11), 0, 255))

            row = {
                "video_id": video_id,
                "video_name": video_path.name,
                "frame_index": frame_index,
                "fps": float(fps),
                "avg_brightness": avg_brightness,
                "avg_ir": avg_ir,
            }
            all_rows.append(row)
            frame_index += 1

        cap.release()

    # csv output

    with open(output_csv, mode="w", newline="") as csvfile:
        fieldnames = [
            "video_id",
            "video_name",
            "frame_index",
            "fps",
            "avg_brightness",
            "avg_ir",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Wrote {len(all_rows)} rows to '{output_csv}'.")

if __name__ == "__main__":
    process_videos(VIDEO_FOLDER, OUTPUT_CSV)