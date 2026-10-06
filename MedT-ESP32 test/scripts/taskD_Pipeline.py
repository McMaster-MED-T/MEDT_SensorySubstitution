import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

try:
    import serial  # pyserial
except ImportError:
    serial = None


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
RNG_SEED = 42
MODE_TO_STATE = {"00": 0, "01": 1, "10": 2, "11": 3}
STATE_TO_MODE = {v: k for k, v in MODE_TO_STATE.items()}


# -----------------------------
# TASK B HELPERS
# -----------------------------
def get_video_files(folder: str) -> List[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        return []
    return sorted([f for f in folder_path.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS])



def compute_avg_brightness(gray_frame: np.ndarray) -> float:
    return float(np.mean(gray_frame))



def simulate_ir_for_video(num_frames: int, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    baseline_ir = rng.integers(10, 246)
    deviations = rng.integers(-10, 11, size=max(num_frames, 0))
    ir_values = np.clip(baseline_ir + deviations, 0, 255)
    return ir_values.astype(int), int(baseline_ir)



def process_videos(video_folder: str, output_csv: str, rng_seed: int = RNG_SEED) -> pd.DataFrame:
    """Task B: extract avg_brightness and simulated avg_ir for every frame."""
    rng = np.random.default_rng(rng_seed)
    video_files = get_video_files(video_folder)

    if not video_files:
        raise FileNotFoundError(f"No video files found in '{video_folder}'.")

    all_rows: List[Dict[str, object]] = []

    for video_id, video_path in enumerate(video_files, start=1):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ir_values, baseline_ir = simulate_ir_for_video(frame_count, rng)

        print(f"\nProcessing video {video_id}: {video_path.name}")
        print(f"FPS: {fps}")
        print(f"Frame count (metadata): {frame_count}")
        print(f"Simulated baseline IR: {baseline_ir}")

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = compute_avg_brightness(gray)

            if frame_index < len(ir_values):
                avg_ir = int(ir_values[frame_index])
            else:
                avg_ir = int(np.clip(baseline_ir + rng.integers(-10, 11), 0, 255))

            all_rows.append(
                {
                    "video_id": video_id,
                    "video_name": video_path.name,
                    "frame_index": frame_index,
                    "fps": fps,
                    "avg_brightness": avg_brightness,
                    "avg_ir": avg_ir,
                }
            )
            frame_index += 1

        cap.release()

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"\nDone. Wrote {len(df)} rows to '{output_csv}'.")
    return df


# -----------------------------
# TASK C HELPERS
# -----------------------------
def train_kmeans_from_frame_csv(
    frame_csv: str,
    clustered_csv: str = "data/clustered_data.csv",
    model_json: str = "models/kmeans_model.json",
    make_plots: bool = True,
) -> Tuple[pd.DataFrame, KMeans, Dict[int, str]]:
    """Train K-means on Task B output and save reusable centroids for Task D."""
    df = pd.read_csv(frame_csv)
    required_cols = {"video_id", "frame_index", "avg_ir", "avg_brightness"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {frame_csv}: {sorted(missing)}")

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(df[["avg_ir", "avg_brightness"]])

    centers = kmeans.cluster_centers_
    mid_ir = float(np.median(centers[:, 0]))
    mid_bright = float(np.median(centers[:, 1]))

    cluster_to_mode: Dict[int, str] = {}
    for cluster_id, center in enumerate(centers):
        ir_center = float(center[0])
        bright_center = float(center[1])
        b_bit = "1" if bright_center > mid_bright else "0"
        i_bit = "1" if ir_center > mid_ir else "0"
        cluster_to_mode[cluster_id] = f"{b_bit}{i_bit}"

    df["mode"] = df["cluster"].map(cluster_to_mode)
    df["state"] = df["mode"].map(MODE_TO_STATE)
    df.to_csv(clustered_csv, index=False)

    model_info = {
        "feature_order": ["avg_ir", "avg_brightness"],
        "centroids": centers.tolist(),
        "cluster_to_mode": {str(k): v for k, v in cluster_to_mode.items()},
        "mode_to_state": MODE_TO_STATE,
    }
    with open(model_json, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print(f"Saved clustered data to '{clustered_csv}'")
    print(f"Saved reusable model to '{model_json}'")

    if make_plots:
        plot_training_data(df, centers, cluster_to_mode)

    return df, kmeans, cluster_to_mode



def plot_training_data(df: pd.DataFrame, centers: np.ndarray, cluster_to_mode: Dict[int, str]) -> None:
    plt.figure(figsize=(9, 6))
    for video_id, subset in df.groupby("video_id"):
        plt.scatter(subset["avg_ir"], subset["avg_brightness"], s=10, alpha=0.55, label=f"Video {video_id}")
    plt.xlabel("Average IR")
    plt.ylabel("Average Brightness")
    plt.title("Training Data by Video")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 6))
    mode_colors = {"00": "tab:blue", "01": "tab:orange", "10": "tab:green", "11": "tab:red"}
    for mode, subset in df.groupby("mode"):
        plt.scatter(subset["avg_ir"], subset["avg_brightness"], s=10, alpha=0.55, color=mode_colors.get(mode), label=f"Mode {mode}")

    for cluster_id, center in enumerate(centers):
        plt.scatter(center[0], center[1], marker="X", s=220, edgecolor="black", linewidth=1.2,
                    label=f"Centroid {cluster_id} -> {cluster_to_mode[cluster_id]}")
    plt.xlabel("Average IR")
    plt.ylabel("Average Brightness")
    plt.title("K-means Clusters and Centroids")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# TASK D HELPERS
# -----------------------------
def load_cluster_model(model_path: str) -> Tuple[np.ndarray, Dict[int, str]]:
    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)
    centroids = np.array(model["centroids"], dtype=float)
    cluster_to_mode = {int(k): v for k, v in model["cluster_to_mode"].items()}
    return centroids, cluster_to_mode



def classify_feature_pair(avg_brightness: float, avg_ir: float, centroids: np.ndarray, cluster_to_mode: Dict[int, str]) -> Tuple[int, str, int]:
    point = np.array([avg_ir, avg_brightness], dtype=float)
    distances = np.linalg.norm(centroids - point, axis=1)
    nearest_cluster = int(np.argmin(distances))
    mode = cluster_to_mode[nearest_cluster]
    state = MODE_TO_STATE[mode]
    return nearest_cluster, mode, state



def build_video_state_dataframe(frame_csv: str, video_id: int, model_json: str) -> pd.DataFrame:
    df = pd.read_csv(frame_csv)
    centroids, cluster_to_mode = load_cluster_model(model_json)

    video_df = df[df["video_id"] == video_id].copy()
    if video_df.empty:
        raise ValueError(f"No data found for video_id={video_id}")

    video_df = video_df.sort_values("frame_index").reset_index(drop=True)

    clusters: List[int] = []
    modes: List[str] = []
    states: List[int] = []

    for _, row in video_df.iterrows():
        cluster, mode, state = classify_feature_pair(
            avg_brightness=float(row["avg_brightness"]),
            avg_ir=float(row["avg_ir"]),
            centroids=centroids,
            cluster_to_mode=cluster_to_mode,
        )
        clusters.append(cluster)
        modes.append(mode)
        states.append(state)

    video_df["cluster"] = clusters
    video_df["mode"] = modes
    video_df["state"] = states
    return video_df



def smooth_states(states: List[int], min_run_length: int = 3) -> List[int]:
    """Simple debounce filter: a new state must persist before switching."""
    if not states:
        return []

    output = [states[0]]
    candidate = states[0]
    candidate_count = 1

    for current in states[1:]:
        previous_output = output[-1]

        if current == candidate:
            candidate_count += 1
        else:
            candidate = current
            candidate_count = 1

        if candidate != previous_output and candidate_count >= min_run_length:
            output.append(candidate)
        else:
            output.append(previous_output)

    return output



def add_pwm_envelope(states: List[int], step: float = 0.01, unstable_threshold: float = 0.05, unstable_frames: int = 15) -> pd.DataFrame:
    """Optional interpretation of the PWM ramp idea from the task notes."""
    if not states:
        return pd.DataFrame(columns=["active_state", "duty_cycle", "unstable"])

    active_state = states[0]
    duty_cycle = step
    low_duty_count = 0

    rows = []
    for observed_state in states:
        unstable = False
        if observed_state == active_state:
            duty_cycle = min(1.0, duty_cycle + step)
        else:
            duty_cycle = max(0.0, duty_cycle - step)
            if duty_cycle <= 0.0:
                active_state = observed_state
                duty_cycle = step

        if duty_cycle <= unstable_threshold:
            low_duty_count += 1
        else:
            low_duty_count = 0

        if low_duty_count >= unstable_frames:
            unstable = True

        rows.append(
            {
                "active_state": active_state,
                "active_mode": STATE_TO_MODE[active_state],
                "duty_cycle": round(duty_cycle, 4),
                "unstable": unstable,
            }
        )

    return pd.DataFrame(rows)



def build_payload(video_df: pd.DataFrame, state_column: str = "smoothed_state") -> Dict[str, object]:
    fps = float(video_df["fps"].iloc[0])
    states = [int(x) for x in video_df[state_column].tolist()]
    return {
        "fps": fps,
        "frame_delay_ms": round(1000.0 / fps, 3) if fps > 0 else None,
        "num_frames": len(states),
        "states": states,
    }


def send_payload_serial(payload, port, baudrate=115200, reset_delay=2.0):
    ser = serial.Serial(port, baudrate, timeout=2)
    
    # help avoid reset glitches
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(reset_delay)

    message = json.dumps(payload) + "\n"
    ser.write(message.encode("utf-8"))
    ser.flush()

    print(f"Sent {payload['num_frames']} states to {port} at {baudrate} baud.")

    # keep port open long enough for playback
    if payload.get("fps", 0) > 0:
        playback_time = payload["num_frames"] / payload["fps"]
    else:
        playback_time = 5

    print(f"Keeping serial open for {playback_time:.1f} s")
    time.sleep(playback_time + 1)

    ser.close()



def export_task_d_outputs(
    frame_csv: str,
    model_json: str,
    video_id: int,
    output_csv: str,
    output_json: str,
    min_run_length: int = 3,
    include_pwm: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    video_df = build_video_state_dataframe(frame_csv, video_id, model_json)
    video_df["smoothed_state"] = smooth_states(video_df["state"].tolist(), min_run_length=min_run_length)
    video_df["smoothed_mode"] = video_df["smoothed_state"].map(STATE_TO_MODE)

    if include_pwm:
        pwm_df = add_pwm_envelope(video_df["smoothed_state"].tolist())
        video_df = pd.concat([video_df, pwm_df], axis=1)

    video_df.to_csv(output_csv, index=False)
    payload = build_payload(video_df, state_column="smoothed_state")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved Task D dataframe to '{output_csv}'")
    print(f"Saved ESP32 payload to '{output_json}'")
    return video_df, payload


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined Task B + C + D pipeline: extract frame features, train k-means, build ESP32 state sequence."
    )
    parser.add_argument("--video-folder", default="videos", help="Folder containing input videos for Task B.")
    parser.add_argument("--frame-csv", default="data/task_b_frame_data.csv", help="Path to Task B frame data CSV.")
    parser.add_argument("--clustered-csv", default="data/clustered_data.csv", help="Path to save clustered training data.")
    parser.add_argument("--model-json", default="models/kmeans_model.json", help="Path to save/load reusable centroid model.")
    parser.add_argument("--video-id", type=int, default=1, help="video_id to export for Task D.")
    parser.add_argument("--taskd-csv", default=None, help="Task D output CSV for one video.")
    parser.add_argument("--payload-json", default=None, help="JSON payload to send to ESP32.")
    parser.add_argument("--min-run-length", type=int, default=3, help="Debounce length for state smoothing.")
    parser.add_argument("--no-plots", action="store_true", help="Disable training plots.")
    parser.add_argument("--skip-taskb", action="store_true", help="Skip Task B extraction and use existing frame CSV.")
    parser.add_argument("--skip-train", action="store_true", help="Skip Task C training and use existing model JSON.")
    parser.add_argument("--serial-port", default=None, help="Serial port for optional ESP32 send, e.g. COM5 or /dev/ttyUSB0.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baud rate.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    taskd_csv = args.taskd_csv or f"data/video_{args.video_id}_classified_states.csv"
    payload_json = args.payload_json or f"data/video_{args.video_id}_esp32_payload.json"

    if not args.skip_taskb:
        process_videos(args.video_folder, args.frame_csv)
    else:
        print("Skipping Task B extraction.")

    if not args.skip_train:
        train_kmeans_from_frame_csv(
            frame_csv=args.frame_csv,
            clustered_csv=args.clustered_csv,
            model_json=args.model_json,
            make_plots=not args.no_plots,
        )
    else:
        print("Skipping Task C training.")

    video_df, payload = export_task_d_outputs(
        frame_csv=args.frame_csv,
        model_json=args.model_json,
        video_id=args.video_id,
        output_csv=taskd_csv,
        output_json=payload_json,
        min_run_length=args.min_run_length,
        include_pwm=True,
    )

    print("\nTask D preview:")
    preview_cols = [
        "frame_index",
        "avg_brightness",
        "avg_ir",
        "mode",
        "state",
        "smoothed_mode",
        "smoothed_state",
    ]
    extra_cols = [col for col in ["active_mode", "duty_cycle", "unstable"] if col in video_df.columns]
    print(video_df[preview_cols + extra_cols].head(12).to_string(index=False))

    if args.serial_port:
        send_payload_serial(payload, port=args.serial_port, baudrate=args.baudrate)
    else:
        print("\nNo serial port provided. Payload was saved to JSON only.")


if __name__ == "__main__":
    main()
