#!/usr/bin/env python3
"""
all_in_one_depth_edge_pipeline.py

sc pipeline for depth and edge density combo 
    camera/video frame
      -> Depth Anything V2 Small inference
      -> 6x4 depth patches
      -> 6x4 Sobel / edge-density patches
      -> classify each metric into 3 levels
      -> combine into 9 states (0..8)
      -> pack 24 states into 12 bytes (2 patches per byte)
    

Dependencies:
    pip install opencv-python numpy pillow torch transformers

Usage:
    python all_in_one_depth_edge_pipeline.py 0
    python all_in_one_depth_edge_pipeline.py myvideo.mp4 --csv results.csv --display

"""

import argparse
import csv
import struct
import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


ROWS = 6
COLS = 4
PATCHES = ROWS * COLS

LOW = 0
MEDIUM = 1
HIGH = 2

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# Placeholder
START_MARKER = b"\xAA\x55"


@dataclass
class Result:
    depth_values: np.ndarray
    edge_values: np.ndarray
    depth_states: np.ndarray
    edge_states: np.ndarray
    combined_states: np.ndarray
    depth_thresholds: Tuple[float, float]
    edge_thresholds: Tuple[float, float]
    depth_ms: float
    total_ms: float


class DepthAnythingV2:
    """Self-contained Depth Anything V2 inference wrapper."""

    def __init__(self, model_id=MODEL_ID, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        print(f"Loading {model_id} on {self.device} ...")

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns predicted depth resized to the original frame size.
        Values are kept RAW rather than normalized to 0..255.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        processed = self.processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(h, w)],
        )

        depth = processed[0]["predicted_depth"]
        return depth.detach().float().cpu().numpy()


def patch_means(metric_map: np.ndarray, rows=ROWS, cols=COLS) -> np.ndarray:
    h, w = metric_map.shape[:2]
    ys = np.linspace(0, h, rows + 1, dtype=int)
    xs = np.linspace(0, w, cols + 1, dtype=int)

    out = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            patch = metric_map[ys[r]:ys[r+1], xs[c]:xs[c+1]]
            out[r, c] = float(np.mean(patch))

    return out


def sobel_patch_values(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return patch_means(mag)


def edge_density_patch_values(frame_bgr: np.ndarray,
                              canny_low=60,
                              canny_high=150) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edge_map = (edges > 0).astype(np.float32)
    return patch_means(edge_map)


def percentile_thresholds(values: np.ndarray,
                          low_p=33.0,
                          high_p=67.0) -> Tuple[float, float]:
    flat = values.flatten()
    lo = float(np.percentile(flat, low_p))
    hi = float(np.percentile(flat, high_p))

    if hi <= lo:
        hi = lo + 1e-6

    return lo, hi


def classify_three_level(values: np.ndarray,
                         thresholds: Tuple[float, float],
                         higher_means_high=True) -> np.ndarray:
    lo, hi = thresholds
    states = np.full(values.shape, MEDIUM, dtype=np.uint8)

    if higher_means_high:
        states[values < lo] = LOW
        states[values >= hi] = HIGH
    else:
        states[values > hi] = LOW
        states[values <= lo] = HIGH

    return states


def combine_states(edge_states: np.ndarray,
                   depth_states: np.ndarray) -> np.ndarray:
    """
    3 edge levels x 3 depth levels = 9 states.

    combined = edge*3 + depth

    0 = edge low,  depth far/low
    1 = edge low,  depth medium
    2 = edge low,  depth near/high
    3 = edge med,  depth far/low
    4 = edge med,  depth medium
    5 = edge med,  depth near/high
    6 = edge high, depth far/low
    7 = edge high, depth medium
    8 = edge high, depth near/high
    """
    return edge_states.astype(np.uint8) * 3 + depth_states.astype(np.uint8)


def pack_4bit_states(states: np.ndarray) -> bytes:
    flat = [int(x) for x in states.flatten(order="C")]

    if len(flat) != 24:
        raise ValueError(f"Expected 24 states, got {len(flat)}")

    if any(x < 0 or x > 15 for x in flat):
        raise ValueError("State must fit in 4 bits")

    packed = bytearray()

    for i in range(0, 24, 2):
        packed.append((flat[i] << 4) | flat[i + 1])

    return bytes(packed)


def checksum_byte(payload: bytes) -> int:
    """
    TEMPORARY checksum only.
    Replace with Larry's actual checksum algorithm.
    """
    return sum(payload) & 0xFF


def build_packet(combined_states: np.ndarray, frame_id: int) -> bytes:
    packed_states = pack_4bit_states(combined_states)
    frame_bytes = struct.pack(">H", frame_id & 0xFFFF)

    body = frame_bytes + packed_states
    packet = START_MARKER + body + bytes([checksum_byte(body)])

    assert len(packet) == 17
    return packet


def analyze_frame(frame_bgr: np.ndarray,
                  depth_model: DepthAnythingV2,
                  edge_method="edge_density",
                  depth_higher_is_nearer=True) -> Result:

    t0 = time.perf_counter()

    td0 = time.perf_counter()
    depth_map = depth_model.infer(frame_bgr)
    depth_ms = (time.perf_counter() - td0) * 1000.0

    depth_values = patch_means(depth_map)

    if edge_method == "edge_density":
        edge_values = edge_density_patch_values(frame_bgr)
    elif edge_method == "sobel":
        edge_values = sobel_patch_values(frame_bgr)
    else:
        raise ValueError("edge_method must be 'edge_density' or 'sobel'")

    depth_t = percentile_thresholds(depth_values)
    edge_t = percentile_thresholds(edge_values)

    depth_states = classify_three_level(
        depth_values,
        depth_t,
        higher_means_high=depth_higher_is_nearer,
    )

    edge_states = classify_three_level(
        edge_values,
        edge_t,
        higher_means_high=True,
    )

    combined = combine_states(edge_states, depth_states)

    total_ms = (time.perf_counter() - t0) * 1000.0

    return Result(
        depth_values=depth_values,
        edge_values=edge_values,
        depth_states=depth_states,
        edge_states=edge_states,
        combined_states=combined,
        depth_thresholds=depth_t,
        edge_thresholds=edge_t,
        depth_ms=depth_ms,
        total_ms=total_ms,
    )


def normalized_depth_for_display(depth_map: np.ndarray) -> np.ndarray:
    mn = float(np.min(depth_map))
    mx = float(np.max(depth_map))

    if mx <= mn:
        return np.zeros(depth_map.shape, dtype=np.uint8)

    norm = (depth_map - mn) / (mx - mn)
    return (norm * 255).astype(np.uint8)


def annotate(frame_bgr: np.ndarray, result: Result) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    ys = np.linspace(0, h, ROWS + 1, dtype=int)
    xs = np.linspace(0, w, COLS + 1, dtype=int)

    for r in range(ROWS):
        for c in range(COLS):
            x0, x1 = xs[c], xs[c+1]
            y0, y1 = ys[r], ys[r+1]

            cv2.rectangle(out, (x0, y0), (x1, y1),
                          (255, 255, 255), 1)

            e = int(result.edge_states[r, c])
            d = int(result.depth_states[r, c])
            combined = int(result.combined_states[r, c])

            label = f"E{e} D{d} C{combined}"

            cv2.putText(
                out,
                label,
                (x0 + 3, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return out


def state_change_rate(previous: np.ndarray,
                      current: np.ndarray) -> float:
    return float(np.mean(previous != current))


def write_header(writer):
    writer.writerow(
        ["frame", "depth_ms", "total_ms",
         "depth_low_t", "depth_high_t",
         "edge_low_t", "edge_high_t"]
        + [f"depth_value_{i}" for i in range(PATCHES)]
        + [f"edge_value_{i}" for i in range(PATCHES)]
        + [f"depth_state_{i}" for i in range(PATCHES)]
        + [f"edge_state_{i}" for i in range(PATCHES)]
        + [f"combined_state_{i}" for i in range(PATCHES)]
    )


def run(source,
        edge_method="edge_density",
        csv_path=None,
        display=False,
        process_every=1,
        depth_higher_is_nearer=True):

    model = DepthAnythingV2()

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    csv_file = open(csv_path, "w", newline="") if csv_path else None
    writer = csv.writer(csv_file) if csv_file else None

    if writer:
        write_header(writer)

    previous_combined = None
    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if not ok:
                break

            if frame_id % process_every != 0:
                frame_id += 1
                continue

            result = analyze_frame(
                frame,
                model,
                edge_method=edge_method,
                depth_higher_is_nearer=depth_higher_is_nearer,
            )

            packed12 = pack_4bit_states(result.combined_states)
            packet17 = build_packet(result.combined_states, frame_id)

            change = None
            if previous_combined is not None:
                change = state_change_rate(
                    previous_combined,
                    result.combined_states
                )

            print("\nFRAME", frame_id)
            print("Depth patch values:")
            print(np.round(result.depth_values, 2))

            print("Edge patch values:")
            print(np.round(result.edge_values, 3))

            print("Depth states:")
            print(result.depth_states)

            print("Edge states:")
            print(result.edge_states)

            print("Combined 0-8 states:")
            print(result.combined_states)

            print(f"Depth inference: {result.depth_ms:.1f} ms")
            print(f"Total processing: {result.total_ms:.1f} ms")
            print(f"12-byte state payload: {packed12.hex()}")
            print(f"17-byte packet skeleton: {packet17.hex()}")

            if change is not None:
                print(f"Combined-state change rate: {change:.1%}")

            if writer:
                writer.writerow(
                    [
                        frame_id,
                        result.depth_ms,
                        result.total_ms,
                        result.depth_thresholds[0],
                        result.depth_thresholds[1],
                        result.edge_thresholds[0],
                        result.edge_thresholds[1],
                    ]
                    + result.depth_values.flatten().tolist()
                    + result.edge_values.flatten().tolist()
                    + result.depth_states.flatten().tolist()
                    + result.edge_states.flatten().tolist()
                    + result.combined_states.flatten().tolist()
                )

            if display:
                shown = annotate(frame, result)
                cv2.imshow("Depth + Edge Combined States", shown)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            previous_combined = result.combined_states.copy()
            frame_id += 1

    finally:
        cap.release()

        if csv_file:
            csv_file.close()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "source",
        help="camera index (e.g. 0) or video path"
    )

    parser.add_argument(
        "--edge-method",
        choices=["edge_density", "sobel"],
        default="edge_density",
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="optional CSV results path"
    )

    parser.add_argument(
        "--display",
        action="store_true"
    )

    parser.add_argument(
        "--process-every",
        type=int,
        default=1,
        help="run expensive depth inference every Nth frame"
    )

    parser.add_argument(
        "--depth-far-is-high",
        action="store_true",
        help="invert depth classification if larger model output means farther"
    )

    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run(
        source,
        edge_method=args.edge_method,
        csv_path=args.csv,
        display=args.display,
        process_every=max(1, args.process_every),
        depth_higher_is_nearer=not args.depth_far_is_high,
    )


if __name__ == "__main__":
    main()
