import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import serial
import time
import struct

PROTOCOL_VERSION = 1
PACKET_HEADER = struct.Struct("<BHB")  # Version, uint16 frame ID, patch count.
VALID_STATES = (0, 1, 2, 4, 5, 6, 8, 9, 10)


def pack_states(states):
    if not 2 <= len(states) <= 254 or len(states) % 2:
        raise ValueError("Expected an even patch count between 2 and 254")
    if any(state not in VALID_STATES for state in states):
        raise ValueError("Invalid four-bit patch state")
    return bytes(
        (states[i] << 4) | states[i + 1]
        for i in range(0, len(states), 2)
    )


def crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def build_packet(frame_id, states):
    packed = pack_states(states)
    payload = PACKET_HEADER.pack(PROTOCOL_VERSION, frame_id, len(states)) + packed
    raw = payload + bytes([crc8(payload)])

    # Each version-1 payload is at most 132 bytes, so every COBS run fits
    # in one code byte. Reserve that byte and fill it when the run ends.
    encoded = bytearray([0])
    code_index = 0
    for byte in raw:
        if byte == 0:
            encoded[code_index] = len(encoded) - code_index
            code_index = len(encoded)
            encoded.append(0)
        else:
            encoded.append(byte)
    encoded[code_index] = len(encoded) - code_index
    encoded.append(0)  # Serial packet delimiter.
    return bytes(encoded)
    
# Used with Camo Cam or some other phone livestreaming app combinatation 

#Change COM5 to whatever ESP32 is connected to:
esp32 = serial.Serial("COM6", 115200, timeout=1)
time.sleep(2)  # ESP32 may reset when serial connects

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

device = 0 if torch.cuda.is_available() else -1
depth_estimator = pipeline(
    "depth-estimation",
    model=MODEL,
    device=device
)

# change this to whatever camera you want to use, 0 is default webcam, 1 is second camera, etc.
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

frame_id = 0
last_send = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    result = depth_estimator(pil_image) 

    depth = np.array(result["depth"], dtype=np.float32)


# depth is the raw depth map from your AI model (a NumPy array)

    h, w = depth.shape
    grid_values = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_h, gray_w = gray.shape
    states = []

    # Normalize the entire depth map to 0–100%.
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    depth_range = depth_max - depth_min

    if depth_range > 1e-8:
        depth_percent = (depth - depth_min) / depth_range * 100.0
    else:
        depth_percent = np.zeros_like(depth, dtype=np.float32)

    # Create the heat map.
    visual = (depth_percent * 2.55).clip(0, 255).astype(np.uint8)
    visual = cv2.applyColorMap(visual, cv2.COLORMAP_INFERNO)

    # Draw the 4x4 grid and percentage labels.
    h, w = depth_percent.shape
    cell_h = h // 4
    cell_w = w // 4

    for row in range(4):
        for col in range(4):
            y1, y2 = row * h // 4, (row + 1) * h // 4
            x1, x2 = col * w // 4, (col + 1) * w // 4

            # Average percentage for this square.
            average = float(np.mean(
                depth_percent[y1:y2, x1:x2]
            ))

            # Apply your arctangent transformation.
            x = average / 100.0
            z = np.arctan(15 * (x - 0.5)) / np.pi + 0.5
            transformed_percent = round(z * 100)

            if transformed_percent < 10:
                transformed_percent = 0
            if transformed_percent > 90:
                transformed_percent = 100
            # Match the same image region using the webcam's own dimensions, classify brightness and depth
            gy1, gy2 = row * gray_h // 4, (row + 1) * gray_h // 4
            gx1, gx2 = col * gray_w // 4, (col + 1) * gray_w // 4
            mean_gray = float(np.mean(gray[gy1:gy2, gx1:gx2]))

            if mean_gray < 85:
                brightness_level = 0
            elif mean_gray < 170:
                brightness_level = 1
            else:
                brightness_level = 2

            if transformed_percent < 33:
                depth_level = 0
            elif transformed_percent < 67:
                depth_level = 1
            else:
                depth_level = 2

            states.append((brightness_level << 2) | depth_level)

            # Find the center of the square.
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Draw square boundaries.
            cv2.rectangle(
                visual,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                1
            )

            # Draw percentage text with a dark outline.
            label = f"{transformed_percent}%"
            cv2.putText(
                visual,
                label,
                (center_x - 25, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                visual,
                label,
                (center_x - 25, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    

            grid_values.append(transformed_percent)
    # Send 16 comma-separated percentages over USB serial.
    now = time.monotonic()

    if now - last_send >= 0.05:
        packet = build_packet(frame_id, states)
        esp32.write(packet)
        frame_id = (frame_id + 1) & 0xFFFF
        last_send = now



    cv2.imshow("Live camera", frame)
    cv2.imshow("AI depth map", visual)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
