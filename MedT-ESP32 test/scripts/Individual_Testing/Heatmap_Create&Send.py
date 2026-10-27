import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import serial
import time

# Used with Camo Cam or some other phone livestreaming app combinatation 

#Change COM5 to whatever ESP32 is connected to:
esp32 = serial.Serial("COM5", 115200, timeout=1)
time.sleep(2)  # ESP32 may reset when serial connects

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

device = 0 if torch.cuda.is_available() else -1
depth_estimator = pipeline(
    "depth-estimation",
    model=MODEL,
    device=device
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

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

    

    # Calculate average depth for each of the 16 squares,
    # then apply the arctangent transformation.
    h, w = depth_percent.shape
    grid_values = []

    for row in range(4):
        for col in range(4):
            y1, y2 = row * h // 4, (row + 1) * h // 4
            x1, x2 = col * w // 4, (col + 1) * w // 4

            average = float(np.mean(depth_percent[y1:y2, x1:x2]))

            # Convert percentage (0–100) to normalized x (0–1).
            x = average / 100.0

            # Apply your arctangent transformation.
            z = np.arctan(15 * (x - 0.5)) / np.pi + 0.5

            # Convert back to percentage (0–100) for the ESP32.
            transformed_percent = round(z * 100)

            grid_values.append(transformed_percent)
    # Send 16 comma-separated percentages over USB serial.
    now = time.monotonic()

    if now - last_send >= 0.05:
        packet = ",".join(map(str, grid_values)) + "\n"
        esp32.write(packet.encode("ascii"))
        last_send = now



    cv2.imshow("16-Square Relative Depth (%)", visual)


    cv2.imshow("Live camera", frame)
    cv2.imshow("AI depth map", visual)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
