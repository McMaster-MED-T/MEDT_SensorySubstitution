
import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
#import serial
import time
import csv

# Used with Camo Cam or some other phone livestreaming app combination

# Change COM6 to whatever ESP32 is connected to:
#esp32 = serial.Serial("COM6", 115200, timeout=1)
#time.sleep(2)

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

device = 0 if torch.cuda.is_available() else -1
depth_estimator = pipeline(
    "depth-estimation",
    model=MODEL,
    device=device
)

# 0 is default webcam, 1 is second camera, etc.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

last_send = 0

# Move data from terminal to CSV
csv_file = open("depth_results.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

# Create column headers for all 24 patches.
csv_writer.writerow(
    ["Frame"] + [f"Patch_{i+1}" for i in range(24)]
)

frame_number = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    result = depth_estimator(pil_image)

    depth = np.array(result["depth"], dtype=np.float32)

    # Relative depth scores
    raw_depth = result["predicted_depth"]
    raw_depth = raw_depth.squeeze().detach().cpu().numpy()

    raw_depth = cv2.resize(
        raw_depth,
        (frame.shape[1], frame.shape[0])
    )

    h, w = depth.shape
    grid_values = []
    raw_grid_values = []

    # Normalize the entire depth map to 0–100%.
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    depth_range = depth_max - depth_min

    if depth_range > 1e-8:
        depth_percent = (
            (depth - depth_min) / depth_range * 100.0
        )
    else:
        depth_percent = np.zeros_like(
            depth, dtype=np.float32
        )

    # Create the heat map.
    visual = (depth_percent * 2.55).clip(
        0, 255
    ).astype(np.uint8)

    visual = cv2.applyColorMap(
        visual, cv2.COLORMAP_INFERNO
    )

    # Draw the 4x6 grid.
    h, w = depth_percent.shape
    num_rows = 4
    num_cols = 6

    cell_h = h // num_rows
    cell_w = w // num_cols

    for row in range(num_rows):
        for col in range(num_cols):

            y1 = row * h // num_rows
            y2 = (row + 1) * h // num_rows

            x1 = col * w // num_cols
            x2 = (col + 1) * w // num_cols

            # Calculate raw depth for this patch.
            raw_patch = raw_depth[y1:y2, x1:x2]
            raw_average = float(np.mean(raw_patch))
            raw_grid_values.append(raw_average)

            # Average normalized percentage.
            average = float(np.mean(
                depth_percent[y1:y2, x1:x2]
            ))

            transformed_percent = round(average, 1)

            # NEW: NEAR / MEDIUM / FAR THRESHOLDS

            if average >= 70:
                category = "NEAR"
                intensity = 100

            elif average >= 35:
                category = "MEDIUM"
                intensity = 50

            else:
                category = "FAR"
                intensity = 0

            # END OF THRESHOLD CLASSIFICATION
            

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

            # Display category and raw depth score.
            labels = [
                (category, center_y - 10),
                (f"{raw_average:.2f}", center_y + 20)
            ]

            for label, text_y in labels:

                font_scale = 0.5
                thickness = 1

                text_size = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    thickness
                )[0]

                text_x = center_x - text_size[0] // 2

                # Dark outline.
                cv2.putText(
                    visual,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA
                )

                # White text.
                cv2.putText(
                    visual,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

            # Store the intensity for each patch.
            grid_values.append(intensity)

    # Save raw depth values.
    now = time.monotonic()

    print("Raw depth values:", np.round(raw_grid_values, 2))

    frame_number += 1

    csv_writer.writerow(
        [frame_number] + raw_grid_values
    )

    # Send 24 comma-separated intensities over USB serial.
    # Hardware transmission is disabled for laptop testing.
    if now - last_send >= 0.05:

        packet = ",".join(map(str, grid_values)) + "\n"

        #esp32.write(packet.encode("ascii"))

        last_send = now

    cv2.imshow("Live camera", frame)
    cv2.imshow("AI depth map", visual)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
