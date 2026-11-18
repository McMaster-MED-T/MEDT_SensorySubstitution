
import cv2
import numpy as np
import serial
import time
import struct

# =========================
# CONFIGURATION
# =========================

COM_PORT = "COM6"
BAUD_RATE = 115200
CAMERA_INDEX = 0

ROWS = 4
COLS = 6
PATCH_COUNT = ROWS * COLS

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.70

MIN_BRIGHTNESS_RANGE = 5.0
SEND_INTERVAL = 0.05

PROTOCOL_VERSION = 1
PACKET_HEADER = struct.Struct("<BHB")

VALID_STATES = (0, 4, 8)


# =========================
# PACKET FUNCTIONS
# =========================

def pack_states(states):
    if not 2 <= len(states) <= 254 or len(states) % 2:
        raise ValueError("Invalid patch count")

    if any(state not in VALID_STATES for state in states):
        raise ValueError("Invalid brightness state")

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

    payload = (
        PACKET_HEADER.pack(
            PROTOCOL_VERSION,
            frame_id,
            len(states)
        )
        + packed
    )

    raw = payload + bytes([crc8(payload)])

    # COBS encoding
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
    encoded.append(0)

    return bytes(encoded)


# =========================
# CAMERA AND SERIAL SETUP
# =========================
USE_ESP32 = False

esp32 = None

if USE_ESP32:
    esp32 = serial.Serial(
        COM_PORT,
        BAUD_RATE,
        timeout=1
    )
    time.sleep(2)


#time.sleep(2)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    if esp32 is not None:
        esp32.close()
    
    raise RuntimeError("Could not open camera")


frame_id = 0
last_send = 0


# =========================
# MAIN PROCESSING LOOP
# =========================

try:
    while True:

        ok, frame = cap.read()

        if not ok:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        h, w = gray.shape

        brightness_values = []

        # -------------------------
        # STEP 1: Calculate average
        # brightness of 24 patches
        # -------------------------

        for row in range(ROWS):
            for col in range(COLS):

                y1 = row * h // ROWS
                y2 = (row + 1) * h // ROWS

                x1 = col * w // COLS
                x2 = (col + 1) * w // COLS

                patch = gray[y1:y2, x1:x2]

                average_brightness = float(
                    np.mean(patch)
                )

                brightness_values.append(
                    average_brightness
                )

        # -------------------------
        # STEP 2: Find scene min
        # and max brightness
        # -------------------------

        brightness_min = min(brightness_values)
        brightness_max = max(brightness_values)

        brightness_range = (
            brightness_max - brightness_min
        )

        # -------------------------
        # STEP 3: Calculate adaptive
        # brightness thresholds which change for each frame
        # -------------------------

        threshold_1 = (
            brightness_min
            + LOW_THRESHOLD * brightness_range
        )

        threshold_2 = (
            brightness_min
            + HIGH_THRESHOLD * brightness_range
        )

        # -------------------------
        # STEP 4: Classify patches
        # into three levels
        # -------------------------

        brightness_levels = []
        states = []

        for brightness in brightness_values:

            if brightness_range < MIN_BRIGHTNESS_RANGE:
                # Insufficient contrast:
                # classification is unreliable.
                level = 0

            elif brightness < threshold_1:
                level = 0

            elif brightness < threshold_2:
                level = 1

            else:
                level = 2

            brightness_levels.append(level)

            # Preserve Larry's original bit layout:
            # Bits 3-2: brightness (0, 1, 2)
            # Bits 1-0: unused (0)
            state = level << 2

            states.append(state)

        # -------------------------
        # STEP 5: Draw 4x6 grid
        # and brightness labels
        # -------------------------

        visual = frame.copy()

        for row in range(ROWS):
            for col in range(COLS):

                index = row * COLS + col

                y1 = row * h // ROWS
                y2 = (row + 1) * h // ROWS

                x1 = col * w // COLS
                x2 = (col + 1) * w // COLS

                level = brightness_levels[index]
                brightness = brightness_values[index]

                if level == 0:
                    color = (255, 0, 0)  # Blue
                elif level == 1:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red

                cv2.rectangle(
                    visual,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = (
                    f"P{index + 1}: "
                    f"{brightness:.0f} "
                    f"L{level}"
                )

                cv2.putText(
                    visual,
                    label,
                    (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA
                )

        # Display adaptive thresholds
        info = (
            f"Min: {brightness_min:.1f}  "
            f"Max: {brightness_max:.1f}  "
            f"Threshold 1: {threshold_1:.1f}"
            f"Threshold 2: {threshold_2:.1f}"
        )

        cv2.putText(
            visual,
            info,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # -------------------------
        # STEP 6: Send 24 states
        # to ESP32
        # -------------------------

        now = time.monotonic()

        if now - last_send >= SEND_INTERVAL:

            packet = build_packet(
                frame_id,
                states
            )

            if USE_ESP32:
                esp32.write(packet)
            else:
                print("Raw Brightness Values:", [round(value, 1) for value in brightness_values])
                print("24 Brightness Levels:", brightness_levels)
                print("24 Packet States:", states)
                print("Packet Bytes:", packet.hex(" "))
            

            print(
                f"Frame {frame_id} | "
                f"Min={brightness_min:.1f} | "
                f"Max={brightness_max:.1f} | "
                f"T1={threshold_1:.1f} | "
                f"T2={threshold_2:.1f}"
            )

            print(
                "Brightness levels:",
                brightness_levels
            )

            print(
                "Packet states:",
                states
            )

            frame_id = (frame_id + 1) & 0xFFFF

            last_send = now

        # -------------------------
        # STEP 7: Display video
        # -------------------------

        cv2.imshow(
            "Adaptive Brightness - 24 Patches",
            visual
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    if esp32 is not None:
        esp32.close()
    cv2.destroyAllWindows()
