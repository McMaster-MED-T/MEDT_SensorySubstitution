import json
import serial
import time

PORT = "COM7"
BAUD = 115200

payload = {
    "fps": 12,
    "num_frames": 36,
    "states": [0,0,0,0,0,0, 1,1,1,1,1,1,
               0,0,0,0,0,0, 1,1,1,1,1,1,
               0,0,0,0,0,0, 1,1,1,1,1,1]
}

ser = serial.Serial(PORT, BAUD, timeout=2)

# give ESP32 time to reset
time.sleep(2)

msg = json.dumps(payload) + "\n"
ser.write(msg.encode("utf-8"))

print("Sent payload:")
print(msg)

ser.close()  