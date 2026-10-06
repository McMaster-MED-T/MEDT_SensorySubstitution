import json
import time
import matplotlib.pyplot as plt

STATE_LABELS = {
    0: "00 (Low B, Low IR)",
    1: "01 (Low B, High IR)",
    2: "10 (High B, Low IR)",
    3: "11 (High B, High IR)"
}

def simulate(payload_file):
    with open(payload_file, "r") as f:
        payload = json.load(f)

    fps = payload["fps"]
    states = payload["states"]
    delay = 1.0 / fps

    print(f"Simulating ESP32 playback...")
    print(f"FPS: {fps}, Delay: {delay:.3f}s")
    print(f"Total frames: {len(states)}\n")

    start = time.time()

    for i, state in enumerate(states):
        elapsed = time.time() - start

        print(f"Frame {i:04d} | t={elapsed:.2f}s | State {state} → {STATE_LABELS[state]}")
        time.sleep(delay)

    # Plot
    states = payload["states"]

    plt.plot(states)
    plt.title("State vs Frame")
    plt.xlabel("Frame")
    plt.ylabel("State (0-3)")
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    simulate("data/video_1_esp32_payload.json")