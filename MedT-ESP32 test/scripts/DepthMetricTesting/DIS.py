
import cv2
import numpy as np
import time

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================
# 1. SETTINGS
# ============================================================

VIDEO_PATH = "IMG_4228.mov"

# Grid dimensions
GRID_ROWS = 6
GRID_COLUMNS = 4

# Dense optical flow settings
DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM

FLOW_VECTOR_SPACING = 12

# Ignore very small and unusually large motion vectors
MIN_FLOW_MAGNITUDE = 0.5
MAX_FLOW_MAGNITUDE = 30

# Vector display scaling
FLOW_DISPLAY_SCALE = 2


# ============================================================
# 2. INITIALIZE DIS
# ============================================================

dis = cv2.DISOpticalFlow_create(DIS_PRESET)

dis.setFinestScale(1)
dis.setPatchSize(8)
dis.setPatchStride(4)
dis.setGradientDescentIterations(25)


# ============================================================
# 3. CALCULATE GRID DIMENSIONS
# ============================================================

def calculate_grid_dimensions(frame):

    height, width = frame.shape[:2]

    cell_width = width // GRID_COLUMNS
    cell_height = height // GRID_ROWS

    return width, height, cell_width, cell_height


# ============================================================
# 4. CALCULATE DENSE FLOW GRID
# ============================================================

def calculate_dense_flow_grid(flow, cell_width, cell_height):

    height, width = flow.shape[:2]

    grid_averages = np.zeros(
        (GRID_ROWS, GRID_COLUMNS),
        dtype=float
    )

    grid_counts = np.zeros(
        (GRID_ROWS, GRID_COLUMNS),
        dtype=int
    )

    # Calculate motion magnitude for every pixel
    magnitude = cv2.magnitude(
        flow[:, :, 0],
        flow[:, :, 1]
    )

    for row in range(GRID_ROWS):

        for column in range(GRID_COLUMNS):

            x1 = column * cell_width
            x2 = min(x1 + cell_width, width)

            y1 = row * cell_height
            y2 = min(y1 + cell_height, height)

            cell_motion = magnitude[y1:y2, x1:x2]

            if cell_motion.size > 0:

                grid_averages[row, column] = np.mean(
                    cell_motion
                )

                grid_counts[row, column] = cell_motion.size

    return grid_averages, grid_counts


# ============================================================
# 5. DRAW DENSE FLOW VECTORS
# ============================================================

def draw_dense_flow_vectors(frame, flow):

    output = frame.copy()

    height, width = frame.shape[:2]

    for y in range(0, height, FLOW_VECTOR_SPACING):

        for x in range(0, width, FLOW_VECTOR_SPACING):

            dx = flow[y, x, 0]
            dy = flow[y, x, 1]

            magnitude = np.sqrt(dx**2 + dy**2)

            if (
                magnitude < MIN_FLOW_MAGNITUDE
                or magnitude > MAX_FLOW_MAGNITUDE
            ):
                continue

            start_point = (x, y)

            end_point = (
                int(x + dx * FLOW_DISPLAY_SCALE),
                int(y + dy * FLOW_DISPLAY_SCALE)
            )

            cv2.arrowedLine(
                output,
                start_point,
                end_point,
                (0, 255, 0),
                1,
                tipLength=0.3
            )

    return output


# ============================================================
# 6. DRAW GRID AND MOTION VALUES
# ============================================================

def draw_dense_flow_grid(
    frame,
    grid_averages,
    cell_width,
    cell_height
):

    output = frame.copy()

    height, width = frame.shape[:2]

    for row in range(GRID_ROWS):

        for column in range(GRID_COLUMNS):

            x1 = column * cell_width
            x2 = min(x1 + cell_width, width)

            y1 = row * cell_height
            y2 = min(y1 + cell_height, height)

            average_motion = grid_averages[row, column]

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                1
            )

            cv2.putText(
                output,
                f"{average_motion:.2f} px",
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

    return output


# ============================================================
# 7. COMPUTER STATISTICS
# ============================================================

def draw_computer_statistics(
    frame_count,
    average_motion,
    processing_fps
):

    stats_frame = np.zeros(
        (300, 420, 3),
        dtype=np.uint8
    )

    y = 35

    cv2.putText(
        stats_frame,
        "COMPUTER STATISTICS",
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    y += 45

    statistics = [
        f"Frame: {frame_count}",
        f"Average dense motion: {average_motion:.2f} px",
        f"Processing FPS: {processing_fps:.2f}"
    ]

    if PSUTIL_AVAILABLE:

        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent

        statistics.append(f"CPU usage: {cpu:.1f}%")
        statistics.append(f"RAM usage: {ram:.1f}%")

    else:

        statistics.append("Install psutil for system statistics")

    for item in statistics:

        cv2.putText(
            stats_frame,
            item,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

        y += 35

    cv2.imshow(
        "Computer Statistics",
        stats_frame
    )


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Could not open video.")
    print("Check that IMG_4228.mov is in the same folder.")

    raise SystemExit


# Read first frame
ret, first_frame = cap.read()

if not ret:

    print("ERROR: Could not read first video frame.")

    cap.release()

    raise SystemExit


# Use original video resolution
FRAME_HEIGHT, FRAME_WIDTH = first_frame.shape[:2]

print("Original video resolution:")
print(FRAME_WIDTH, "x", FRAME_HEIGHT)

print("Grid:")
print(GRID_COLUMNS, "columns x", GRID_ROWS, "rows")


# Calculate grid cell dimensions
FRAME_WIDTH, FRAME_HEIGHT, CELL_WIDTH, CELL_HEIGHT = (
    calculate_grid_dimensions(first_frame)
)


# Convert first frame to grayscale
previous_gray = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)


# Initialize variables
frame_count = 0
paused = False

print("DIS dense optical flow started.")
print("Press SPACE to pause or resume.")
print("Press Q to quit.")


# ============================================================
# 9. VIDEO LOOP
# ============================================================

while True:

    if paused:

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            paused = False

        continue


    start_time = time.time()

    ret, frame = cap.read()

    if not ret:

        print("Video finished.")

        break


    # Do not resize the video
    current_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    frame_count += 1


    # --------------------------------------------------------
    # A. CALCULATE DIS DENSE OPTICAL FLOW
    # --------------------------------------------------------

    dense_flow = dis.calc(
        previous_gray,
        current_gray,
        None
    )


    # --------------------------------------------------------
    # B. CALCULATE GRID AVERAGES
    # --------------------------------------------------------

    dense_grid_averages, dense_grid_counts = (
        calculate_dense_flow_grid(
            dense_flow,
            CELL_WIDTH,
            CELL_HEIGHT
        )
    )

    average_dense_motion = float(
        np.mean(dense_grid_averages)
    )


    # --------------------------------------------------------
    # C. DRAW MOTION VECTORS
    # --------------------------------------------------------

    output_frame = draw_dense_flow_vectors(
        frame,
        dense_flow
    )


    # --------------------------------------------------------
    # D. DRAW GRID
    # --------------------------------------------------------

    output_frame = draw_dense_flow_grid(
        output_frame,
        dense_grid_averages,
        CELL_WIDTH,
        CELL_HEIGHT
    )


    # --------------------------------------------------------
    # E. DISPLAY FLOW WINDOW
    # --------------------------------------------------------

    cv2.imshow(
        "DIS Dense Optical Flow",
        output_frame
    )


    # --------------------------------------------------------
    # F. COMPUTER STATISTICS
    # --------------------------------------------------------

    processing_time = time.time() - start_time

    if processing_time > 0:

        processing_fps = 1.0 / processing_time

    else:

        processing_fps = 0


    draw_computer_statistics(
        frame_count,
        average_dense_motion,
        processing_fps
    )


    # --------------------------------------------------------
    # G. UPDATE PREVIOUS FRAME
    # --------------------------------------------------------

    previous_gray = current_gray.copy()


    # --------------------------------------------------------
    # H. KEYBOARD CONTROLS
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):

        break

    elif key == ord(' '):

        paused = True


# ============================================================
# 10. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Program terminated.")
