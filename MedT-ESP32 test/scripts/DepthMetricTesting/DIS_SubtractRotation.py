# Same DIS method but using an average flow of the entire frame to subtract estimated camera rotation vectors at the expense of FPS
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

# Processing resolution
PROCESS_WIDTH = 580
PROCESS_HEIGHT = 720

# DIS dense optical flow settings
DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST

# ORB + RANSAC camera-motion estimation settings
ORB_FEATURES = 1000
RANSAC_REPROJECTION_THRESHOLD = 3.0
MINIMUM_GOOD_MATCHES = 8

# Dense flow vector display settings
FLOW_VECTOR_SPACING = 12
MIN_FLOW_MAGNITUDE = 0.5
MAX_FLOW_MAGNITUDE = 30
FLOW_DISPLAY_SCALE = 2

# Percentage display:
# A residual motion of 20 pixels or more is displayed as 100%.
MAX_RELATIVE_MOTION_PX = 20.0


# ============================================================
# 2. INITIALIZE DIS AND ORB
# ============================================================

dis = cv2.DISOpticalFlow_create(DIS_PRESET)

orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


# ============================================================
# 3. CALCULATE GRID DIMENSIONS
# ============================================================

def calculate_grid_dimensions(frame):
    height, width = frame.shape[:2]

    cell_width = width // GRID_COLUMNS
    cell_height = height // GRID_ROWS

    return width, height, cell_width, cell_height


# ============================================================
# 4. ESTIMATE GLOBAL CAMERA-INDUCED FLOW
# ============================================================

def estimate_global_camera_flow(previous_gray, current_gray):
    """
    Estimate the dominant image transformation between frames using
    ORB feature matching and a RANSAC homography.

    The homography approximates image motion caused by camera rotation.
    It can also absorb some translation and dominant scene motion, so
    the residual should be interpreted as approximate parallax, not
    perfectly isolated physical translation.
    """

    height, width = previous_gray.shape[:2]

    keypoints1, descriptors1 = orb.detectAndCompute(
        previous_gray, None
    )
    keypoints2, descriptors2 = orb.detectAndCompute(
        current_gray, None
    )

    # Not enough features/descriptors to estimate global motion
    if (
        descriptors1 is None
        or descriptors2 is None
        or len(keypoints1) < MINIMUM_GOOD_MATCHES
        or len(keypoints2) < MINIMUM_GOOD_MATCHES
    ):
        return None, 0, 0

    # K-nearest-neighbour matching with Lowe's ratio test
    knn_matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in knn_matches:
        if len(pair) == 2:
            first_match, second_match = pair

            if first_match.distance < 0.75 * second_match.distance:
                good_matches.append(first_match)

    if len(good_matches) < MINIMUM_GOOD_MATCHES:
        return None, len(good_matches), 0

    points_previous = np.float32([
        keypoints1[m.queryIdx].pt for m in good_matches
    ])

    points_current = np.float32([
        keypoints2[m.trainIdx].pt for m in good_matches
    ])

    # Estimate transformation from previous-frame coordinates
    # to current-frame coordinates.
    homography, inlier_mask = cv2.findHomography(
        points_previous,
        points_current,
        cv2.RANSAC,
        RANSAC_REPROJECTION_THRESHOLD
    )

    if homography is None or inlier_mask is None:
        return None, len(good_matches), 0

    inlier_count = int(np.sum(inlier_mask))

    # Generate a coordinate for every pixel in the previous frame
    x_coordinates, y_coordinates = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )

    pixel_coordinates = np.stack(
        (x_coordinates, y_coordinates),
        axis=-1
    )

    # Predict where each pixel moves under the estimated global transform
    predicted_positions = cv2.perspectiveTransform(
        pixel_coordinates.reshape(-1, 1, 2),
        homography
    )

    predicted_positions = predicted_positions.reshape(
        height, width, 2
    )

    # Global camera-induced flow: predicted position - original position
    global_flow = predicted_positions - pixel_coordinates

    return global_flow.astype(np.float32), len(good_matches), inlier_count


# ============================================================
# 5. CALCULATE DENSE FLOW GRID
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

    # Calculate residual motion magnitude for every pixel
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
                grid_averages[row, column] = np.mean(cell_motion)
                grid_counts[row, column] = cell_motion.size

    return grid_averages, grid_counts


# ============================================================
# 6. DRAW DENSE FLOW VECTORS
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
# 7. DRAW GRID AND RESIDUAL MOTION VALUES
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
                f"{average_motion:.2f} px residual",
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

            relative_magnitude = np.clip(
                average_motion / MAX_RELATIVE_MOTION_PX * 100,
                0,
                100
            )

            # Keep the existing three-level colour scheme
            if relative_magnitude < 22.5:
                color = (230, 216, 173)
            elif relative_magnitude < 45:
                color = (0, 200, 255)
            else:
                color = (0, 0, 255)

            cv2.putText(
                output,
                f"{relative_magnitude:.2f}% relative mag",
                (x1 + 5, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

    return output


# ============================================================
# 8. COMPUTER STATISTICS
# ============================================================

def draw_computer_statistics(
    frame_count,
    average_motion,
    processing_fps,
    camera_status,
    good_matches,
    inlier_count
):

    stats_frame = np.zeros(
        (340, 460, 3),
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
        f"Average residual motion: {average_motion:.2f} px",
        f"Processing FPS: {processing_fps:.2f}",
        f"Camera compensation: {camera_status}",
        f"ORB good matches: {good_matches}",
        f"RANSAC inliers: {inlier_count}"
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
            0.48,
            (0, 255, 255),
            1
        )

        y += 34

    cv2.imshow(
        "Computer Statistics",
        stats_frame
    )


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    print(f"Check that {VIDEO_PATH} is in the same folder.")
    raise SystemExit


# Read first frame
ret, first_frame = cap.read()

if not ret:
    print("ERROR: Could not read first video frame.")
    cap.release()
    raise SystemExit


# Set processing resolution
first_frame = cv2.resize(
    first_frame,
    (PROCESS_WIDTH, PROCESS_HEIGHT)
)

FRAME_WIDTH, FRAME_HEIGHT, CELL_WIDTH, CELL_HEIGHT = (
    calculate_grid_dimensions(first_frame)
)

print("Processing resolution:")
print(FRAME_WIDTH, "x", FRAME_HEIGHT)

print("Grid:")
print(GRID_COLUMNS, "columns x", GRID_ROWS, "rows")


# Convert first frame to grayscale
previous_gray = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)


# Initialize variables
frame_count = 0
paused = False

print("DIS dense optical flow with global motion compensation started.")
print("ORB + RANSAC estimates the dominant global image transformation.")
print("Press SPACE to pause or resume.")
print("Press Q to quit.")


# ============================================================
# 10. VIDEO LOOP
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

    # Resize every frame to the same processing resolution
    frame = cv2.resize(
        frame,
        (PROCESS_WIDTH, PROCESS_HEIGHT)
    )

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
    # B. ESTIMATE GLOBAL CAMERA-INDUCED FLOW
    # --------------------------------------------------------

    global_flow, good_matches, inlier_count = (
        estimate_global_camera_flow(
            previous_gray,
            current_gray
        )
    )

    if global_flow is not None:
        # Subtract estimated dominant global motion from DIS flow.
        # The result approximates parallax and independently moving objects.
        residual_flow = dense_flow - global_flow
        camera_status = "Compensated"
    else:
        # If estimation fails, retain the original DIS flow rather than
        # incorrectly displaying zero motion.
        residual_flow = dense_flow
        camera_status = "Unavailable - raw DIS"


    # --------------------------------------------------------
    # C. CALCULATE GRID AVERAGES FROM RESIDUAL FLOW
    # --------------------------------------------------------

    residual_grid_averages, residual_grid_counts = (
        calculate_dense_flow_grid(
            residual_flow,
            CELL_WIDTH,
            CELL_HEIGHT
        )
    )

    average_residual_motion = float(
        np.mean(residual_grid_averages)
    )


    # --------------------------------------------------------
    # D. DRAW RESIDUAL MOTION VECTORS
    # --------------------------------------------------------

    output_frame = draw_dense_flow_vectors(
        frame,
        residual_flow
    )


    # --------------------------------------------------------
    # E. DRAW GRID
    # --------------------------------------------------------

    output_frame = draw_dense_flow_grid(
        output_frame,
        residual_grid_averages,
        CELL_WIDTH,
        CELL_HEIGHT
    )


    # --------------------------------------------------------
    # F. DISPLAY FLOW WINDOW
    # --------------------------------------------------------

    cv2.imshow(
        "DIS Residual Optical Flow",
        output_frame
    )


    # --------------------------------------------------------
    # G. COMPUTER STATISTICS
    # --------------------------------------------------------

    processing_time = time.time() - start_time

    if processing_time > 0:
        processing_fps = 1.0 / processing_time
    else:
        processing_fps = 0

    draw_computer_statistics(
        frame_count,
        average_residual_motion,
        processing_fps,
        camera_status,
        good_matches,
        inlier_count
    )


    # --------------------------------------------------------
    # H. UPDATE PREVIOUS FRAME
    # --------------------------------------------------------

    previous_gray = current_gray.copy()


    # --------------------------------------------------------
    # I. KEYBOARD CONTROLS
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        paused = True


# ============================================================
# 11. CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Program terminated.")
