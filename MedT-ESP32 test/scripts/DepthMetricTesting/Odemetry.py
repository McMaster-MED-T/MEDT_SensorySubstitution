
import cv2
import numpy as np
import psutil
import time
import os


# ============================================================
# 1. SETTINGS
# ============================================================

VIDEO_PATH = "IMG_4228.mov"

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

GRID_COLUMNS = 4
GRID_ROWS = 6

CELL_WIDTH = FRAME_WIDTH / GRID_COLUMNS
CELL_HEIGHT = FRAME_HEIGHT / GRID_ROWS

MAX_FEATURES = 1500
FAST_THRESHOLD = 15

LK_WINDOW = (21, 21)
LK_MAX_LEVEL = 3
FB_THRESHOLD = 1.0

RANSAC_THRESHOLD = 1.0
RANSAC_CONFIDENCE = 0.999

MIN_TRACKS = 8
REDETECT_INTERVAL = 5

# Approximate camera intrinsics
FX = 0.9 * FRAME_WIDTH
FY = 0.9 * FRAME_WIDTH
CX = FRAME_WIDTH / 2
CY = FRAME_HEIGHT / 2

CAMERA_MATRIX = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
], dtype=np.float64)


# ============================================================
# 2. INITIALIZE ORB
# ============================================================

orb = cv2.ORB_create(
    nfeatures=MAX_FEATURES,
    fastThreshold=FAST_THRESHOLD
)


# ============================================================
# 3. DETECT FEATURES
# ============================================================

def detect_features(gray):

    keypoints = orb.detect(gray, None)

    if len(keypoints) == 0:
        return None

    points = np.array(
        [kp.pt for kp in keypoints],
        dtype=np.float32
    )

    return points.reshape(-1, 1, 2)


# ============================================================
# 4. TRACK FEATURES USING LUCAS-KANADE
# ============================================================

def track_features(previous_gray, current_gray, old_points):

    if old_points is None or len(old_points) == 0:
        return None, None

    new_points, status_forward, error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        old_points,
        None,
        winSize=LK_WINDOW,
        maxLevel=LK_MAX_LEVEL
    )

    if new_points is None:
        return None, None

    # Track backwards to validate the points
    backward_points, status_backward, error_backward = (
        cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            new_points,
            None,
            winSize=LK_WINDOW,
            maxLevel=LK_MAX_LEVEL
        )
    )

    if backward_points is None:
        return None, None

    old_flat = old_points.reshape(-1, 2)
    new_flat = new_points.reshape(-1, 2)
    backward_flat = backward_points.reshape(-1, 2)

    forward_status = status_forward.flatten() == 1
    backward_status = status_backward.flatten() == 1

    # Forward-backward tracking error
    fb_error = np.linalg.norm(
        old_flat - backward_flat,
        axis=1
    )

    # Check that points remain inside the image
    inside = (
        (new_flat[:, 0] >= 0) &
        (new_flat[:, 0] < FRAME_WIDTH) &
        (new_flat[:, 1] >= 0) &
        (new_flat[:, 1] < FRAME_HEIGHT)
    )

    valid = (
        forward_status &
        backward_status &
        (fb_error < FB_THRESHOLD) &
        inside
    )

    tracked_old = old_points[valid]
    tracked_new = new_points[valid]

    if len(tracked_old) == 0:
        return None, None

    return tracked_old, tracked_new


# ============================================================
# 5. ESSENTIAL MATRIX AND CAMERA POSE
# ============================================================

def estimate_camera_pose(old_points, new_points):

    if old_points is None or new_points is None:
        return None, None, 0

    if len(old_points) < MIN_TRACKS:
        return None, None, 0

    essential_matrix, mask = cv2.findEssentialMat(
        old_points,
        new_points,
        CAMERA_MATRIX,
        method=cv2.RANSAC,
        prob=RANSAC_CONFIDENCE,
        threshold=RANSAC_THRESHOLD
    )

    if essential_matrix is None or mask is None:
        return None, None, 0

    inlier_count = int(np.sum(mask))

    if inlier_count < MIN_TRACKS:
        return None, None, inlier_count

    inlier_old = old_points[mask.ravel() == 1]
    inlier_new = new_points[mask.ravel() == 1]

    try:

        recovered, rotation, translation, pose_mask = (
            cv2.recoverPose(
                essential_matrix,
                inlier_old,
                inlier_new,
                CAMERA_MATRIX
            )
        )

        return rotation, translation, int(recovered)

    except cv2.error:

        return None, None, inlier_count


# ============================================================
# 6. CALCULATE 2D GRID AVERAGES
# ============================================================

def calculate_grid_averages(old_points, new_points):

    grid_averages = np.zeros(
        (GRID_ROWS, GRID_COLUMNS),
        dtype=float
    )

    grid_counts = np.zeros(
        (GRID_ROWS, GRID_COLUMNS),
        dtype=int
    )

    if old_points is None or new_points is None:
        return grid_averages, grid_counts

    if len(old_points) == 0 or len(new_points) == 0:
        return grid_averages, grid_counts

    # Calculate 2D pixel movement between frames
    movements = np.linalg.norm(
        new_points.reshape(-1, 2) -
        old_points.reshape(-1, 2),
        axis=1
    )

    # Assign each tracked point to its current image cell
    for point, movement in zip(new_points, movements):

        x, y = point.ravel()

        column = int(x / CELL_WIDTH)
        row = int(y / CELL_HEIGHT)

        if (
            0 <= column < GRID_COLUMNS and
            0 <= row < GRID_ROWS
        ):

            grid_averages[row, column] += movement
            grid_counts[row, column] += 1

    # Calculate average movement in each cell
    valid_cells = grid_counts > 0

    grid_averages[valid_cells] /= grid_counts[valid_cells]

    return grid_averages, grid_counts


# ============================================================
# 7. DRAW TRACKED POINTS
# ============================================================

def draw_tracking(frame, old_points, new_points):

    if old_points is None or new_points is None:
        return frame

    for old, new in zip(old_points, new_points):

        x1, y1 = old.ravel()
        x2, y2 = new.ravel()

        # Motion vector
        cv2.line(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 165, 255),
            1
        )

        # Current tracked feature
        cv2.circle(
            frame,
            (int(x2), int(y2)),
            2,
            (0, 255, 0),
            -1
        )

    return frame


# ============================================================
# 8. DRAW 4 x 6 MOTION GRID
# ============================================================

def draw_motion_grid(frame, grid_averages, grid_counts):

    for row in range(GRID_ROWS):

        for column in range(GRID_COLUMNS):

            x1 = int(column * CELL_WIDTH)
            y1 = int(row * CELL_HEIGHT)

            x2 = int((column + 1) * CELL_WIDTH)
            y2 = int((row + 1) * CELL_HEIGHT)

            # Draw cell boundaries
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                1
            )

            average = grid_averages[row, column]
            count = grid_counts[row, column]

            # Average 2D movement
            motion_text = f"{average:.2f}px"
            count_text = f"Pts: {count}"

            if count == 0:
                text_color = (180, 180, 180)
            else:
                text_color = (0, 255, 255)

            # Average movement
            cv2.putText(
                frame,
                motion_text,
                (x1 + 4, y1 + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                text_color,
                1,
                cv2.LINE_AA
            )

            # Number of tracked points
            cv2.putText(
                frame,
                count_text,
                (x1 + 4, y1 + 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return frame


# ============================================================
# 9. COMPUTER STATISTICS WINDOW
# ============================================================

def draw_computer_statistics(frame, fps, tracked_count,
                             inlier_count, cpu_usage,
                             ram_usage, frame_number):

    frame[:] = 0

    information = [
        "MONOCULAR VISUAL ODOMETRY",
        "",
        f"Frame: {frame_number}",
        f"FPS: {fps:.2f}",
        f"Tracked 2D points: {tracked_count}",
        f"Essential matrix inliers: {inlier_count}",
        "",
        f"CPU usage: {cpu_usage:.1f}%",
        f"RAM usage: {ram_usage:.1f}%",
        "",
        "Grid: 4 columns x 6 rows",
        "Grid measurement: Average 2D motion",
        "",
        "Press SPACE to pause",
        "Press Q to quit"
    ]

    y = 25

    for line in information:

        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y += 20

    return frame


# ============================================================
# 10. OPEN VIDEO
# ============================================================

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Could not find video: {VIDEO_PATH}"
    )

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video file.")


# ============================================================
# 11. INITIALIZE FIRST FRAME
# ============================================================

ret, first_frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError("Could not read the first video frame.")

first_frame = cv2.resize(
    first_frame,
    (FRAME_WIDTH, FRAME_HEIGHT)
)

previous_gray = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)

old_points = detect_features(previous_gray)

if old_points is None:
    old_points = np.empty((0, 1, 2), dtype=np.float32)


# ============================================================
# 12. CREATE WINDOWS
# ============================================================

cv2.namedWindow(
    "Monocular Visual Odometry",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "Computer Statistics",
    cv2.WINDOW_NORMAL
)

stats_frame = np.zeros(
    (350, 420, 3),
    dtype=np.uint8
)


# ============================================================
# 13. MAIN LOOP
# ============================================================

frame_number = 0
paused = False

previous_time = time.time()
fps = 0

try:

    while True:

        if not paused:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT)
            )

            current_gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            frame_number += 1

            # --------------------------------------------
            # Track existing 2D features
            # --------------------------------------------

            tracked_old, tracked_new = track_features(
                previous_gray,
                current_gray,
                old_points
            )

            if tracked_old is None:

                tracked_count = 0
                tracked_old = None
                tracked_new = None

            else:

                tracked_count = len(tracked_new)

            # --------------------------------------------
            # Calculate grid using 2D points
            # --------------------------------------------

            grid_averages, grid_counts = calculate_grid_averages(
                tracked_old,
                tracked_new
            )

            # --------------------------------------------
            # Estimate camera pose
            # --------------------------------------------

            rotation, translation, inlier_count = (
                estimate_camera_pose(
                    tracked_old,
                    tracked_new
                )
            )

            # --------------------------------------------
            # Draw tracked features
            # --------------------------------------------

            display_frame = frame.copy()

            display_frame = draw_tracking(
                display_frame,
                tracked_old,
                tracked_new
            )

            # --------------------------------------------
            # Draw 4 x 6 grid
            # --------------------------------------------

            display_frame = draw_motion_grid(
                display_frame,
                grid_averages,
                grid_counts
            )

            # --------------------------------------------
            # Calculate FPS and computer statistics
            # --------------------------------------------

            current_time = time.time()

            elapsed = current_time - previous_time

            if elapsed > 0:
                fps = 1.0 / elapsed

            previous_time = current_time

            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent

            stats_frame = draw_computer_statistics(
                stats_frame,
                fps,
                tracked_count,
                inlier_count,
                cpu_usage,
                ram_usage,
                frame_number
            )

            # --------------------------------------------
            # Show windows
            # --------------------------------------------

            cv2.imshow(
                "Monocular Visual Odometry",
                display_frame
            )

            cv2.imshow(
                "Computer Statistics",
                stats_frame
            )

            # --------------------------------------------
            # Prepare next frame
            # --------------------------------------------

            if (
                frame_number % REDETECT_INTERVAL == 0 or
                tracked_new is None or
                len(tracked_new) < MIN_TRACKS
            ):

                old_points = detect_features(current_gray)

                if old_points is None:
                    old_points = np.empty(
                        (0, 1, 2),
                        dtype=np.float32
                    )

            else:

                old_points = tracked_new

            previous_gray = current_gray.copy()

        # --------------------------------------------
        # Keyboard controls
        # --------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            paused = not paused

finally:

    cap.release()
    cv2.destroyAllWindows()
