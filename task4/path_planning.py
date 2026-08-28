import cv2
import numpy as np
import os
import heapq
import math


# ============================================================
# SETTINGS
# ============================================================

OBSTACLE_MARGIN = 14
POTHOLE_MARGIN = 12
GRID_STEP = 5


# ============================================================
# ROAD DETECTION
# ============================================================

def detect_road(image):
    """
    Detect the grey road area from the aerial image.
    The road is slightly brighter than the surrounding background.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, w = gray.shape

    # Estimate background brightness from image borders
    border_pixels = np.concatenate([
        gray[:20, :].ravel(),
        gray[-20:, :].ravel(),
        gray[:, :20].ravel(),
        gray[:, -20:].ravel()
    ])

    background_value = float(np.median(border_pixels))

    # Road is generally brighter than the background
    threshold_value = background_value + 3

    road_mask = np.zeros_like(gray)

    road_mask[gray > threshold_value] = 255

    # Road has low saturation because it is grey
    road_mask[hsv[:, :, 1] < 55] = np.minimum(
        road_mask[hsv[:, :, 1] < 55], 255
    )

    # Morphological cleanup
    kernel1 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (11, 11)
    )

    kernel2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (21, 21)
    )

    road_mask = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_CLOSE,
        kernel2
    )

    road_mask = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_OPEN,
        kernel1
    )

    # Keep large connected road regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        road_mask,
        8
    )

    clean_mask = np.zeros_like(road_mask)

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]

        # Keep largest components
        order = np.argsort(areas)[::-1]

        for idx in order[:3]:
            label = idx + 1

            if stats[label, cv2.CC_STAT_AREA] > 5000:
                clean_mask[labels == label] = 255

    # Fill holes inside road
    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(clean_mask)

    for contour in contours:
        if cv2.contourArea(contour) > 3000:
            cv2.drawContours(
                filled,
                [contour],
                -1,
                255,
                -1
            )

    return filled


# ============================================================
# OBSTACLE DETECTION
# ============================================================

def detect_obstacles(image, road_mask):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    masks = []

    # Yellow / brown / orange objects
    masks.append(
        cv2.inRange(
            hsv,
            np.array([8, 45, 40]),
            np.array([40, 255, 255])
        )
    )

    # Blue objects
    masks.append(
        cv2.inRange(
            hsv,
            np.array([90, 50, 30]),
            np.array([135, 255, 255])
        )
    )

    # Red objects
    masks.append(
        cv2.inRange(
            hsv,
            np.array([0, 60, 30]),
            np.array([10, 255, 255])
        )
    )

    masks.append(
        cv2.inRange(
            hsv,
            np.array([170, 60, 30]),
            np.array([180, 255, 255])
        )
    )

    # Green objects
    masks.append(
        cv2.inRange(
            hsv,
            np.array([35, 40, 30]),
            np.array([90, 255, 255])
        )
    )

    combined = np.zeros_like(masks[0])

    for mask in masks:
        combined = cv2.bitwise_or(
            combined,
            mask
        )

    # Only objects lying on the road
    combined = cv2.bitwise_and(
        combined,
        road_mask
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        kernel
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    obstacle_mask = np.zeros_like(combined)
    obstacle_boxes = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 30:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w > image.shape[1] * 0.4:
            continue

        if h > image.shape[0] * 0.4:
            continue

        obstacle_boxes.append((x, y, w, h))

        cv2.drawContours(
            obstacle_mask,
            [contour],
            -1,
            255,
            -1
        )

    return obstacle_mask, obstacle_boxes


# ============================================================
# POTHOLE DETECTION
# ============================================================

def detect_potholes(image, road_mask):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Dark regions
    dark_mask = cv2.inRange(
        gray,
        20,
        75
    )

    # Keep only road area
    dark_mask = cv2.bitwise_and(
        dark_mask,
        road_mask
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    pothole_mask = np.zeros_like(dark_mask)
    pothole_boxes = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 20 or area > 5000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w > image.shape[1] * 0.25:
            continue

        if h > image.shape[0] * 0.25:
            continue

        pothole_boxes.append(
            (x, y, w, h)
        )

        cv2.drawContours(
            pothole_mask,
            [contour],
            -1,
            255,
            -1
        )

    return pothole_mask, pothole_boxes


# ============================================================
# FIND START POINT
# ============================================================

def find_start(image, road_mask):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Bright / white pixels
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 150]),
        np.array([180, 100, 255])
    )

    # Connected components
    num, labels, stats, centers = cv2.connectedComponentsWithStats(
        white_mask,
        8
    )

    candidates = []

    h, w = gray.shape

    for i in range(1, num):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 15 or area > 1000:
            continue

        # Arrow-like components are usually elongated
        ratio = max(ww, hh) / max(1, min(ww, hh))

        if ratio < 1.4:
            continue

        cx = int(centers[i][0])
        cy = int(centers[i][1])

        # Must be near road
        if road_mask[
            max(0, cy - 5):min(h, cy + 6),
            max(0, cx - 5):min(w, cx + 6)
        ].mean() < 100:
            continue

        candidates.append(
            (cx, cy, area, ratio)
        )

    if candidates:

        # Select elongated component nearest road center
        best = max(
            candidates,
            key=lambda p: p[3] * min(p[2], 200)
        )

        return best[0], best[1]

    # Fallback:
    # choose a road point near lower half of image
    ys, xs = np.where(road_mask > 0)

    if len(xs) == 0:
        return w // 2, h // 2

    target_y = int(h * 0.65)

    idx = np.argmin(
        np.abs(ys - target_y)
    )

    return int(xs[idx]), int(ys[idx])


# ============================================================
# CREATE SAFE MASK
# ============================================================

def create_safe_mask(
    road_mask,
    obstacle_mask,
    pothole_mask
):

    unsafe = cv2.bitwise_or(
        obstacle_mask,
        pothole_mask
    )

    # Inflate unsafe regions for safety
    kernel_obs = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (OBSTACLE_MARGIN * 2 + 1,
         OBSTACLE_MARGIN * 2 + 1)
    )

    unsafe = cv2.dilate(
        unsafe,
        kernel_obs
    )

    safe_mask = road_mask.copy()

    safe_mask[unsafe > 0] = 0

    # Small erosion makes path stay away from road boundary
    boundary_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    safe_mask = cv2.erode(
        safe_mask,
        boundary_kernel
    )

    return safe_mask


# ============================================================
# A* PATH PLANNING
# ============================================================

def heuristic(a, b):

    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def astar(
    safe_mask,
    start,
    goal
):

    h, w = safe_mask.shape

    sx, sy = start
    gx, gy = goal

    if not (
        0 <= sx < w and
        0 <= sy < h
    ):
        return []

    if not (
        0 <= gx < w and
        0 <= gy < h
    ):
        return []

    if safe_mask[sy, sx] == 0:
        return []

    if safe_mask[gy, gx] == 0:
        return []

    # 8-connected movement
    moves = [
        (-1, -1, math.sqrt(2)),
        (0, -1, 1),
        (1, -1, math.sqrt(2)),
        (-1, 0, 1),
        (1, 0, 1),
        (-1, 1, math.sqrt(2)),
        (0, 1, 1),
        (1, 1, math.sqrt(2))
    ]

    open_set = []

    heapq.heappush(
        open_set,
        (0, (sx, sy))
    )

    came_from = {}

    g_score = {
        (sx, sy): 0
    }

    visited = set()

    while open_set:

        _, current = heapq.heappop(
            open_set
        )

        if current in visited:
            continue

        visited.add(current)

        if current == (gx, gy):

            path = []

            node = current

            while node in came_from:

                path.append(node)
                node = came_from[node]

            path.append((sx, sy))

            path.reverse()

            return path

        cx, cy = current

        for dx, dy, cost in moves:

            nx = cx + dx
            ny = cy + dy

            if nx < 0 or nx >= w:
                continue

            if ny < 0 or ny >= h:
                continue

            if safe_mask[ny, nx] == 0:
                continue

            neighbour = (nx, ny)

            tentative_g = (
                g_score[current] + cost
            )

            if tentative_g < g_score.get(
                neighbour,
                float("inf")
            ):

                came_from[neighbour] = current

                g_score[neighbour] = tentative_g

                f = (
                    tentative_g +
                    heuristic(
                        neighbour,
                        (gx, gy)
                    )
                )

                heapq.heappush(
                    open_set,
                    (f, neighbour)
                )

    return []


# ============================================================
# FIND A LOOP
# ============================================================

def create_loop_path(
    safe_mask,
    start
):

    h, w = safe_mask.shape

    sx, sy = start

    # Find points around road boundary at different angles
    # around the start point.

    road_points = []

    ys, xs = np.where(
        safe_mask > 0
    )

    if len(xs) == 0:
        return []

    # Use farthest safe points as checkpoints
    distances = (
        (xs - sx) ** 2 +
        (ys - sy) ** 2
    )

    # Select several points distributed around the road
    order = np.argsort(distances)[::-1]

    selected = []

    for idx in order:

        point = (
            int(xs[idx]),
            int(ys[idx])
        )

        if all(
            heuristic(point, p) > 80
            for p in selected
        ):

            selected.append(point)

        if len(selected) >= 8:
            break

    if len(selected) < 2:
        return []

    # Sort checkpoints by angle around approximate road centre
    cx = np.mean(xs)
    cy = np.mean(ys)

    selected.sort(
        key=lambda p: math.atan2(
            p[1] - cy,
            p[0] - cx
        )
    )

    # Find closest checkpoint to start
    start_index = min(
        range(len(selected)),
        key=lambda i: heuristic(
            selected[i],
            start
        )
    )

    selected = (
        selected[start_index:] +
        selected[:start_index]
    )

    full_path = []

    current = start

    # Connect checkpoints using A*
    for checkpoint in selected:

        segment = astar(
            safe_mask,
            current,
            checkpoint
        )

        if not segment:
            continue

        if full_path:
            full_path.extend(
                segment[1:]
            )
        else:
            full_path.extend(
                segment
            )

        current = checkpoint

    # Finally return to start
    segment = astar(
        safe_mask,
        current,
        start
    )

    if segment:

        if full_path:
            full_path.extend(
                segment[1:]
            )
        else:
            full_path.extend(
                segment
            )

    return full_path


# ============================================================
# SMOOTH PATH
# ============================================================

def smooth_path(path):

    if len(path) < 5:
        return path

    result = []

    step = 5

    for i in range(
        0,
        len(path),
        step
    ):

        result.append(
            path[i]
        )

    if result[-1] != path[-1]:
        result.append(path[-1])

    return result


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    output_path
):

    print(
        "\nProcessing:",
        os.path.basename(image_path)
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "ERROR: Image could not be read."
        )

        return False

    original = image.copy()

    # --------------------------------------------------------
    # ROAD
    # --------------------------------------------------------

    road_mask = detect_road(
        image
    )

    road_area = cv2.countNonZero(
        road_mask
    )

    if road_area < 5000:

        print(
            "Road detection failed."
        )

        # Fallback using grayscale threshold
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        _, road_mask = cv2.threshold(
            gray,
            int(np.median(gray)),
            255,
            cv2.THRESH_BINARY
        )

    # --------------------------------------------------------
    # OBSTACLES
    # --------------------------------------------------------

    obstacle_mask, obstacle_boxes = detect_obstacles(
        image,
        road_mask
    )

    # --------------------------------------------------------
    # POTHOLES
    # --------------------------------------------------------

    pothole_mask, pothole_boxes = detect_potholes(
        image,
        road_mask
    )

    print(
        "Obstacles detected:",
        len(obstacle_boxes)
    )

    print(
        "Potholes detected:",
        len(pothole_boxes)
    )

    # --------------------------------------------------------
    # SAFE AREA
    # --------------------------------------------------------

    safe_mask = create_safe_mask(
        road_mask,
        obstacle_mask,
        pothole_mask
    )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    start = find_start(
        image,
        road_mask
    )

    # Move start to nearest safe pixel
    sx, sy = start

    if (
        sx < 0 or
        sy < 0 or
        sx >= image.shape[1] or
        sy >= image.shape[0] or
        safe_mask[sy, sx] == 0
    ):

        safe_points = np.column_stack(
            np.where(safe_mask > 0)
        )

        if len(safe_points) > 0:

            distances = (
                (safe_points[:, 1] - sx) ** 2 +
                (safe_points[:, 0] - sy) ** 2
            )

            idx = np.argmin(
                distances
            )

            sy = int(
                safe_points[idx][0]
            )

            sx = int(
                safe_points[idx][1]
            )

    start = (sx, sy)

    print(
        "Start point:",
        start
    )

    # --------------------------------------------------------
    # CREATE LOOP
    # --------------------------------------------------------

    path = create_loop_path(
        safe_mask,
        start
    )

    if len(path) < 20:

        print(
            "Loop path failed. Trying direct A* fallback."
        )

        # Find a far point on road
        ys, xs = np.where(
            safe_mask > 0
        )

        if len(xs) > 0:

            distances = (
                (xs - sx) ** 2 +
                (ys - sy) ** 2
            )

            idx = np.argmax(
                distances
            )

            goal = (
                int(xs[idx]),
                int(ys[idx])
            )

            path = astar(
                safe_mask,
                start,
                goal
            )

    # --------------------------------------------------------
    # DRAW RESULT
    # --------------------------------------------------------

    result = original.copy()

    # Road outline
    road_contours, _ = cv2.findContours(
        road_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        result,
        road_contours,
        -1,
        (255, 200, 0),
        2
    )

    # Draw obstacles in red
    for x, y, w, h in obstacle_boxes:

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        cv2.putText(
            result,
            "Obstacle",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1
        )

    # Draw potholes in purple
    for x, y, w, h in pothole_boxes:

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 255),
            2
        )

        cv2.putText(
            result,
            "Pothole",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1
        )

    # Draw path
    if len(path) >= 2:

        path_array = np.array(
            path,
            dtype=np.int32
        ).reshape((-1, 1, 2))

        cv2.polylines(
            result,
            [path_array],
            False,
            (0, 255, 0),
            4,
            cv2.LINE_AA
        )

        # Draw direction arrows
        for i in range(
            0,
            len(path) - 10,
            30
        ):

            p1 = path[i]
            p2 = path[
                min(i + 8, len(path) - 1)
            ]

            cv2.arrowedLine(
                result,
                p1,
                p2,
                (0, 255, 0),
                2,
                tipLength=0.3
            )

    # Start point
    cv2.circle(
        result,
        start,
        8,
        (255, 255, 255),
        -1
    )

    cv2.circle(
        result,
        start,
        10,
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        "START",
        (
            start[0] + 12,
            start[1]
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Information box
    cv2.rectangle(
        result,
        (10, 10),
        (430, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        result,
        f"Obstacles: {len(obstacle_boxes)}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        result,
        f"Potholes: {len(pothole_boxes)}",
        (220, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        result,
        f"Path points: {len(path)}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    # Save
    cv2.imwrite(
        output_path,
        result
    )

    print(
        "Saved:",
        output_path
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    input_dir = os.path.join(
        script_dir,
        "input"
    )

    output_dir = os.path.join(
        script_dir,
        "output"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    success = 0

    print(
        "\n======================================"
    )

    print(
        "TASK 4 - AERIAL PATH PLANNING"
    )

    print(
        "======================================"
    )

    for i in range(1, 10):

        input_file = os.path.join(
            input_dir,
            f"file{i}.png"
        )

        output_file = os.path.join(
            output_dir,
            f"file{i}_path.png"
        )

        if not os.path.exists(
            input_file
        ):

            print(
                f"Skipped file{i}.png - not found"
            )

            continue

        try:

            if process_image(
                input_file,
                output_file
            ):

                success += 1

        except Exception as e:

            print(
                f"ERROR in file{i}.png:"
            )

            print(e)

    print(
        "\n======================================"
    )

    print(
        f"Completed: {success}/9 images"
    )

    print(
        "Output folder:",
        output_dir
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()