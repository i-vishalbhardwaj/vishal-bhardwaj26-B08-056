import cv2
import numpy as np

# -----------------------------
# Read input image
# -----------------------------
image = cv2.imread("task3/input/road.png")

if image is None:
    print("Image not found!")
    exit()

result = image.copy()

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# =========================================================
# 1. YELLOW OBSTACLE DETECTION
# =========================================================

lower_yellow = np.array([20, 100, 80])
upper_yellow = np.array([40, 255, 255])

yellow_mask = cv2.inRange(
    hsv,
    lower_yellow,
    upper_yellow
)

# Remove small noise
kernel = np.ones((5, 5), np.uint8)
yellow_mask = cv2.morphologyEx(
    yellow_mask,
    cv2.MORPH_OPEN,
    kernel
)


# Distance transform helps separate touching objects
dist = cv2.distanceTransform(
    yellow_mask,
    cv2.DIST_L2,
    5
)

_, sure_fg = cv2.threshold(
    dist,
    0.35 * dist.max(),
    255,
    0
)

sure_fg = np.uint8(sure_fg)

sure_bg = cv2.dilate(
    yellow_mask,
    kernel,
    iterations=3
)

unknown = cv2.subtract(
    sure_bg,
    sure_fg
)

num_labels, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(
    image,
    markers
)

yellow_count = 0

for label in range(2, num_labels + 1):

    mask = np.zeros(
        yellow_mask.shape,
        dtype=np.uint8
    )

    mask[markers == label] = 255

    area = cv2.countNonZero(mask)

    if area < 500:
        continue

    x, y, w, h = cv2.boundingRect(mask)

    # Draw yellow bounding box
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 255, 255),
        3
    )

    # Coordinate
    cv2.putText(
        result,
        f"Obstacle ({x},{y})",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    yellow_count += 1


# =========================================================
# 2. WHITE POTHOLE DETECTION
# =========================================================

lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 80, 255])

white_mask = cv2.inRange(
    hsv,
    lower_white,
    upper_white
)

# Remove thin road markings
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (21, 21)
)

white_clean = cv2.morphologyEx(
    white_mask,
    cv2.MORPH_OPEN,
    kernel
)

# Find white blobs
contours, _ = cv2.findContours(
    white_clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

white_count = 0

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 300 or area > 5000:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    # Potholes should be small
    if w > 100 or h > 100:
        continue

    # Avoid long road lines
    aspect_ratio = w / float(h)

    if aspect_ratio < 0.45 or aspect_ratio > 2.2:
        continue

    # Check circularity
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        continue

    circularity = (
        4 * np.pi * area
        / (perimeter * perimeter)
    )

    if circularity < 0.25:
        continue

    # Draw pothole box
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (255, 255, 255),
        3
    )

    # Coordinates
    cv2.putText(
        result,
        f"Pothole ({x},{y})",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    white_count += 1
# =========================================================
# 3. DISPLAY COUNTS
# =========================================================

total_count = yellow_count + white_count

cv2.putText(
    result,
    f"Total Objects: {total_count}",
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 0),
    3
)

cv2.putText(
    result,
    f"Obstacles: {yellow_count}",
    (30, 75),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 255),
    2
)

cv2.putText(
    result,
    f"Potholes: {white_count}",
    (30, 105),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)


# =========================================================
# 4. SAVE OUTPUT
# =========================================================

cv2.imwrite(
    "task3/output/road_detected.png",
    result
)

print("Obstacle and pothole detection completed!")
print("Obstacles:", yellow_count)
print("Potholes:", white_count)
print("Total:", total_count)
print("Output saved as task3/output/road_detected.png")