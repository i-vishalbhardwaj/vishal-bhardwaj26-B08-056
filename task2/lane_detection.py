import cv2
import numpy as np

# Input image
image = cv2.imread("input/road1.png")

if image is None:
    print("Image not found!")
    exit()

# Make a copy for drawing
result = image.copy()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges
edges = cv2.Canny(gray, 50, 150)

# Get image dimensions
height, width = image.shape[:2]

# Region of Interest (road area)
mask = np.zeros_like(edges)

roi = np.array([[
    (0, height),
    (width, height),
    (int(width * 0.60), int(height * 0.55)),
    (int(width * 0.40), int(height * 0.55))
]])

cv2.fillPoly(mask, roi, 255)

# Keep only road-area edges
roi_edges = cv2.bitwise_and(edges, mask)

# Detect lane lines
lines = cv2.HoughLinesP(
    roi_edges,
    1,
    np.pi / 180,
    threshold=50,
    minLineLength=50,
    maxLineGap=100
)

left_lines = []
right_lines = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        if slope < -0.5:
            left_lines.append((x1, y1, x2, y2))

        elif slope > 0.5:
            right_lines.append((x1, y1, x2, y2))


def average_line(lines, height):
    if not lines:
        return None

    slopes = []
    intercepts = []

    for x1, y1, x2, y2 in lines:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        slopes.append(slope)
        intercepts.append(intercept)

    slope = np.mean(slopes)
    intercept = np.mean(intercepts)

        # Line points
    y1 = height
    y2 = int(height * 0.60)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return x1, y1, x2, y2
# Get averaged left and right lane lines
left_lane = average_line(left_lines, height)
right_lane = average_line(right_lines, height)

# Draw lanes and drivable area
if left_lane is not None and right_lane is not None:
    lx1, ly1, lx2, ly2 = left_lane
    rx1, ry1, rx2, ry2 = right_lane

    # Draw two separate lane boundaries
    cv2.line(result, (lx1, ly1), (lx2, ly2), (0, 255, 0), 6)
    cv2.line(result, (rx1, ry1), (rx2, ry2), (0, 255, 0), 6)

    # Drivable area - only translucent green
    polygon = np.array([
        [(lx1, ly1), (rx1, ry1),
         (rx2, ry2), (lx2, ly2)]
    ], dtype=np.int32)

    overlay = result.copy()
    cv2.fillPoly(overlay, polygon, (0, 255, 0))

    result = cv2.addWeighted(overlay, 0.20, result, 0.80, 0)

# Save result
cv2.imwrite("output/road1_detected.png", result)

print("Lane detection completed!")
print("Output saved as output/road1_detected.png")