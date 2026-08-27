import cv2
import numpy as np
import os

# Input and output folders
input_folder = "input"
output_folder = "output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process all 10 road images
for i in range(1, 11):

    extension = ".png" if i <= 8 else ".jpeg"
    input_path = os.path.join(input_folder, f"road{i}{extension}")
    output_path = os.path.join(output_folder, f"road{i}_detected.png")

    print(f"\nProcessing road{i}.png...")

    # Input image
    image = cv2.imread(input_path)

    if image is None:
        print(f"Image not found: {input_path}")
        continue

    result = image.copy()

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Image dimensions
    height, width = image.shape[:2]

    # Region of Interest
    mask = np.zeros_like(edges)

    roi = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.60), int(height * 0.55)),
        (int(width * 0.40), int(height * 0.55))
    ]])

    cv2.fillPoly(mask, roi, 255)
    roi_edges = cv2.bitwise_and(edges, mask)

    # Hough Line Detection
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

    # Calculate average lane line
    def average_line(lines, height):

        if not lines:
            return None

        slopes = []
        intercepts = []

        for x1, y1, x2, y2 in lines:

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            slopes.append(slope)
            intercepts.append(intercept)

        if not slopes:
            return None

        slope = np.mean(slopes)
        intercept = np.mean(intercepts)

        y1 = int(height * 0.95)
        y2 = int(height * 0.60)

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return x1, y1, x2, y2

    # Get left and right lanes
    left_lane = average_line(left_lines, height)
    right_lane = average_line(right_lines, height)

    # Draw lanes and drivable area
    if left_lane is not None and right_lane is not None:

        lx1, ly1, lx2, ly2 = left_lane
        rx1, ry1, rx2, ry2 = right_lane

        # Lane boundaries
        cv2.line(
            result,
            (lx1, ly1),
            (lx2, ly2),
            (255, 255, 255),
            5
        )

        cv2.line(
            result,
            (rx1, ry1),
            (rx2, ry2),
            (255, 255, 255),
            5
        )

        # Drivable area
        polygon = np.array([
            [
                (lx1, ly1),
                (rx1, ry1),
                (rx2, ry2),
                (lx2, ly2)
            ]
        ], dtype=np.int32)

        overlay = result.copy()

        cv2.fillPoly(
            overlay,
            polygon,
            (0, 255, 0)
        )

        result = cv2.addWeighted(
            overlay,
            0.20,
            result,
            0.80,
            0
        )

    # Save output
    cv2.imwrite(output_path, result)

    print(f"Output saved: {output_path}")

print("\nAll 10 images processed!")