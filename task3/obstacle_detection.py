import cv2
import numpy as np
import os

def process_image(img_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]

    pothole_count = 0
    obstacle_count = 0

    # -------------------------------------------------------------
    # 1. OPTIMIZED POTHOLE DETECTION (Covers page1 and page10 edge potholes)
    # -------------------------------------------------------------
    # Slower threshold to catch dim/distant white patches
    _, white_mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    
    # Mild kernel to retain original circular shape
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_mask_cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small)

    contours_white, _ = cv2.findContours(white_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_white:
        area = cv2.contourArea(cnt)
        if area > 25:  # Lowered min area to detect far-away small potholes
            x, y, w, h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            aspect_ratio = float(w) / h

            # Filter out very long/thin continuous lane markings (aspect ratio extreme check)
            if 0.18 < circularity and 0.25 < aspect_ratio < 4.0 and w < (width * 0.45) and h < (height * 0.45):
                # Avoid border-touching full frame contours
                if x > 1 and y > 1 and (x + w) < (width - 1) and (y + h) < (height - 1):
                    pothole_count += 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, f"Pothole ({x},{y})", (x, max(15, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # -------------------------------------------------------------
    # 2. OBSTACLE DETECTION (Yellow, Blue, Green, Crates)
    # -------------------------------------------------------------
    masks = []

    # Yellow / Gold / Brownish-Yellow (Cylinders & Wooden Crates)
    masks.append(cv2.inRange(hsv, np.array([10, 40, 40]), np.array([38, 255, 255])))

    # Dark Blue / Light Blue Cylinders
    masks.append(cv2.inRange(hsv, np.array([90, 60, 30]), np.array([135, 255, 255])))

    # Green Cylinders
    masks.append(cv2.inRange(hsv, np.array([38, 40, 30]), np.array([85, 255, 255])))

    combined_obstacle_mask = masks[0]
    for m in masks[1:]:
        combined_obstacle_mask = cv2.bitwise_or(combined_obstacle_mask, m)

    kernel_obs = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_obstacle_mask = cv2.morphologyEx(combined_obstacle_mask, cv2.MORPH_CLOSE, kernel_obs)

    contours_obs, _ = cv2.findContours(combined_obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_obs:
        area = cv2.contourArea(cnt)
        if area > 80:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            # Splitting overlapping side-by-side cylinders
            if aspect_ratio > 1.75 and area > 1200:
                half_w = w // 2
                
                # Cylinder 1
                obstacle_count += 1
                cv2.rectangle(img, (x, y), (x + half_w, y + h), (0, 0, 255), 2)
                cv2.putText(img, f"Obstacle ({x},{y})", (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                
                # Cylinder 2
                obstacle_count += 1
                cv2.rectangle(img, (x + half_w, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, f"Obstacle ({x+half_w},{y})", (x + half_w, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            else:
                obstacle_count += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, f"Obstacle ({x},{y})", (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # -------------------------------------------------------------
    # 3. DRAW SUMMARY ON TOP CORNER
    # -------------------------------------------------------------
    summary = f"Total Potholes: {pothole_count} | Total Obstacles: {obstacle_count}"
    cv2.rectangle(img, (10, 10), (520, 50), (0, 0, 0), -1)
    cv2.putText(img, summary, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(output_path, img)
    print(f"Done: {os.path.basename(img_path)} -> Potholes: {pothole_count}, Obstacles: {obstacle_count}")

# Dynamic Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "input")
output_dir = os.path.join(script_dir, "output")

os.makedirs(output_dir, exist_ok=True)

for i in range(1, 11):
    file_name = f"page{i}.png"
    in_file_path = os.path.join(input_dir, file_name)
    out_file_path = os.path.join(output_dir, f"page{i}_detected.png")

    if os.path.exists(in_file_path):
        process_image(in_file_path, out_file_path)
    else:
        print(f"Skipped: {file_name} not found in {input_dir}")