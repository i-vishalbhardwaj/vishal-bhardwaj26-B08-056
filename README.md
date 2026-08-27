## Daily Progress

### Task 1 — GitHub Repository and Command Line

**Progress:**
- Created and configured the GitHub repository.
- Learned basic Git and GitHub commands.
- Used VS Code terminal for Git operations.
- Added, committed and pushed files to GitHub.

**Problems Faced:**
- Initially had difficulty understanding Git commands.
- Faced issues while adding and committing files.
- Learned how to check Git status and resolve the issues.

**What I Learned:**
- `git status`
- `git add`
- `git commit`
- `git push`
- How to upload and manage project files on GitHub.

### Task 2 — Lane Detection

**Progress:**
- Set up Python and OpenCV.
- Loaded the road image using OpenCV.
- Detected edges using Canny Edge Detection.
- Used Hough Line Transform to detect lane lines.
- Created the final lane detection output image.

**Problems Faced:**
- Initially OpenCV was not installed correctly.
- Faced an error while processing Hough lines.
- Faced input image path issues.
- Fixed the errors and successfully generated the output.

**What I Learned:**
- Basic Python and OpenCV workflow.
- Image loading and processing.
- Edge detection.
- Hough Line Transform.
- Drawing lane lines and highlighting the drivable area.

### TASK 3 – OBSTACLE & POTHOLE DETECTION

**Progress:**
- Implemented obstacle and pothole detection using Python and OpenCV.
- Detected yellow obstacles and white potholes using image processing.
- Marked detected objects with rectangular bounding boxes.
- Added pixel coordinates of detected objects.
- Added the total number of obstacles and potholes.
- Generated the final output image with detected objects.

**Problems Faced:**
- Initially, the road boundary/corner was incorrectly detected as an object.
- Some potholes were not detected correctly.
- Some false detections appeared during the initial implementation.
- Bounding boxes were not accurate in the beginning.
- Adjusted detection conditions and filtering to remove false detections.
- After tuning the parameters, the required objects were detected correctly.

**What I Learned:**
- Learned the basics of image processing using OpenCV.
- Learned how to use grayscale and color-based detection.
- Learned how thresholding and contours can be used for object detection.
- Learned how to detect and classify objects based on their color and shape.
- Learned how to draw rectangular bounding boxes using pixel coordinates.
- Learned how image pixel coordinates (x, y) work.
- Learned how to count detected objects.
- Learned how to reduce false detections by adjusting image-processing parameters.