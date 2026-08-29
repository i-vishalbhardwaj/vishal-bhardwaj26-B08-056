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

### Task 2: Lane Detection

Yesterday, Task 2 was initially implemented and tested on a single road image. The lane detection algorithm was able to detect the lane boundaries and highlight the drivable area.

Today, the implementation was modified to work with multiple input images instead of only one image.

### Changes Made
- Modified the code to process all 10 road images automatically.
- Added a loop to process each image one by one using the same lane detection algorithm.
- Handled different image formats:
  - `road1.png` to `road8.png`
  - `road9.jpeg` and `road10.jpeg`
- Generated a separate output for each input image.
- Stored the processed images in the `output` folder.
- Tested the same algorithm across all 10 images to evaluate its performance.
- Observed that the algorithm works well on some images but does not detect lanes accurately in every image.
- The current implementation uses Canny Edge Detection, Region of Interest (ROI), and Hough Line Transform.
- The updated Task 2 code and outputs were committed and pushed to GitHub.

### Current Status
Task 2 has been successfully modified from single-image processing to multiple-image processing. The same algorithm is now tested on all 10 provided road images. Further improvement of lane detection accuracy may be required for images where the current approach does not detect the lanes correctly.  

### Task -2 summery 
Maine OpenCV-based lane detection approach use ki hai. Pehle image ko grayscale mein convert karke Canny Edge Detection se edges nikale. Phir Region of Interest (ROI) apply karke sirf road ke relevant area ko rakha. Uske baad Probabilistic Hough Transform se line segments detect kiye. Slope ke basis par left aur right lane candidates ko separate kiya, phir unke average slope aur intercept se representative lane boundaries banayi. Finally, dono lane boundaries ke beech polygon bana kar transparent green overlay se drivable area highlight kiya aur processed images ko output folder mein save kiya.

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

### Task 3 – Obstacle and Pothole Detection

* Implemented obstacle and pothole detection using OpenCV.
* Initially tested the algorithm on a single image.
* Modified the program to process multiple input images automatically.
* The algorithm now processes all 10 images (`page1.png` to `page10.png`).
* Used HSV color segmentation, morphological operations, contour detection, and shape-based filtering.
* Detected obstacles and potholes are marked with bounding boxes and coordinates.
* Detection results and object counts are displayed on each output image.
* Separate output images are generated and saved in the `task3/output` folder.
* Tested the algorithm on the complete set of input images.

### task3 summery

Maine OpenCV-based pothole and obstacle detection approach use ki hai. Pehle image ko grayscale aur HSV mein convert kiya. Grayscale image par thresholding aur morphological operations apply karke pothole candidates identify kiye, phir contours detect karke area, circularity aur aspect ratio ke basis par potholes filter kiye. Obstacles ke liye HSV color segmentation use karke yellow, blue, red aur green regions ke masks banaye. In masks ko combine karke morphological operations aur contour filtering ke through obstacles detect kiye. Finally, potholes ko green bounding boxes aur obstacles ko red bounding boxes se mark kiya, coordinates aur total counts display kiye, aur processed images ko output folder mein save kiya.

### TASK 4 – AERIAL PATH PLANNING

#### Progress:
- Analyzed the given aerial images of the track.
- Processed all 9 input images.
- Detected the road/track boundary from the images.
- Detected obstacles and potholes that could make the path unsafe.
- Created a safe path inside the road boundary while avoiding detected obstacles and potholes.
- Used a checkpoint-based path planning approach to generate the route.
- Generated a separate output image for each input image with the calculated safe path.
- Saved all 9 processed output images in the `task4/output` folder.
- The final path is highlighted on each output image for easy visualization.

#### Problems Faced:
- Initially, the generated path was going outside the road boundary.
- Some detected obstacles and potholes were interfering with the path.
- The path sometimes created unnecessary branches instead of following one safe route.
- Path generation parameters had to be adjusted to make the route safer and smoother.
- After tuning the detection and path-planning conditions, the program successfully generated paths for all 9 images.

#### What I Learned:
- Learned how aerial images can be used for path planning.
- Learned how to identify road boundaries from an image.
- Learned how obstacle and pothole detection can be combined with path planning.
- Learned about safe path generation using checkpoints.
- Learned how to check whether a path stays inside the road.
- Learned how to avoid obstacles and potholes while generating a route.
- Learned how to process multiple images automatically using Python.
- Learned how to save the calculated path as output images.

### task4 summery 
Maine OpenCV-based aerial image path planning approach use ki hai. Pehle image se grayscale brightness aur HSV saturation ke basis par road area detect karke road mask create kiya. Uske baad HSV color segmentation ka use karke yellow, blue, red aur green regions se obstacles detect kiye, aur grayscale thresholding se dark regions ko pothole candidates ke roop mein identify kiya. Morphological operations aur contour filtering ke through noise remove karke obstacle aur pothole regions ko refine kiya. Phir obstacles aur potholes ko unsafe regions mark karke dilation ke through safety margin add kiya aur road boundary ko erode karke safe traversable area create kiya. Start point detect karne ke baad A* path-finding algorithm with Euclidean distance heuristic ka use karke safe area ke andar path generate kiya. Multiple checkpoints ko connect karke loop-like path banaya aur final output image mein road boundary, obstacles, potholes, start point aur planned path ko highlight karke output folder mein save kiya.

### Task 5 – Introduction to ROS 2 (Bonus)

Maine ROS 2 Jazzy ko Ubuntu 24.04 (WSL) environment mein install aur setup kiya. ROS 2 workspace create karke task5 naam ka Python package banaya. Publisher aur subscriber nodes ke liye basic Python files create ki aur setup.py mein required entry points configure kiye. ROS 2 package ko colcon build ke through successfully build bhi kiya.

However, while trying to run the publisher-subscriber communication, I faced a ROS 2 package discovery/environment setup issue. Although the package was successfully detected by colcon and built successfully, ROS 2 was unable to recognize the task5 package through the ROS package environment. Due to this issue, I could not complete and upload the final publisher-subscriber and service-client implementation.
