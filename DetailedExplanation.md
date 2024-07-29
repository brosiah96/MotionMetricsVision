# MotionMetricsVision Detailed Explanation

This document provides a detailed explanation of the classes, methods, variables, and functions in the `MotionMetricsVision` project, specifically focusing on the following files:

- `config.py`
- `object_tracker.py`
- `video_tracker.py`
- `main.py`

## config.py

### Purpose

The `config.py` file is used to store configuration settings and parameters that are used across the project. This helps in keeping the code modular and maintainable by centralizing configuration details.

### Content

```python
# Configuration parameters
VIDEO_PATH = "path/to/video/file"  # Path to the video file to be processed
OUTPUT_PATH = "path/to/output/file"  # Path where the output video will be saved
TRACKER_TYPE = "CSRT"  # Type of object tracker to be used (e.g., CSRT, KCF, MOSSE, etc.)
```

### Variables

- `VIDEO_PATH`: A string representing the path to the video file that needs to be processed.
- `OUTPUT_PATH`: A string representing the path where the output video will be saved.
- `TRACKER_TYPE`: A string representing the type of object tracker to be used. Common options include 'CSRT', 'KCF', 'MOSSE', etc.

## object_tracker.py

### Purpose

The `object_tracker.py` file contains the implementation of the `ObjectTracker` class, which is responsible for tracking objects within a video frame using a specified tracking algorithm.

### Class: ObjectTracker

#### Initialization

```python
class ObjectTracker:
    def __init__(self, tracker_type):
        self.tracker_type = tracker_type
        self.tracker = self._initialize_tracker(tracker_type)
```

- **tracker_type**: The type of tracker to be used (e.g., 'CSRT', 'KCF').
- **tracker**: An instance of the specified tracker type.

#### Methods

- **_initialize_tracker(tracker_type)**: Initializes and returns the tracker object based on the given tracker type.

```python
def _initialize_tracker(self, tracker_type):
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
    elif tracker_type == "KCF":
        return cv2.TrackerKCF_create()
    # Add other trackers as needed
```

- **init(frame, bbox)**: Initializes the tracker with the first frame and the bounding box of the object to be tracked.

```python
def init(self, frame, bbox):
    self.tracker.init(frame, bbox)
```

- **update(frame)**: Updates the tracker with the new frame and returns the updated bounding box and a boolean indicating if the update was successful.

```python
def update(self, frame):
    success, bbox = self.tracker.update(frame)
    return success, bbox
```

## video_tracker.py

### Purpose

The `video_tracker.py` file contains the `VideoTracker` class, which is responsible for managing the video capture and processing each frame to track objects.

### Class: VideoTracker

#### Initialization

```python
class VideoTracker:
    def __init__(self, video_path, tracker_type):
        self.video_path = video_path
        self.tracker_type = tracker_type
        self.cap = cv2.VideoCapture(video_path)
        self.object_tracker = ObjectTracker(tracker_type)
        self.init_bbox = None
        self.initialized = False
```

- **video_path**: Path to the video file.
- **tracker_type**: Type of object tracker to be used.
- **cap**: VideoCapture object for reading the video file.
- **object_tracker**: Instance of the `ObjectTracker` class.
- **init_bbox**: Initial bounding box for the object to be tracked.
- **initialized**: Boolean indicating if the tracker has been initialized with the first frame.

#### Methods

- **select_roi()**: Allows the user to select the region of interest (ROI) for tracking in the first frame.

```python
def select_roi(self):
    ret, frame = self.cap.read()
    if not ret:
        raise ValueError("Cannot read video file")
    self.init_bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Frame")
```

- **process()**: Processes the video frame by frame, initializing the tracker with the first frame and updating it for subsequent frames.

```python
def process(self):
    while True:
        ret, frame = self.cap.read()
        if not ret:
            break

        if not self.initialized:
            self.object_tracker.init(frame, self.init_bbox)
            self.initialized = True

        success, bbox = self.object_tracker.update(frame)
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    self.cap.release()
    cv2.destroyAllWindows()
```

## main.py

### Purpose

The `main.py` file serves as the entry point for the application. It reads the configuration, initializes the `VideoTracker` class, and starts the video processing.

### Main Function

```python
def main():
    video_tracker = VideoTracker(Config.VIDEO_PATH, Config.TRACKER_TYPE)
    video_tracker.select_roi()
    video_tracker.process()
```

### Execution

- **main()**: The main function that initializes the `VideoTracker` with the video path and tracker type from the configuration file. It then allows the user to select the region of interest (ROI) and starts processing the video.

### Entry Point

```python
if __name__ == "__main__":
    main()
```

- **if __name__ == "__main__"**: This ensures that the `main()` function is called only when the script is run directly, not when imported as a module.



