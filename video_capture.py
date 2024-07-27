# video_capture.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import queue
import threading
from collections import deque
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Any

# Importing constants from config
import config

# Import the ObjectTracker class
from object_tracker import ObjectTracker

# Setup logging
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class VideoCapture:
    """
    Captures video, processes frames, and handles object tracking and plotting.

    Attributes:
        cap (cv2.VideoCapture): Video capture object.
        tracker (ObjectTracker): Object tracker instance.
        fig (plt.Figure): Matplotlib figure for plotting.
        ax (List[plt.Axes]): List of axes for subplots.
        data_queue (queue.Queue): Queue for storing tracking data.
        is_running (bool): Flag to control the running state of the capture thread.
        lines (Dict[int, List[Any]]): Dictionary to store plot lines for each object.
    """

    def __init__(self, source: int = 0, tracker: ObjectTracker = None):
        """
        Initializes the VideoCapture object with the video source and tracker.

        Args:
            source (int): Video source (default is 0, which typically corresponds to the webcam).
            tracker (ObjectTracker): ObjectTracker instance for tracking objects.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Failed to open the video capture device")
        self.calibrate()
        self.tracker = tracker if tracker is not None else ObjectTracker()
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(3, 1, figsize=(10, 15))
        self.setup_plots()
        self.data_queue = queue.Queue()
        self.is_running = True
        self.lines = {}

    def calibrate(self) -> None:
        """
        Calibrates the video capture to determine pixels per inch for the video feed.
        """
        ret, frame = self.cap.read()
        if ret:
            height, width = frame.shape[:2]
            self.pixels_per_inch_x = width / config.FRAME_WIDTH_INCHES
            self.pixels_per_inch_y = height / config.FRAME_HEIGHT_INCHES

            # Update the global conversion factors
            global PIXELS_TO_INCHES, PIXELS_TO_FEET
            PIXELS_TO_INCHES = (self.pixels_per_inch_x + self.pixels_per_inch_y) / 2
            PIXELS_TO_FEET = PIXELS_TO_INCHES / 12
            logging.info("Calibration complete")
        else:
            raise ValueError("Could not read frame for calibration")

    def setup_plots(self) -> None:
        """
        Sets up the initial plots for object trajectories, speeds, and accelerations.
        """
        titles = ["Object Trajectories", "Object Speeds", "Object Accelerations"]
        ylabels = ["Distance (inches)", "Speed (inches/s)", "Acceleration (inches/s^2)"]
        for i, (title, ylabel) in enumerate(zip(titles, ylabels)):
            self.ax[i].set_title(title)
            self.ax[i].set_xlabel("Distance (inches)" if i == 0 else "Time (s)")
            self.ax[i].set_ylabel(ylabel)
            self.ax[i].grid(True)
        self.fig.tight_layout()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a video frame to detect objects and update trackers.

        Args:
            frame (np.ndarray): The current video frame.

        Returns:
            np.ndarray: The processed frame with tracking information overlaid.
        """
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array(config.LOWER_WHITE)
        upper_white = np.array(config.UPPER_WHITE)
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Apply morphological operations to reduce noise
        kernel = np.ones(config.KERNEL_SIZE, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)

        # Apply Gaussian blur to smooth the mask
        mask = cv2.GaussianBlur(mask, config.GAUSSIAN_BLUR_KERNEL_SIZE, 0)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        contours = [c for c in contours if config.CONTOUR_MIN_AREA < cv2.contourArea(c) < config.CONTOUR_MAX_AREA]

        # Initialize or update trackers
        if not self.tracker.trackers:
            self.tracker.initialize_trackers(frame, contours)
        else:
            frame = self.tracker.update_trackers(frame)
            self.tracker.calculate_motion_parameters()
            frame = self.tracker.display_data(frame)

        return frame

    def capture_thread(self) -> None:
        """
        Thread function to capture and process video frames continuously.
        """
        frame_count = 0
        while self.is_running:
            logging.debug(f"Capturing frame {frame_count}")
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to capture frame from camera.")
                break

            if frame.size == 0:
                logging.warning("Empty frame received. Skipping processing.")
                continue

            logging.debug(f"Processing frame {frame_count}")
            frame = self.process_frame(frame)
            self.data_queue.put(self.tracker.object_positions.copy())

            logging.debug(f"Displaying frame {frame_count}")
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break

            frame_count += 1

        logging.info(f"Capture thread ended after processing {frame_count} frames")
        self.cap.release()
        cv2.destroyAllWindows()

    def update_plot(self, frame: int) -> List[Any]:
        """
        Updates the plots with the latest tracking data.

        Args:
            frame (int): The current frame number.

        Returns:
            List[Any]: List of updated plot lines.
        """
        try:
            data = self.data_queue.get_nowait()
        except queue.Empty:
            return []

        for i in range(3):
            self.ax[i].clear()
            self.setup_plots()  # Reset titles and labels

        updated_lines = []
        for object_id, positions in data.items():
            if len(positions) > 1:
                positions_np = np.array(positions)
                times = positions_np[:, 2] - positions_np[0, 2]  # Relative time

                if object_id not in self.lines:
                    self.lines[object_id] = [None, None, None]

                # Plot trajectories
                self.lines[object_id][0], = self.ax[0].plot(positions_np[:, 0], positions_np[:, 1],
                                                            label=f'Object {object_id}')
                updated_lines.append(self.lines[object_id][0])

                # Plot speeds (only for non-static objects)
                if object_id in self.tracker.object_speeds and len(
                        self.tracker.object_speeds[object_id]) > 0 and not np.allclose(positions_np[:-1, :2],
                                                                                       positions_np[1:, :2]):
                    speeds = np.linalg.norm(self.tracker.object_speeds[object_id], axis=1)
                    if len(times[:-1]) == len(speeds):
                        self.lines[object_id][1], = self.ax[1].plot(times[:-1], speeds, label=f'Object {object_id}')
                        updated_lines.append(self.lines[object_id][1])

                # Plot accelerations (only for non-static objects)
                if object_id in self.tracker.object_accelerations and len(
                        self.tracker.object_accelerations[object_id]) > 0 and not np.allclose(positions_np[:-1, :2],
                                                                                              positions_np[1:, :2]):
                    accelerations = np.linalg.norm(self.tracker.object_accelerations[object_id], axis=1)
                    if len(times[:-2]) == len(accelerations):
                        self.lines[object_id][2], = self.ax[2].plot(times[:-2], accelerations,
                                                                    label=f'Object {object_id}')
                        updated_lines.append(self.lines[object_id][2])

        for i in range(3):
            if len(self.ax[i].lines) > 0:
                self.ax[i].legend()
            self.ax[i].relim()
            self.ax[i].autoscale_view()

        self.fig.canvas.draw()
        return updated_lines

    def run(self) -> None:
        """
        Starts the video capture and processing.
        """
        capture_thread = threading.Thread(target=self.capture_thread)
        capture_thread.start()

        # Make sure the anim variable persists until plt.show() is called
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=config.PLOT_UPDATE_INTERVAL, blit=True,
                                  cache_frame_data=False)
        plt.show(block=True)  # Use block=True to keep the main thread alive

        self.is_running = False
        capture_thread.join()
        plt.ioff()  # Turn off interactive mode
        plt.close(self.fig)  # Close the figure when done
