# object_tracker.py
import cv2
import numpy as np
from collections import deque
import logging
from typing import List, Tuple, Dict, Deque

# Importing constants from config
import config

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ObjectTracker:
    """
    Tracks objects using OpenCV trackers and calculates their motion parameters.

    Attributes:
        max_frames_to_track (int): Maximum number of frames to track.
        min_area (int): Minimum area of the object to track.
        max_failures (int): Maximum consecutive failures before removing a tracker.
    """

    def __init__(self, max_frames_to_track: int = config.MAX_FRAMES_TO_TRACK,
                 min_area: int = config.MIN_AREA,
                 max_failures: int = config.MAX_FAILURES):
        """
        Initializes the ObjectTracker with default or provided parameters.

        Args:
            max_frames_to_track (int): Maximum frames to track an object.
            min_area (int): Minimum area of the object to start tracking.
            max_failures (int): Maximum number of allowed consecutive tracking failures.
        """
        self.trackers: Dict[int, cv2.Tracker] = {}
        self.object_positions: Dict[int, Deque[Tuple[float, float, float]]] = {}
        self.object_tracks: Dict[int, Deque[Tuple[float, float]]] = {}
        self.object_speeds: Dict[int, np.ndarray] = {}
        self.object_accelerations: Dict[int, np.ndarray] = {}
        self.failures: Dict[int, int] = {}
        self.max_failures: int = max_failures
        self.max_frames_to_track: int = max_frames_to_track
        self.min_area: int = min_area
        self.next_object_id: int = 1  # Next object ID to assign

    def initialize_trackers(self, frame: np.ndarray, contours: List[np.ndarray]) -> None:
        """
        Initializes trackers for detected contours.

        Args:
            frame (np.ndarray): The current frame from the video feed.
            contours (List[np.ndarray]): List of detected contours.
        """
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                tracker = self.create_tracker()
                tracker.init(frame, (x, y, w, h))
                object_id = self.next_object_id
                self.next_object_id += 1
                self.trackers[object_id] = tracker
                self.object_positions[object_id] = deque(maxlen=self.max_frames_to_track)
                self.object_tracks[object_id] = deque(maxlen=self.max_frames_to_track)
                self.failures[object_id] = 0
                self.object_positions[object_id].append((x + w / 2, y + h / 2, self.current_time()))
                logging.info(f"Initialized tracker for object {object_id} at position {(x + w / 2, y + h / 2)}")

    def create_tracker(self) -> cv2.Tracker:
        """
        Creates and returns a tracker object.

        Returns:
            cv2.Tracker: The created tracker object.
        """
        try:
            return cv2.TrackerKCF_create()
        except AttributeError:
            logging.warning("KCF tracker not available. Using MIL tracker as fallback.")
            return cv2.TrackerMIL_create()

    def update_trackers(self, frame: np.ndarray) -> np.ndarray:
        """
        Updates all active trackers with the current frame.

        Args:
            frame (np.ndarray): The current frame from the video feed.

        Returns:
            np.ndarray: The frame with tracking information overlaid.
        """
        remove_ids = []
        for object_id, tracker in self.trackers.items():
            success, box = tracker.update(frame)
            if success:
                self.failures[object_id] = 0
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
                x = int(box[0] + box[2] / 2)
                y = int(box[1] + box[3] / 2)
                self.object_positions[object_id].append((x, y, self.current_time()))
                self.object_tracks[object_id].append((x, y))
                logging.debug(f"Updated tracker for object {object_id} to position {(x, y)}")
            else:
                self.failures[object_id] += 1
                logging.warning(
                    f"Failed to update tracker for object {object_id}, failure count: {self.failures[object_id]}")
                if self.failures[object_id] >= self.max_failures:
                    remove_ids.append(object_id)

        for object_id in remove_ids:
            self.remove_tracker(object_id)

        return frame

    def remove_tracker(self, object_id: int) -> None:
        """
        Removes the tracker for a given object ID due to multiple failures.

        Args:
            object_id (int): The ID of the object to remove the tracker for.
        """
        logging.info(f"Removing tracker for object {object_id} due to multiple failures.")

        if object_id in self.object_positions:
            del self.object_positions[object_id]
        if object_id in self.object_tracks:
            del self.object_tracks[object_id]
        if object_id in self.object_speeds:
            del self.object_speeds[object_id]
        if object_id in self.object_accelerations:
            del self.object_accelerations[object_id]
        if object_id in self.failures:
            del self.failures[object_id]
        if object_id in self.trackers:
            del self.trackers[object_id]

    def smooth_positions(self, positions: Deque[Tuple[float, float, float]], window_size: int = 5) -> Deque[np.ndarray]:
        """
        Smooths the positions using a moving average filter.

        Args:
            positions (Deque[Tuple[float, float, float]]): Deque of positions to smooth.
            window_size (int): Size of the moving window for smoothing.

        Returns:
            Deque[np.ndarray]: Smoothed positions.
        """
        positions_np = np.array(positions)
        smoothed_positions = np.zeros_like(positions_np)

        # Handle case when there are fewer positions than the window size
        if len(positions_np) < window_size:
            window_size = len(positions_np)

        smoothed_positions[:, 0] = np.convolve(positions_np[:, 0], np.ones(window_size) / window_size, mode='same')
        smoothed_positions[:, 1] = np.convolve(positions_np[:, 1], np.ones(window_size) / window_size, mode='same')
        smoothed_positions[:, 2] = positions_np[:, 2]

        return deque(smoothed_positions, maxlen=self.max_frames_to_track)

    def calculate_motion_parameters(self) -> None:
        """
        Calculates speed and acceleration for each tracked object.
        """
        for object_id, positions in self.object_positions.items():
            if len(positions) >= 2:
                positions_np = np.array(positions)
                times = positions_np[:, 2]

                # Apply smoothing to reduce noise
                positions_smoothed = self.smooth_positions(positions)
                positions_smoothed_np = np.array(positions_smoothed)

                # Calculate displacements between consecutive positions
                displacements = np.linalg.norm(np.diff(positions_smoothed_np[:, :2], axis=0), axis=1)

                # Set a threshold for detecting motion (adjust as needed)
                motion_threshold = config.MOTION_THRESHOLD_PIXELS

                # Check if the object is static
                if np.all(displacements < motion_threshold):
                    velocities = np.zeros((len(positions) - 1, 2))
                    accelerations = np.zeros((len(positions) - 2, 2))
                else:
                    time_diffs = np.diff(times)
                    if np.any(time_diffs == 0):
                        logging.warning(
                            f"Division by zero encountered for object {object_id}. Skipping velocity calculation.")
                        velocities = np.zeros((len(positions) - 1, 2))
                    else:
                        velocities = np.diff(positions_smoothed_np[:, :2], axis=0) / time_diffs.reshape(-1, 1)

                    if len(times) > 2:
                        time_diffs_acc = np.diff(times[:-1])
                        if np.any(time_diffs_acc == 0):
                            logging.warning(
                                f"Division by zero encountered for object {object_id}. Skipping acceleration calculation.")
                            accelerations = np.zeros((len(positions) - 2, 2))
                        else:
                            accelerations = np.diff(velocities, axis=0) / time_diffs_acc.reshape(-1, 1)
                    else:
                        accelerations = np.zeros((len(positions) - 2, 2))

                # Convert velocities to feet per second and distances to inches
                velocities *= config.PIXELS_TO_FEET
                positions_converted = positions_smoothed_np.copy()
                positions_converted[:, :2] *= config.PIXELS_TO_INCHES

                # Check for invalid values
                velocities = np.where(np.isfinite(velocities), velocities, 0)
                accelerations = np.where(np.isfinite(accelerations), accelerations, 0)
                positions_converted = np.where(np.isfinite(positions_converted), positions_converted, 0)

                self.object_speeds[object_id] = velocities
                self.object_accelerations[object_id] = accelerations
                self.object_positions[object_id] = deque(positions_converted, maxlen=self.max_frames_to_track)

                logging.debug(f"Calculated velocities for object {object_id}: {velocities}")
                logging.debug(f"Calculated accelerations for object {object_id}: {accelerations}")

    def display_data(self, frame: np.ndarray) -> np.ndarray:
        """
        Displays tracking data on the frame.

        Args:
            frame (np.ndarray): The current frame from the video feed.

        Returns:
            np.ndarray: The frame with tracking data overlaid.
        """
        for object_id, positions in self.object_positions.items():
            if len(positions) >= 2:
                positions_np = np.array(positions)
                latest_position = positions_np[-1]
                x, y = round(latest_position[0]), round(latest_position[1])
                cv2.putText(frame, f"ID: {object_id}", (
                    self.clamp(x, 0, frame.shape[1]), self.clamp(y - 20, 0, frame.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if object_id in self.object_speeds and len(self.object_speeds[object_id]) > 0:
                    speed = np.linalg.norm(self.object_speeds[object_id][-1])
                    cv2.putText(frame, f"Speed: {speed:.2f} ft/s", (
                        self.clamp(x, 0, frame.shape[1]), self.clamp(y, 0, frame.shape[0])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if object_id in self.object_accelerations and len(self.object_accelerations[object_id]) > 0:
                    acceleration = np.linalg.norm(self.object_accelerations[object_id][-1])
                    cv2.putText(frame, f"Acc: {acceleration:.2f} ft/s^2",
                                (self.clamp(x, 0, frame.shape[1]), self.clamp(y + 20, 0, frame.shape[0])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Draw trajectory
                for j in range(1, len(positions)):
                    pt1 = (self.clamp(round(positions[j - 1][0]), 0, frame.shape[1]),
                           self.clamp(round(positions[j - 1][1]), 0, frame.shape[0]))
                    pt2 = (self.clamp(round(positions[j][0]), 0, frame.shape[1]),
                           self.clamp(round(positions[j][1]), 0, frame.shape[0]))
                    logging.debug(f"Drawing line from {pt1} to {pt2}")
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        return frame

    @staticmethod
    def clamp(value: float, min_value: float, max_value: float) -> float:
        """
        Clamps a value within the specified range.

        Args:
            value (float): The value to clamp.
            min_value (float): The minimum allowable value.
            max_value (float): The maximum allowable value.

        Returns:
            float: The clamped value.
        """
        return max(min_value, min(value, max_value))

    @staticmethod
    def current_time() -> float:
        """
        Gets the current time in seconds.

        Returns:
            float: The current time in seconds.
        """
        return cv2.getTickCount() / cv2.getTickFrequency()
