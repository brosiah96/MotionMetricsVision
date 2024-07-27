import unittest
import numpy as np
import cv2
from object_tracker import ObjectTracker
import logging
from typing import List, Tuple

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TestObjectTracker(unittest.TestCase):
    """
    Unit tests for the ObjectTracker class.
    """

    def setUp(self) -> None:
        """
        Set up test resources before each test.
        """
        self.tracker = ObjectTracker()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.contours = self._generate_contours([(100, 100, 50, 50), (200, 200, 50, 50)])

    def tearDown(self) -> None:
        """
        Clean up test resources after each test.
        """
        del self.tracker
        del self.frame
        del self.contours

    @staticmethod
    def _generate_contours(boxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Generate contours based on a list of bounding boxes.

        Args:
            boxes (List[Tuple[int, int, int, int]]): List of bounding boxes (x, y, w, h).

        Returns:
            List[np.ndarray]: List of contours.
        """
        contours = []
        for box in boxes:
            x, y, w, h = box
            contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            contours.append(contour)
        return contours

    def test_initialize_trackers(self) -> None:
        """
        Test initializing trackers with valid contours.
        """
        self.tracker.initialize_trackers(self.frame, self.contours)
        self.assertEqual(len(self.tracker.trackers), 2, "Should initialize 2 trackers")

    def test_update_trackers(self) -> None:
        """
        Test updating trackers with a frame.
        """
        self.tracker.initialize_trackers(self.frame, self.contours)
        updated_frame = self.tracker.update_trackers(self.frame)
        self.assertIsInstance(updated_frame, np.ndarray, "Updated frame should be an ndarray")

    def test_remove_tracker(self) -> None:
        """
        Test removing a tracker.
        """
        self.tracker.initialize_trackers(self.frame, self.contours)
        initial_tracker_count = len(self.tracker.trackers)
        self.tracker.remove_tracker(1)
        self.assertEqual(len(self.tracker.trackers), initial_tracker_count - 1, "Should remove 1 tracker")

    def test_smooth_positions(self) -> None:
        """
        Test smoothing positions.
        """
        positions = deque([(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)], maxlen=4)
        smoothed_positions = self.tracker.smooth_positions(positions, window_size=3)
        self.assertEqual(len(smoothed_positions), 4, "Smoothed positions should have the same length")

    def test_calculate_motion_parameters(self) -> None:
        """
        Test calculating motion parameters.
        """
        self.tracker.initialize_trackers(self.frame, self.contours)
        self.tracker.update_trackers(self.frame)
        self.tracker.calculate_motion_parameters()
        self.assertTrue(self.tracker.object_speeds, "Object speeds should be calculated")
        self.assertTrue(self.tracker.object_accelerations, "Object accelerations should be calculated")

    def test_display_data(self) -> None:
        """
        Test displaying tracking data on the frame.
        """
        self.tracker.initialize_trackers(self.frame, self.contours)
        updated_frame = self.tracker.display_data(self.frame)
        self.assertIsInstance(updated_frame, np.ndarray, "Updated frame with display data should be an ndarray")


if __name__ == "__main__":
    unittest.main()
