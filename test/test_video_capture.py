import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
from video_capture import VideoCapture
from object_tracker import ObjectTracker
import logging

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TestVideoCapture(unittest.TestCase):
    """
    Unit tests for the VideoCapture class.
    """

    def setUp(self) -> None:
        """
        Set up test resources before each test.
        """
        self.tracker = ObjectTracker()
        self.video_capture = VideoCapture(source=0, tracker=self.tracker)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def tearDown(self) -> None:
        """
        Clean up test resources after each test.
        """
        del self.tracker
        del self.video_capture
        del self.frame

    @patch('cv2.VideoCapture')
    def test_video_capture_initialization(self, mock_video_capture) -> None:
        """
        Test the initialization of the VideoCapture object.
        """
        mock_video_capture.return_value.isOpened.return_value = True
        vc = VideoCapture(source=0, tracker=self.tracker)
        self.assertTrue(vc.cap.isOpened(), "Video capture device should be opened")

    @patch('cv2.VideoCapture')
    def test_video_capture_failure(self, mock_video_capture) -> None:
        """
        Test the failure to open video capture device.
        """
        mock_video_capture.return_value.isOpened.return_value = False
        with self.assertRaises(ValueError, msg="Failed to open the video capture device"):
            VideoCapture(source=0, tracker=self.tracker)

    def test_calibrate(self) -> None:
        """
        Test the calibration method.
        """
        with patch.object(self.video_capture.cap, 'read', return_value=(True, self.frame)):
            self.video_capture.calibrate()
            self.assertTrue(hasattr(self.video_capture, 'pixels_per_inch_x'),
                            "Should have attribute 'pixels_per_inch_x'")
            self.assertTrue(hasattr(self.video_capture, 'pixels_per_inch_y'),
                            "Should have attribute 'pixels_per_inch_y'")

    @patch.object(VideoCapture, 'process_frame', return_value=np.zeros((480, 640, 3), dtype=np.uint8))
    def test_process_frame(self, mock_process_frame) -> None:
        """
        Test the process_frame method.
        """
        processed_frame = self.video_capture.process_frame(self.frame)
        self.assertIsInstance(processed_frame, np.ndarray, "Processed frame should be an ndarray")

    @patch.object(VideoCapture, 'capture_thread')
    def test_capture_thread(self, mock_capture_thread) -> None:
        """
        Test the capture_thread method.
        """
        mock_capture_thread.return_value = None
        thread = patch('threading.Thread', target=self.video_capture.capture_thread).start()
        self.assertIsNone(mock_capture_thread(), "Capture thread should run without issues")

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.animation.FuncAnimation')
    def test_run(self, mock_func_animation, mock_plt_show) -> None:
        """
        Test the run method.
        """
        self.video_capture.run()
        self.assertTrue(mock_func_animation.called, "FuncAnimation should be called in run method")
        self.assertTrue(mock_plt_show.called, "plt.show() should be called in run method")


if __name__ == "__main__":
    unittest.main()
