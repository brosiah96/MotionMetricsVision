# test_main.py
import logging
import unittest
from unittest.mock import patch

import main

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TestMain(unittest.TestCase):
    """
    Unit tests for the main module.
    """

    @patch('main.VideoCapture')
    @patch('main.ObjectTracker')
    def test_main_success(self, mock_object_tracker, mock_video_capture) -> None:
        """
        Test the main function runs without errors.
        """
        mock_video_capture_instance = mock_video_capture.return_value
        mock_video_capture_instance.run.return_value = None

        with patch('main.cv2.destroyAllWindows') as mock_destroy_all_windows, \
                patch('main.plt.close') as mock_plt_close:
            main.main()
            mock_video_capture.assert_called_once()
            mock_object_tracker.assert_called_once()
            mock_video_capture_instance.run.assert_called_once()
            mock_destroy_all_windows.assert_called_once()
            mock_plt_close.assert_called_once()

    @patch('main.VideoCapture', side_effect=ValueError("Failed to open the video capture device"))
    def test_main_value_error(self, mock_video_capture) -> None:
        """
        Test the main function handles ValueError.
        """
        with patch('main.logging.error') as mock_logging_error, \
                patch('main.cv2.destroyAllWindows'), \
                patch('main.plt.close'):
            main.main()
            mock_video_capture.assert_called_once()
            mock_logging_error.assert_called_with("ValueError: Failed to open the video capture device")

    @patch('main.VideoCapture', side_effect=KeyboardInterrupt)
    def test_main_keyboard_interrupt(self, mock_video_capture) -> None:
        """
        Test the main function handles KeyboardInterrupt.
        """
        with patch('main.logging.info') as mock_logging_info, \
                patch('main.cv2.destroyAllWindows'), \
                patch('main.plt.close'):
            main.main()
            mock_video_capture.assert_called_once()
            mock_logging_info.assert_called_with("Program interrupted by user.")

    @patch('main.VideoCapture', side_effect=Exception("Unexpected error"))
    def test_main_unexpected_error(self, mock_video_capture) -> None:
        """
        Test the main function handles unexpected exceptions.
        """
        with patch('main.logging.error') as mock_logging_error, \
                patch('main.cv2.destroyAllWindows'), \
                patch('main.plt.close'):
            main.main()
            mock_video_capture.assert_called_once()
            mock_logging_error.assert_called_with("Unexpected error: Unexpected error")


if __name__ == "__main__":
    unittest.main()
