# main.py
import logging
import cv2
import matplotlib.pyplot as plt

from object_tracker import ObjectTracker
from video_capture import VideoCapture

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    """
    Main function to initialize and run the video capture system.
    """
    try:
        tracker = ObjectTracker()
        video_capture = VideoCapture(source=1, tracker=tracker)
        video_capture.run()
    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except cv2.error as e:
        logging.error(f"OpenCV Error: {e}")
    except KeyboardInterrupt:
        logging.info("Program interrupted by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        cv2.destroyAllWindows()
        plt.close('all')


if __name__ == "__main__":
    main()
